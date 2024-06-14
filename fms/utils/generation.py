from typing import Any, Callable, List, MutableMapping, Optional, Union

import torch
import torch.nn.functional as F

from fms.modules.quarot import utils
import pickle


def _make_cache_contiguous(past_key_value_states):
    # kv updates are required for torch.compile with
    # mode='reduce-overhead'
    n_kv_s: List[List[torch.Tensor]] = []
    for layer_idx in range(len(past_key_value_states)):
        n_kv_s.append([])
        for tensor_idx in range(len(past_key_value_states[layer_idx])):
            n_kv_s[layer_idx].append(
                past_key_value_states[layer_idx][tensor_idx]
                .clone(memory_format=torch.contiguous_format)
                .detach()
            )
            # torch._dynamo.mark_dynamic(n_kv_s[layer_idx][tensor_idx], 2)
    return n_kv_s


def generate(
    model: Union[Callable, torch.nn.Module],
    input_ids: torch.Tensor,
    max_seq_len: int = 4096,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 10,
    do_sample: bool = True,
    num_beams: int = 1,
    use_cache: bool = False,
    contiguous_cache: bool = False,
    eos_token_id: Optional[int] = None,
):
    """
    A trivial generate function that can be used for validation/testing in
    cases where HF is not available.
    We could add implementations for other types of generation, but this is
    enough for making sure a model is working.
    Does not implement batching nor beam search, but those could be added.

    Args:
        model: A function or nn.Module that takes a batch of input_ids and
            returns logits
        prefix: A tensor of token IDs.
        max_seq_len: the sequence length of the model
        max_new_tokens: max tokens to generate
        temperature: temperature of softmax when sampling
        top_k: only search among top k tokens
        do_sample: multinomial sampling. False for greedy.
        num_beams: TODO: support beam search
        use_cache: requires that the model accept use_cache and
            past_key_value_states args in forward method.
    """
    batched = False
    if num_beams != 1:
        raise NotImplementedError("generate() does yet not support beam search")
    if type(input_ids) == torch.Tensor:
        if input_ids.dim() != 1:
            batched = True
    else:
        raise RuntimeError("generate() requires a tensor of token ids as the prefix")

    if not batched:
        input_ids = input_ids.unsqueeze(0)

    eos_found = torch.zeros(
        input_ids.shape[0], dtype=torch.bool, device=input_ids.device
    )

    result = input_ids
    next_input = input_ids
    kwargs: MutableMapping[str, Any] = dict()
    kwargs["past_key_value_states"] = None
    kwargs["use_cache"] = use_cache

    # TODO: consider removing or cleaning up (perplexity)
    chosen_probs = []
    softmax_dots = []
    truth_token_ids = []
    truth_softmax = []
    is_truth = utils.qdtype == torch.float16
    if not is_truth:
        with open('correct_token_ids.pickle', 'rb') as f:
            truth_token_ids = pickle.load(f)
            print('loaded truth tokens')
        with open('correct_softmax.pickle', 'rb') as f:
            truth_softmax = pickle.load(f)
            print('loaded truth softmax')

    for i in range(max_new_tokens): # TODO: was _
        input_ids = next_input[:, -max_seq_len:]
        output = model(input_ids, **kwargs)
        if use_cache:
            logits, past_key_value_states = output
            # TODO: this should go away when reduce-overhead issues are fixed, or
            # maybe could be moved into model code to be more portable.
            if contiguous_cache:
                kwargs["past_key_value_states"] = _make_cache_contiguous(
                    past_key_value_states
                )
            else:
                kwargs["past_key_value_states"] = past_key_value_states
        else:
            logits = output
        logits = logits[:, -1, :]

        if do_sample:
            # get logits from last value in sequence nad scale
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_val = torch.multinomial(probs, num_samples=1)
        else:
            # TODO: consider removing or cleaning up (perplexity)
            probs = F.softmax(logits, dim=-1)

            next_val = torch.argmax(logits, dim=-1).unsqueeze(0).t()

        if is_truth:
            truth_token_ids.append(next_val)
            truth_softmax.append(probs.reshape(-1))
        elif utils.test_against_truth:
            next_val = truth_token_ids[i] # TODO: remove. For testing, this keeps testing preplecity on ground truth tokens

        # TODO: consider removing or cleaning up (perplexity)
        chosen_probs.append(probs[0, next_val])
        # print(f"prob {probs[0, next_val]}")
        softmax_dots.append(torch.cosine_similarity(probs.reshape(-1).to(torch.float32), truth_softmax[i].to(torch.float32), dim=0))


        result = torch.cat((result, next_val), dim=-1)

        # avoid continuing to generate if all have reached EOS
        if eos_token_id is not None:
            eos_found = torch.logical_or(eos_found, next_val == eos_token_id)
            if torch.sum(eos_found) == input_ids.shape[0]:
                break

        if use_cache:
            next_input = next_val
        else:
            next_input = result

    chosen_probs = torch.tensor(chosen_probs)
    logs = chosen_probs.log()
    mean = logs.mean()
    perp = (-mean).exp()

    softmax_metric = torch.tensor(softmax_dots).mean()

    print(f"val: {utils.current_float_val}, perplexity: {perp}, softmax metric: {softmax_metric}")
    utils.current_score = softmax_metric

    if is_truth:
        with open('correct_token_ids.pickle', 'wb') as f:
            pickle.dump(truth_token_ids, f)
            print('saved truth tokens')
        with open('correct_softmax.pickle', 'wb') as f:
            pickle.dump(truth_softmax, f)
            print('saved truth softmax')


    if not batched:
        result = result[0]
    return result


def truncate_after_eos(result, eos_token_id):
    """
    Helper function to return a truncated sequence of token IDs stopping at
    (and including) the 'end of sentence' token.
    Currently only handles unbatched sequences.
    """
    if eos_token_id is None:
        return result

    eos_idx = torch.where(result == eos_token_id)
    eos_idx = eos_idx[0]
    if eos_idx.shape[0] >= 1:
        eos_idx = eos_idx[0].item()
        result = result[: eos_idx + 1]
    return result
