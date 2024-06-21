from typing import List, Tuple

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance  # type: ignore
from lm_eval.api.model import LM  # type: ignore
from lm_eval.api.registry import register_model  # type: ignore
from torch import nn

from fms.utils import tokenizers
from tqdm import tqdm


@register_model("fms")
class FMSEvalHarnessLM(LM):
    def __init__(
        self,
        model: nn.Module,
        tokenizer: tokenizers.BaseTokenizer,
        device="cpu",
        rank=0,
        world_size=1,
    ):
        self.wrapped_model = model
        self.tokenizer = tokenizer
        self._rank = rank
        self._world_size = world_size
        self.device = device
        # workaround for https://github.com/EleutherAI/lm-evaluation-harness/issues/1333
        # until the fix is in a release
        generic_object = lambda: None
        self.model = generic_object
        self.model.config = generic_object  # type: ignore
        self.model.config._name_or_path = "FMSEvalHarnessLM"  # type: ignore

    def loglikelihood_one(self, context: str, continuation: str) -> Tuple[float, bool]:
        context_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(context)
        )
        if not len(context_ids):
            context_ids = [self.tokenizer.bos_token_id]
        
        max_len = self.wrapped_model.config.max_expected_seq_len

        continuation_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(continuation)
        )
        input_ids = context_ids + continuation_ids[:-1]
        input_ids = torch.tensor(
            input_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0).cuda()

        curr_cache = None
        logits = None
        for i in range(0, input_ids.shape[1], max_len):
            cache_len = curr_cache[0][0].shape[2] if curr_cache is not None else 0
            if cache_len >= max_len:
                curr_cache = [(layer_keys[:, :, -max_len+1:], layer_vals[:, :, -max_len+1:]) for layer_keys, layer_vals in curr_cache]
                cache_len = curr_cache[0][0].shape[2]
            
            curr_tokens = input_ids[:, i : min(input_ids.shape[1], i + max_len)]
            input_len = curr_tokens.shape[1]

            total_len = cache_len + input_len
            assert total_len < 2 * max_len
            i_range = torch.arange(cache_len, total_len).view(-1, 1)
            j_range = torch.arange(0, total_len).view(1, -1)
            mask = ((j_range <= i_range) & (j_range > i_range - max_len)).to(input_ids.device).broadcast_to(1, 32, input_len, total_len)

            model_out, curr_cache = self.wrapped_model(curr_tokens, use_cache=True, past_key_value_states=curr_cache, mask=mask)
            curr_logits = F.log_softmax(model_out[0], -1).cpu()
            logits = curr_logits if logits is None else torch.cat([logits, curr_logits], dim=0)

        continuation_probs = logits[len(context_ids) - 1:, :]
        loglikelihood = torch.gather(
            continuation_probs, 1, torch.tensor(continuation_ids).unsqueeze(1)
        ).squeeze()
        predicted = torch.argmax(continuation_probs, -1).tolist()
        greedy = predicted == continuation_ids
        return loglikelihood.sum().cpu().item(), greedy

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        result = []
        for request in tqdm(requests):
            context, continuation = request.args
            result.append(self.loglikelihood_one(context, continuation))
        return result

    def loglikelihood_rolling(
        self, requests: List[Instance]
    ) -> List[Tuple[float, bool]]:
        # TODO: check implementation
        result = []
        for request in tqdm(requests):
            continuation = request.args[0]
            result.append(self.loglikelihood_one("", continuation)[0])
        return result
        raise NotImplementedError("not implemented yet")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("not implemented yet")
