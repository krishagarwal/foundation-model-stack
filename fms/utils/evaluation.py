from typing import List, Tuple

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance  # type: ignore
from lm_eval.api.model import LM  # type: ignore
from lm_eval.api.registry import register_model  # type: ignore
from torch import nn

from fms.utils import tokenizers


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

        continuation_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(continuation)
        )
        input_ids = context_ids + continuation_ids[:-1]
        input_len = len(input_ids)
        input_ids = torch.tensor(
            input_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        i_range = torch.arange(0, input_len).view(-1, 1)
        j_range = torch.arange(0, input_len).view(1, -1)
        mask = torch.where((j_range <= i_range) & (i_range < j_range + self.wrapped_model.config.max_expected_seq_len), 0, -torch.inf).to(input_ids.device)
        mask = mask.broadcast_to(1, 32, input_len, input_len)

        logits = F.log_softmax(self.wrapped_model(input_ids, mask=mask)[0], -1)
        continuation_probs = logits[len(context_ids) - 1 :].cpu()
        loglikelihood = torch.gather(
            continuation_probs, 1, torch.tensor(continuation_ids).unsqueeze(1)
        ).squeeze()
        predicted = torch.argmax(continuation_probs, -1).tolist()
        greedy = predicted == continuation_ids
        return loglikelihood.sum().cpu().item(), greedy

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        result = []
        for request in requests:
            context, continuation = request.args
            result.append(self.loglikelihood_one(context, continuation))
        return result

    def loglikelihood_rolling(
            self, requests: List[Instance]
        ) -> List[Tuple[float, bool]]:
            # TODO: check implementation
            result = []
            for request in requests:
                continuation = request.args[0]
                result.append(self.loglikelihood_one("<|endoftext|>", continuation)[0])
            return result
            raise NotImplementedError("not implemented yet")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("not implemented yet")
