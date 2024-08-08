import argparse
import os

from tqdm import tqdm

import lm_eval
import torch
import torch._inductor.config
from lm_eval.utils import make_table
from torch import distributed as dist

from fms.models import get_model
from fms.utils import evaluation, tokenizers

import datasets


"""
Example use:
```
srun -N 1 --gres=gpu:1 --cpus-per-task=12 --mem=128G --unbuffered --gres-flags=enforce-binding  python scripts/eval_harness.py --model_path=~/models/7B-F/ --tokenizer=~/models/tokenizer.model --model_source=meta --tasks=hellaswag --num_fewshot=10

|  Tasks  |Version|Filter|n-shot| Metric |Value |   |Stderr|
|---------|-------|------|-----:|--------|-----:|---|-----:|
|hellaswag|Yaml   |none  |    10|acc     |0.5915|±  |0.0049|
|         |       |none  |    10|acc_norm|0.7713|±  |0.0042|
```
"""


parser = argparse.ArgumentParser(description="Script to evaluate a causal model")
parser.add_argument("--device_type", type=str, default="cuda")
parser.add_argument(
    "--architecture",
    type=str,
    default="llama",
    help="The model architecture to benchmark",
)
parser.add_argument(
    "--variant",
    type=str,
    default="7b",
    help="The model variant (configuration) to benchmark. E.g. 7b, 13b, 70b.",
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the directory containing LLaMa weights (.pth files sharded by tensor parallel rank, not HF weights)",
)
parser.add_argument(
    "--model_source",
    type=str,
    help="Source of the checkpoint. E.g. 'meta', 'hf', None",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    required=True,
    help="Path to the tokenizer (e.g. ~/tokenizer.model)",
)
parser.add_argument(
    "--no_use_cache",
    action="store_false",
    help="Disable the kv-cache (on by default)",
)
parser.add_argument(
    "--compile",
    action="store_true",
    help="Use torch.compile (slow for first inference pass)",
)
parser.add_argument(
    "--compile_mode",
    type=str,
    help="Mode for compilation",
    default="default",
    choices=["default", "reduce-overhead"],
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Set torch.use_deterministic_algorithms? Requires env variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`",
)
parser.add_argument(
    "--distributed",
    action="store_true",
    help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
)
parser.add_argument("--tasks", type=str, help="Task names to pass to lm_eval")
parser.add_argument(
    "--num_fewshot",
    type=int,
    default=None,
    help="Number of examples in few-shot context",
)
parser.add_argument(
    "--quant_dtype",
    type=str,
    help="enables quantization to the specified dtype",
    default="",
    choices=["", "int8", "int4-fake"],
)
parser.add_argument(
    "--activ_clip_ratio",
    type=float,
    help="ratio for scale of activations when quantized (typically <= 1)",
    default=0.9, # TODO: check if setting a good but not None-like default is proper fms style
)
parser.add_argument(
    "--kv_clip_ratio",
    type=float,
    help="ratio for scale of keys and values when quantized for caching (typically <= 1)",
    default=0.95, # TODO: check if setting a good but not None-like default is proper fms style
)
parser.add_argument(
    "--rotate",
    action="store_true",
)

args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
if args.device_type == "cuda":
    device = torch.device(args.device_type, local_rank)
    torch.cuda.set_device(device)
else:
    device = torch.device(args.device_type)

torch.set_default_dtype(torch.half)

# requires setting environment variable: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
if args.deterministic:
    torch.use_deterministic_algorithms(True)

if args.distributed:
    dist.init_process_group()
    # Fix until PT 2.3
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

print("loading model")
if args.distributed:
    distr_param = "tp"
else:
    if torch.cuda.device_count() > 1 and world_size == 1:
        distr_param = "mp"
    else:
        distr_param = None

model = get_model(
    args.architecture,
    args.variant,
    model_path=args.model_path,
    device_type=args.device_type,
    source=args.model_source,
    distributed_strategy=distr_param,
    group=dist.group.WORLD,
    quant_dtype=args.quant_dtype,
    activ_clip_ratio=args.activ_clip_ratio,
    kv_clip_ratio=args.kv_clip_ratio,
    rotate=args.rotate,
)
tokenizer = tokenizers.get_tokenizer(args.tokenizer)
model.eval()
torch.set_grad_enabled(False)
print("loading complete on rank", local_rank)

if args.compile:
    print("compiling model")
    # Bug with kv-cache in PT2.1
    torch._inductor.config.joint_graph_constant_folding = False
    # compiling can make first inference pass slow
    model = torch.compile(model, mode=args.compile_mode)


# lm_obj = evaluation.FMSEvalHarnessLM(model=model, tokenizer=tokenizer, device=device)

# # lm_eval.tasks.initialize_tasks()

# results = lm_eval.simple_evaluate(
#     model=lm_obj,
#     tasks=args.tasks.split(","),
#     num_fewshot=args.num_fewshot,
# )

testdata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
testenc = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("\n\n".join(testdata['text'])))


# adapted to match https://github.com/spcl/QuaRot/blob/main/fake_quant/eval_utils.py
batch_size = 1 #32
dev = args.device_type
seqlen = 2048 #model.config.max_expected_seq_len # QuaRot uses 2048 for some reason

# Convert the whole text of evaluation dataset into batches of sequences.
input_ids = torch.tensor(testenc).view(1, -1)  # (1, text_len)
nsamples = input_ids.numel() // seqlen  # The tail is truncated.
input_ids = input_ids[:, :nsamples * seqlen].view(nsamples, seqlen).to(dev)  # (nsamples, seqlen)

input_ids = [input_ids[i:i + batch_size] for i in range(0, nsamples, batch_size)] # (nbatches, batch_size, seqlen)
nbatches = len(input_ids)

# torch.cuda.empty_cache()

nlls = []
loss_fct = torch.nn.CrossEntropyLoss(reduction = "none")
for i in tqdm(range(nbatches)):
    lm_logits = model(input_ids[i])

    shift_logits = lm_logits[:, :-1, :]
    shift_labels = input_ids[i][:, 1:]
    loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
    neg_log_likelihood = loss.float().mean(dim=1)
    nlls.append(neg_log_likelihood)
nlls_tensor = torch.cat(nlls)
ppl = torch.exp(nlls_tensor.mean())
print("perplexity:", ppl.item())
