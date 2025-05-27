import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW  # Import optimizer
from streambp.stream_model import StreamModel, global_dict
import time
import argparse
import logging
from deepspeed import comm as dist
from deepspeed.utils import logger

logger.setLevel(logging.WARNING)

VOCAB_SIZE = 128256
MAX_PAD_RATIO = 0.2

torch.set_printoptions(precision=8)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--chunk_size", type=int, default=2000)
parser.add_argument("--mode", type=str, default="stream")
parser.add_argument("--seq_len", type=int, default=5000)
parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="Model to use for training")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--iterations", type=int, default=2, help="Iterations")
parser.add_argument("--local_rank", type=int, default=0, help="Just a placeholder, the actual rank is determined by deepspeed")
parser.add_argument("--zero_stage", type=int, default=2, help="Zero stage")
parser.add_argument("--master_addr", type=str, default="127.0.0.1", help="Master address")
parser.add_argument("--master_port", type=int, default=29500, help="Master port")

args = parser.parse_args()

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)

if args.mode == "stream":
    model = StreamModel(model, gradient_accumulation_steps=1, logits_chunk_size=100, checkpoint_chunk_size=args.chunk_size, stream_checkpoint=True)

model.train()
model.gradient_checkpointing_enable()

# Create DeepSpeed config dictionary
ds_config = {
    # "train_batch_size": 2,
    "train_micro_batch_size_per_gpu": args.batch_size,
    "gradient_accumulation_steps": args.iterations,
    "zero_optimization": {
        "stage": args.zero_stage
    },
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": False # by default, use bf16 mixed precision training
    }
}

# Define optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Initialize DeepSpeed engine with optimizer
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    optimizer=optimizer,
    config=ds_config
)

rank = dist.get_rank()
world_size = dist.get_world_size()

# generate data
input_ids = torch.randint(0, VOCAB_SIZE, (world_size, args.batch_size, args.seq_len))

local_input_ids = input_ids[rank]
local_attention_mask = torch.ones_like(local_input_ids)
for mask in local_attention_mask:
    mask[-torch.randint(1, int(args.seq_len*MAX_PAD_RATIO), (1,)):] = 0

local_labels = local_input_ids.clone()
local_labels[local_attention_mask == 0] = -100

local_input_ids = local_input_ids.to(model.device)
local_attention_mask = local_attention_mask.to(model.device)
local_labels = local_labels.to(model.device)

torch.cuda.synchronize()
t1 = time.perf_counter()

# get the reference of the optimizer in the fused_backward_model.py
if isinstance(model, StreamModel) and model.stream_checkpoint and args.zero_stage >= 2:
    global_dict["zero2_optimizer"] = optimizer

for i in range(args.iterations):
    outputs = model_engine(input_ids=local_input_ids, labels=local_labels, attention_mask=local_attention_mask)
    loss = outputs.loss
    if loss.requires_grad:
        # for default backward
        model_engine.backward(loss)
    else:
        # for stream model; gradients are already computed during the forward pass
        model_engine._backward_epilogue() # reduce and release the ipg grads

    causal_model = model_engine.model if args.mode == "stream" else model_engine
    lm_head_grad = deepspeed.utils.safe_get_full_grad(causal_model.lm_head.weight)
    q_grad = deepspeed.utils.safe_get_full_grad(causal_model.model.layers[0].self_attn.q_proj.weight)
    if rank == 0:
        print("========== step", i, "==========")
        print(lm_head_grad[:5, :5])
        print(q_grad[:5, :5])
    # need to use engine.step for correctly preparing the averaged gradients
    model_engine.step()

torch.cuda.synchronize()
total_time = time.perf_counter() - t1
per_sample_time = total_time / (args.batch_size * args.iterations)

print(f"=========SEQLEN:{args.seq_len}, MODE:{args.mode}, GPUNUM:{world_size}=========")
print(f"Process {rank} Time taken: {total_time}")

print(f"Process {rank} allocated: {torch.cuda.memory_allocated() / 2**30} max allocated: {torch.cuda.max_memory_allocated() / 2**30}")
