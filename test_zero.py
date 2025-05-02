import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW  # Import optimizer
from fused_backward_model import StreamModel, global_dict
import time
import argparse
from deepspeed import comm as dist

VOCAB_SIZE = 128256
MAX_PAD_RATIO = 0.2
BATCH_SIZE = 1
ITERATIONS = 2

torch.set_printoptions(precision=8)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--chunk_size", type=int, default=2000)
parser.add_argument("--mode", type=str, default="stream")
parser.add_argument("--seq_len", type=int, default=5000)
parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="Model to use for training")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--use_stream_checkpoint", type=bool, default=True, help="Use stream checkpoint")
parser.add_argument("--local_rank", type=int, default=0, help="Just a placeholder, the actual rank is determined by deepspeed")
parser.add_argument("--zero_stage", type=int, default=2, help="Zero stage")

args = parser.parse_args()

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)

if args.mode == "stream":
    model = StreamModel(model, gradient_accumulation_steps=1, logits_chunk_size=100, checkpoint_chunk_size=args.chunk_size, stream_checkpoint=args.use_stream_checkpoint)

model.train()
model.gradient_checkpointing_enable()

# Create DeepSpeed config dictionary
ds_config = {
    # "train_batch_size": 2,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "zero_optimization": {
        "stage": args.zero_stage
    },
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": True # by default, use bf16 mixed precision training
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
input_ids = torch.randint(0, VOCAB_SIZE, (world_size, BATCH_SIZE, args.seq_len))

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
if args.use_stream_checkpoint:
    global_dict["zero2_optimizer"] = optimizer

for _ in range(ITERATIONS):
    outputs = model_engine(input_ids=local_input_ids, labels=local_labels, attention_mask=local_attention_mask)
    loss = outputs.loss
    if loss.requires_grad:
        # for default backward
        model_engine.backward(loss)
    else:
        # for stream model; gradients are already computed during the forward pass
        model_engine._backward_epilogue() # reduce and release the ipg grads

    # # need to use engine.step for correctly preparing the averaged gradients
    # model_engine.step()

torch.cuda.synchronize()
total_time = time.perf_counter() - t1
print(f"Process {rank} Time taken: {total_time}")

print(f"Process {rank} allocated: {torch.cuda.memory_allocated() / 2**30} max allocated: {torch.cuda.max_memory_allocated() / 2**30}")