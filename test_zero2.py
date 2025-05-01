import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW  # Import optimizer
from fused_backward_model import StreamModel
import time

VOCAB_SIZE = 128256
MAX_PAD_RATIO = 0.2
GRAD_ACCUMULATION_STEPS = 1
BATCH_SIZE = 1
SEQ_LEN = 5000
ITERATIONS = 10
# 4*5 v.s. 1*20
torch.set_printoptions(precision=8)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Load model and tokenizer
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

model = StreamModel(model, gradient_accumulation_steps=1, logits_chunk_size=100, checkpoint_chunk_size=500, stream_checkpoint=False)
model.train()
model.gradient_checkpointing_enable()

# Create DeepSpeed config dictionary
ds_config = {
    # "train_batch_size": 2,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "zero_optimization": {
        "stage": 2
    },
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": True
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

from deepspeed import comm as dist
rank = dist.get_rank()
world_size = dist.get_world_size()

# generate data
input_ids = torch.randint(0, VOCAB_SIZE, (world_size, BATCH_SIZE, SEQ_LEN))

local_input_ids = input_ids[rank]
local_attention_mask = torch.ones_like(local_input_ids)
for mask in local_attention_mask:
    mask[-torch.randint(1, int(SEQ_LEN*MAX_PAD_RATIO), (1,)):] = 0

local_labels = local_input_ids.clone()
local_labels[local_attention_mask == 0] = -100

local_input_ids = local_input_ids.to(model.device)
local_attention_mask = local_attention_mask.to(model.device)
local_labels = local_labels.to(model.device)

torch.cuda.synchronize()
t1 = time.perf_counter()

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
print("Time taken: ", total_time)

print("allocated: ", torch.cuda.memory_allocated() / 2**30, "max allocated: ", torch.cuda.max_memory_allocated() / 2**30)