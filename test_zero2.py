import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW  # Import optimizer
from fused_backward_model import StreamModel

VOCAB_SIZE = 128256
MAX_PAD_RATIO = 0.2
GRAD_ACCUMULATION_STEPS = 1
BATCH_SIZE = 1
SEQ_LEN = 10000

torch.set_printoptions(precision=8)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# model = StreamModel(model, gradient_accumulation_steps=1, logits_chunk_size=100, checkpoint_chunk_size=500, stream_checkpoint=False)
model.train()
model.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# generate data
input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
attention_mask = torch.ones_like(input_ids)
for mask in attention_mask:
    mask[-torch.randint(1, int(SEQ_LEN*MAX_PAD_RATIO), (1,)):] = 0
labels = input_ids.clone()
labels[attention_mask == 0] = -100

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

input_ids = input_ids.to(model.device)
attention_mask = attention_mask.to(model.device)
labels = labels.to(model.device)

# Forward and backward pass
outputs = model_engine(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
loss = outputs.loss
if loss.requires_grad:
    model_engine.backward(loss)
else:
    model_engine._backward_epilogue()
model_engine.step()

print("allocated: ", torch.cuda.memory_allocated() / 2**30, "max allocated: ", torch.cuda.max_memory_allocated() / 2**30)