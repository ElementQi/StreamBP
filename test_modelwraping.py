from fused_backward_model import StreamModel, time_record
from transformers import AutoModelForCausalLM
import torch
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_chunksize", type=int, default=500)
parser.add_argument("--seq_expansion", type=int, default=1)
args = parser.parse_args()

torch.set_printoptions(precision=8)

def set_deterministic(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    # Enhanced deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
    # For PyTorch 2.0+
    # Force deterministic algorithms with warn_only=False to ensure strict enforcement
    torch.use_deterministic_algorithms(True, warn_only=False)
    
    # Set environment variable for CUDA operations
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # # Enable anomaly detection to catch non-deterministic operations
    # torch.autograd.set_detect_anomaly(True)
seed = 42 # any number 
set_deterministic(seed=seed)

def clean_grad(model):
    for param in model.parameters():
        param.grad = None

# USE_STREAM = True
# USE_MINIS = True
# MODE = "minis"
# MODE = "stream"
MODE = ""
RECORD_MEMORY = False
CHECKPOINT_CHUNKSIZE = args.checkpoint_chunksize
gradient_accumulation_steps = 1

# load data
input_ids = torch.load("test_data_model/input_ids.pt")
attention_mask = torch.load("test_data_model/attention_mask.pt")
labels = torch.load("test_data_model/labels.pt")

batch_size = 1
input_ids = torch.cat([input_ids for _ in range(batch_size)], dim=0)
attention_mask = torch.cat([attention_mask for _ in range(batch_size)], dim=0)
labels = torch.cat([labels for _ in range(batch_size)], dim=0)

seq_expansion = args.seq_expansion
input_ids = torch.cat([input_ids for _ in range(seq_expansion)], dim=1)
attention_mask = torch.cat([attention_mask for _ in range(seq_expansion)], dim=1)
labels = torch.cat([labels for _ in range(seq_expansion)], dim=1)

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").bfloat16().to(input_ids.device)
# base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B").to(input_ids.device)
base_model.train()

forward_model = base_model
if MODE == "stream":
    print("using stream model")
    # forward_model = StreamModel(base_model, logits_chunk_size=3000, checkpoint_chunk_size=3000*seq_expansion)
    forward_model = StreamModel(base_model, logits_chunk_size=100, checkpoint_chunk_size=CHECKPOINT_CHUNKSIZE, stream_checkpoint=True)
    forward_model.gradient_checkpointing_enable()
elif MODE == "minis":
    from minis.mini_sequence import minisequence
    forward_model = minisequence(base_model, chunk_size=1000)
else:
    print("using base model with gradient checkpointing")
    base_model.gradient_checkpointing_enable()

if RECORD_MEMORY:
    torch.cuda.memory._record_memory_history(max_entries=1000000)

# forward_model.gradient_checkpointing_enable()
t1 = time.perf_counter()

for i in range(gradient_accumulation_steps):
    output = forward_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False, # TODO: make it more elegant
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True
    )

    if output.loss.requires_grad:
        output.loss.backward()
    
    # clean_grad(forward_model)
    
if RECORD_MEMORY:
    torch.cuda.memory._dump_snapshot(f"test_data_model/memory_record_modelwrap.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

total_time = time.perf_counter() - t1
print("Time taken: ", total_time)
print("allocated: ", torch.cuda.memory_allocated() / 1e9, "max allocated: ", torch.cuda.max_memory_allocated() / 1e9)

print(output.loss)

# Check if file exists and is empty
file_exists = os.path.exists("time_record.csv") and os.path.getsize("time_record.csv") > 0

if MODE == "stream" and time_record:
    get_avg = lambda x: sum(x) / len(x)
    seq_len = input_ids.size(1)
    values = [str(seq_len), str(CHECKPOINT_CHUNKSIZE), str(total_time), str(torch.cuda.max_memory_allocated() / 2**30)] + [str(get_avg(time_record[k])) for k in time_record.keys()]
    
    with open("time_record.csv", "a") as f:
        if not file_exists:
            keys = ["SEQ_LEN", "CHECKPOINT_CHUNKSIZE", "TOTAL_TIME", "MAX_MEMORY"] + list(time_record.keys())
            f.write(",".join(keys) + "\n")
        f.write(",".join(values) + "\n")

else:
    print("No time record data available")


print(forward_model.lm_head.weight.grad[0])
# print(forward_model.model.model.layers[0].self_attn.q_proj.weight.grad[0])
print(forward_model.model.layers[0].self_attn.q_proj.weight.grad[0])