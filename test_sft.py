from fused_backward_model import StreamModel, time_record
from transformers import AutoModelForCausalLM
import torch
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--chunk_size", type=int, default=500)
parser.add_argument("--seq_len", type=int, default=3000)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--mode", type=str, default="stream")
args = parser.parse_args()

torch.set_printoptions(precision=8)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

def clean_grad(model):
    for param in model.parameters():
        param.grad = None

RECORD_MEMORY = False
VOCAB_SIZE = 128256
MAX_PAD_RATIO = 0.2
GRAD_ACCUMULATION_STEPS = 2
MODEL_NAME = "Qwen/Qwen2.5-7B"

# generate data
input_ids = torch.randint(0, VOCAB_SIZE, (args.batch_size, args.seq_len)).cuda()
attention_mask = torch.ones_like(input_ids).cuda()
for mask in attention_mask:
    mask[-torch.randint(1, int(args.seq_len*MAX_PAD_RATIO), (1,)):] = 0
labels = input_ids.clone()
labels[attention_mask == 0] = -100

# base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").bfloat16().to(input_ids.device)
# base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(input_ids.device)
# base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-32B").bfloat16().to(input_ids.device)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).bfloat16().to(input_ids.device)
base_model.train()

# def remove_grad_hook(param):
#     def func(x):
#         param.grad = None
#     return func

# for param in base_model.parameters():
#     param.register_post_accumulate_grad_hook(remove_grad_hook(param))

forward_model = base_model
if args.mode == "stream":
    print("using stream model")
    forward_model = StreamModel(base_model, gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS, logits_chunk_size=100, checkpoint_chunk_size=args.chunk_size, stream_checkpoint=True)
    forward_model.gradient_checkpointing_enable()
elif args.mode == "minis":
    print("using minis model")
    from minis.mini_sequence import minisequence
    forward_model = minisequence(base_model, logits_chunk_size=100, chunk_size=args.chunk_size)
elif args.mode == "base":
    print("using base model with gradient checkpointing")
    base_model.gradient_checkpointing_enable()
else:
    raise ValueError(f"Invalid mode: {args.mode}")

if RECORD_MEMORY:
    torch.cuda.memory._record_memory_history(max_entries=1000000)

# forward_model.gradient_checkpointing_enable()
torch.cuda.synchronize()
t1 = time.perf_counter()

for i in range(GRAD_ACCUMULATION_STEPS):
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
    torch.cuda.memory._dump_snapshot(f"test_data_model/memory_{args.mode}_cz{args.chunk_size}_seqlen{args.seq_len}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

torch.cuda.synchronize()
total_time = time.perf_counter() - t1
print("Time taken: ", total_time)
print("Per sample time taken: ", total_time / args.batch_size)
print("allocated: ", torch.cuda.memory_allocated() / 2**30, "max allocated: ", torch.cuda.max_memory_allocated() / 2**30)

print("loss: ", output.loss.item())

# # Check if file exists and is empty
# file_exists = os.path.exists("time_record.csv") and os.path.getsize("time_record.csv") > 0

# if MODE == "stream" and time_record:
#     get_avg = lambda x: sum(x) / len(x)
#     seq_len = input_ids.size(1)
#     values = [str(seq_len), str(CHECKPOINT_CHUNKSIZE), str(total_time), str(torch.cuda.max_memory_allocated() / 2**30)] + [str(get_avg(time_record[k])) for k in time_record.keys()]
    
#     with open("time_record.csv", "a") as f:
#         if not file_exists:
#             keys = ["SEQ_LEN", "CHECKPOINT_CHUNKSIZE", "TOTAL_TIME", "MAX_MEMORY"] + list(time_record.keys())
#             f.write(",".join(keys) + "\n")
#         f.write(",".join(values) + "\n")

# else:
#     print("No time record data available")


# print(forward_model.lm_head.weight.grad[0])

# minis
if hasattr(forward_model, "module"):
    causal_model = forward_model.module
# stream
elif hasattr(forward_model.model, "model"):
    causal_model = forward_model.model
# base
else:
    causal_model = forward_model

# print("q_proj:")
# print(causal_model.model.layers[0].self_attn.q_proj.weight.grad[0])

# print("k_proj:")
# print(causal_model.model.layers[0].self_attn.k_proj.weight.grad[0])

# print("lm_head:")
# print(causal_model.lm_head.weight.grad[0])