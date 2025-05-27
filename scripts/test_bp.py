from streambp.stream_model import StreamModel
from transformers import AutoModelForCausalLM
import torch
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--chunk_size", type=int, default=3000)
parser.add_argument("--seq_len", type=int, default=9000)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--mode", type=str, default="stream")
parser.add_argument("--iterations", type=int, default=1)
args = parser.parse_args()

torch.set_printoptions(precision=8)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

VOCAB_SIZE = 128256
MAX_PAD_RATIO = 0.2
MODEL_NAME = "Qwen/Qwen3-4B"

# generate data
input_ids = torch.randint(0, VOCAB_SIZE, (args.batch_size, args.seq_len)).cuda()
attention_mask = torch.ones_like(input_ids).cuda()
for mask in attention_mask:
    mask[-torch.randint(1, int(args.seq_len*MAX_PAD_RATIO), (1,)):] = 0
    mask[:torch.randint(1, int(args.seq_len*MAX_PAD_RATIO), (1,))] = 0
labels = input_ids.clone()
labels[attention_mask == 0] = -100

# base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(input_ids.device)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(input_ids.device)
base_model.train()

forward_model = base_model
if args.mode == "stream":
    print("using stream model")
    forward_model = StreamModel(base_model, gradient_accumulation_steps=args.iterations, logits_chunk_size=100, checkpoint_chunk_size=args.chunk_size, stream_checkpoint=True)
    forward_model.gradient_checkpointing_enable()
elif args.mode == "base":
    print("using base model with gradient checkpointing")
    base_model.gradient_checkpointing_enable()
elif args.mode == "base_no_ckpt":
    print("using base model without gradient checkpointing")
else:
    raise ValueError(f"Invalid mode: {args.mode}")

torch.cuda.synchronize()
t1 = time.perf_counter()

for i in range(args.iterations):
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
    
torch.cuda.synchronize()
total_time = time.perf_counter() - t1
per_sample_time = total_time / (args.batch_size * args.iterations)
print("Time taken: ", total_time)
print("Per sample time taken: ", per_sample_time)
print("allocated: ", torch.cuda.memory_allocated() / 2**30, "max allocated: ", torch.cuda.max_memory_allocated() / 2**30)

print("loss: ", output.loss.item())

# import csv
# with open("paper_results/bp_results_seqlen_time_normcorrected.csv", "a") as f:
#     writer = csv.writer(f)
#     writer.writerow([args.mode, args.chunk_size, args.seq_len, args.batch_size, total_time, per_sample_time, torch.cuda.memory_allocated() / 2**30, torch.cuda.max_memory_allocated() / 2**30, output.loss.item()])

# stream
if hasattr(forward_model.model, "model"):
    causal_model = forward_model.model
# base
else:
    causal_model = forward_model

print("q_proj:")
print(causal_model.model.layers[0].self_attn.q_proj.weight.grad[0])

print("k_proj:")
print(causal_model.model.layers[0].self_attn.k_proj.weight.grad[0])

print("lm_head:")
print(causal_model.lm_head.weight.grad[0])