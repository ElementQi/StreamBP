import torch
import time

T = 100000
d0, d1, d2 = 2**14, 2**14, 2**14 # 16384
chunk_size = 2000
num_chunks = T // chunk_size
if T % chunk_size != 0:
    num_chunks += 1

X = torch.randn(size=(T, d0)).cuda()
W1 = torch.randn(size=(d0, d1), device=X.device)
W2 = torch.randn(size=(d1, d2), device=X.device)

W1.requires_grad_()
W2.requires_grad_()

t1 = time.time()

# Standard BP
Y = (X @ W1) @ W2
print("standard BP forward memory:", torch.cuda.memory_allocated() / 1e9)
print("standard BP forward max memory:", torch.cuda.max_memory_allocated() / 1e9)
Y.sum().backward()
del Y
torch.cuda.synchronize()
t2 = time.time()
print("time cost:", t2 - t1)
print("standard BP backward max memory:", torch.cuda.max_memory_allocated() / 1e9)
print("standard BP backward memory:", torch.cuda.memory_allocated() / 1e9)

# ## StreamBP
# Y = []
# for i in range(num_chunks):
#     start = i * chunk_size
#     end = min(T, (i + 1) * chunk_size)
#     X_chunk = X[start:end]
#     Y_chunk = (X_chunk @ W1) @ W2
#     Y_chunk.sum().backward()
#     del Y_chunk

# torch.cuda.synchronize()
# t2 = time.time()
# print("chunk size:", chunk_size)
# print("time cost:", t2 - t1)
# print("stream BP backward max memory:", torch.cuda.max_memory_allocated() / 1e9)
# print("stream BP backward memory:", torch.cuda.memory_allocated() / 1e9)
