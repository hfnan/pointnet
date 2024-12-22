import torch 
import torch.nn as nn
import triton
import triton.language as tl 


# @triton.jit 
# def batchnorm_kernel(
#     x_ptr, m_ptr, v_ptr, w_ptr, b_ptr, y_ptr,
#     M, N,
#     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr 
# ):
#     # 每个program处理BLOCK_SIZE_M行
#     pid_m = tl.program_id(axis=0)
#     pid_n = tl.program_id(axis=1)
#     offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

#     x_ptrs = x_ptr + offsets_m[:, None] * N + offsets_n[None, :] 
#     m_ptrs = m_ptr + offsets_n
#     v_ptrs = v_ptr + offsets_n
#     w_ptrs = w_ptr + offsets_n 
#     b_ptrs = b_ptr + offsets_n
#     y_ptrs = y_ptr + offsets_m[:, None] * N + offsets_n[None, :] 

#     x = tl.load(x_ptrs, mask=(offsets_m < M)[:, None] & (offsets_n < N))
#     m = tl.load(m_ptrs, mask=offsets_n < N)
#     v = tl.load(v_ptrs, mask=offsets_n < N)
#     w = tl.load(w_ptrs, mask=offsets_n < N)
#     b = tl.load(b_ptrs, mask=offsets_n < N)

#     eps = 1e-5
#     norm = (x - m) / tl.sqrt(v + eps)
#     y = norm * w + b 
    
#     tl.store(y_ptrs, y, mask=(offsets_m < M)[:, None] & (offsets_n < N))


# def batchnorm1d(x, m, v, w, b):
#     # [B, C, N] -> [C] -> [C] -> [C] -> [C] -> [B, C, N]
#     if len(x.shape) == 3:
#         C = x.shape[1]
#         nx = x.transpose(2, 1).reshape(-1, C)
#     else: 
#         nx = x 
#     assert nx.is_contiguous(), "Matrix x must be contiguous"
#     M, N = nx.shape 

#     y = torch.empty_like(x)
#     grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]), triton.cdiv(N, meta["BLOCK_SIZE_N"]) )
#     batchnorm_kernel[grid](
#         nx, m, v, w, b, y, 
#         M, N, 
#         BLOCK_SIZE_M=32, BLOCK_SIZE_N=32
#     )
#     if len(x.shape) == 3:
#         y = y.reshape(x.shape[0], x.shape[2], -1).transpose(2, 1)
#     return y 

# def torch_batchnorm1d(x, m, v, w, b):
#     eps = 1e-5 
#     norm = (x - m) / torch.sqrt(v + eps)
#     return norm * w + b 

@triton.jit
def batchnorm_kernel(
    x_ptr, y_ptr, 
    mean_ptr, var_ptr, weight_ptr, bias_ptr, 
    B, C, L,
    stride_b, stride_c, stride_l,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr     
):
    cid = tl.program_id(axis=0)
    
    mean = tl.load(mean_ptr + cid, mask=cid < C).to(tl.float32)
    var = tl.load(var_ptr + cid, mask=cid < C).to(tl.float32)
    weight = tl.load(weight_ptr + cid, mask=cid < C).to(tl.float32)
    bias = tl.load(bias_ptr + cid, mask=cid < C).to(tl.float32)

    std_inv = 1.0 / tl.sqrt(var + eps)

    offsets = tl.arange(0, BLOCK_SIZE)
    for b in range(B):
        start_idx = b * stride_b + cid * stride_c + offsets * stride_l 
        mask = offsets < L 
        
        x = tl.load(x_ptr + start_idx, mask=mask, other=0.0).to(tl.float32)
        x_norm = (x - mean) * std_inv 
        y = weight * x_norm + bias 
        
        y.to(tl.float16)
        tl.store(y_ptr + start_idx, y, mask=mask)


def batchnorm(x, mean, var, weight, bias):
    assert x.shape[1] == mean.shape[0] == var.shape[0] == weight.shape[0] == bias.shape[0], "Mismatch"
    if len(x.shape) == 3:
        B, C, L = x.shape 
    elif len(x.shape) == 2:
        B, C = x.shape 
        L = 1

    y = torch.empty_like(x)
    grid = (C, )
    batchnorm_kernel[grid](
        x, y, 
        mean, var, weight, bias, 
        B, C, L, 
        x.stride(0), x.stride(1), 1,
        eps=1e-5, BLOCK_SIZE=triton.next_power_of_2(L)
    )
    return y 


torch.manual_seed(13)
x = torch.randn((1000, 64), device="cuda", dtype=torch.float16)
m = torch.randn(64, device="cuda", dtype=torch.float16)
v = torch.randn(64, device="cuda", dtype=torch.float16)
w = torch.randn(64, device="cuda", dtype=torch.float16)
b = torch.randn(64, device="cuda", dtype=torch.float16)
triton_output = batchnorm(x, m, v, w, b)
# triton_output = torch_batchnorm1d(x, m, v, w, b)

layer = nn.BatchNorm1d(64).half()
layer.running_mean = nn.Parameter(m, requires_grad=False)
layer.running_var = nn.Parameter(v, requires_grad=False)
layer.weight = nn.Parameter(w, requires_grad=False)
layer.bias = nn.Parameter(b, requires_grad=False)
layer.eval()
torch_output = layer(x)
print(triton_output.shape)
print(torch_output.shape)
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")
# Bigger tolerance for AMD MI200 devices.
# MI200 devices use reduced precision fp16 and bf16 and flush input and
# output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
rtol = 0
if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

