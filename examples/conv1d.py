import torch
import torch.nn as nn 
import triton
import triton.language as tl 
import os 
import numpy as np 


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ]

@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K']
)
@triton.jit 
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, 
    M, N, K, 
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, 
    GROUP_SIZE_M: tl.constexpr,
):
    # 每个program计算C的一个BLOCK: [BLOCK_SIZE_M, BLOCK_SIZE_N]
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n # 
    group_id = pid // num_pid_in_group 
    first_pid_m = group_id * GROUP_SIZE_M #
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M) #
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M 
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N 
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak 
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn 

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        accumulator = tl.dot(a, b, accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak 
        b_ptrs += BLOCK_SIZE_K * stride_bk 

    # if ACTIVATION == "leaky_relu":
    #     accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# @triton.jit 
# def leaky_relu(x):
#     return tl.where(x >= 0, x, 0.01 * x)

@triton.jit 
def add_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, 
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # 每个program处理c的BLOCK_SIZE_M行。保证BLOCK_SIZE_N >= N
    pid = tl.program_id(axis=0)
    
    offsets_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + offsets_m[:, None] * N + offsets_n[None, :]
    b_ptrs = b_ptr + offsets_n 
    
    a = tl.load(a_ptrs, mask=(offsets_m < M)[:, None] & (offsets_n < N))
    b = tl.load(b_ptrs, mask=offsets_n < N)[None, :]
    c = a + b 

    c_ptrs = c_ptr + offsets_m[:, None] * N + offsets_n[None, :]
    tl.store(c_ptrs, c, mask=(offsets_m < M)[:, None] & (offsets_n < N))    


def conv1d(x, w, b):
    assert x.shape[1] == w.shape[1], "Incompatible dimensions"
    dout, din = w.shape
    bs, din, d = x.shape 

    x = x.transpose(2, 1).reshape(-1, din)
    w = w.transpose(0, 1)
    x_w = torch.empty((bs * d, dout), dtype=torch.float16, device=x.device)
    grid = lambda meta: (triton.cdiv(bs * d, meta["BLOCK_SIZE_M"]) * triton.cdiv(dout, meta["BLOCK_SIZE_N"]),)
    matmul_kernel[grid](
        x, w, x_w, 
        bs * d, dout, din, 
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        x_w.stride(0), x_w.stride(1)
    )

    y = torch.empty_like(x_w)
    grid = lambda meta: (triton.cdiv(bs * d, meta["BLOCK_SIZE_M"]),)
    add_kernel[grid](
        x_w, b, y, 
        bs * d, dout, 
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=triton.next_power_of_2(dout)
    )
    y = y.view(bs, d, -1).transpose(2, 1)
    return y 
    

def load_model_params_and_buffers_from_txt(model, directory, keystr):
    # 加载所有参数
    for name, param in model.named_parameters():
        param_path = os.path.join(directory, keystr + f'{name}.txt')
        if os.path.exists(param_path):
            # 加载并重塑为参数原始形状
            loaded_param = np.loadtxt(param_path).reshape(param.shape)
            param.data.copy_(torch.tensor(loaded_param, dtype=param.dtype))
        else:
            print(f"Warning: Parameter file '{param_path}' not found.")

    for name, buffer in model.named_buffers():
        print(name)
        buffer_path = os.path.join(directory, keystr + f'{name}.txt')
        if os.path.exists(buffer_path):
            # 加载并重塑为缓冲区原始形状
            loaded_buffer = np.loadtxt(buffer_path).reshape(buffer.shape)
            buffer.data.copy_(torch.tensor(loaded_buffer, dtype=buffer.dtype))
        else:
            print(f"Warning: Buffer file '{buffer_path}' not found.")


def read_params(dir):
    # 列出所有txt文件
    files = [f for f in os.listdir(dir) if f.endswith('.txt')]
    params = {}
    for fileName in files:
        data = []
        with open(os.path.join(dir,fileName), 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                value = float(line)
                data.append(value)
        modelName = fileName.replace(".txt","")
        params[modelName] = data
    return params

params = read_params("./params")
keystr = "feat.conv3."
dout = 64
din = 32

torch.manual_seed(263)
x = torch.randn((1000, din, 3), device="cuda", dtype=torch.float16)
# w = torch.randn((128, 64), device="cuda", dtype=torch.float16)
# b = torch.randn(128, device="cuda", dtype=torch.float16)
w = torch.tensor(params[keystr + "weight"], dtype=torch.float16, device="cuda").reshape(dout, din)
b = torch.tensor(params[keystr + "bias"], dtype=torch.float16, device="cuda").reshape(dout, 1)

triton_output = conv1d(x, w, b)
layer = torch.nn.Conv1d(din, dout, 1)
load_model_params_and_buffers_from_txt(layer, "./params", keystr)
# layer.weight = torch.nn.Parameter(w[:, :, None]) 
# layer.bias = torch.nn.Parameter(b) 
layer.half().cuda()
layer.eval()
torch_output = layer(x)

print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")
# Bigger tolerance for AMD MI200 devices.
# MI200 devices use reduced precision fp16 and bf16 and flush input and
# output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
rtol = 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

