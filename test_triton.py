# 这是附加题模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现triton的深度学习推理过程，请严格保持输出格式输出
import os
import h5py
import time
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import triton
import triton.language as tl

num_class = 10

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
        buffer_path = os.path.join(directory, keystr + f'{name}.txt')
        if os.path.exists(buffer_path):
            # 加载并重塑为缓冲区原始形状
            loaded_buffer = np.loadtxt(buffer_path).reshape(buffer.shape)
            buffer.data.copy_(torch.tensor(loaded_buffer, dtype=buffer.dtype))
        else:
            print(f"Warning: Buffer file '{buffer_path}' not found.")


# def get_cuda_autotune_config():
#     return [
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
#                       num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
#                       num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
#                       num_warps=2),
#     ]

# @triton.autotune(
#     configs=get_cuda_autotune_config(),
#     key=['M', 'N', 'K']
# )
# @triton.jit 
# def matmul_kernel(
#     a_ptr, b_ptr, c_ptr, 
#     M, N, K, 
#     stride_am, stride_ak,
#     stride_bk, stride_bn,
#     stride_cm, stride_cn,
#     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, 
#     GROUP_SIZE_M: tl.constexpr,
# ):
#     # 每个program计算C的一个BLOCK: [BLOCK_SIZE_M, BLOCK_SIZE_N]
#     pid = tl.program_id(axis=0)
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#     num_pid_in_group = GROUP_SIZE_M * num_pid_n # 
#     group_id = pid // num_pid_in_group 
#     first_pid_m = group_id * GROUP_SIZE_M #
#     group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M) #
#     pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
#     pid_n = (pid % num_pid_in_group) // group_size_m

#     offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M 
#     offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N 
#     offs_k = tl.arange(0, BLOCK_SIZE_K)
#     a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak 
#     b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn 

#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
#         b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

#         accumulator = tl.dot(a, b, accumulator)

#         a_ptrs += BLOCK_SIZE_K * stride_ak 
#         b_ptrs += BLOCK_SIZE_K * stride_bk 

#     c = accumulator.to(tl.float16)

#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
#     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#     tl.store(c_ptrs, c, mask=c_mask)

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

# @triton.jit 
# def add_kernel(
#     a_ptr, b_ptr, c_ptr,
#     M, N, 
#     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
# ):
#     # 每个program处理c的BLOCK_SIZE_M行。保证BLOCK_SIZE_N >= N
#     pid = tl.program_id(axis=0)
    
#     offsets_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offsets_n = tl.arange(0, BLOCK_SIZE_N)
#     a_ptrs = a_ptr + offsets_m[:, None] * N + offsets_n[None, :]
#     b_ptrs = b_ptr + offsets_n 
    
#     a = tl.load(a_ptrs, mask=(offsets_m < M)[:, None] & (offsets_n < N))
#     b = tl.load(b_ptrs, mask=offsets_n < N)[None, :]
#     c = a + b 

#     c_ptrs = c_ptr + offsets_m[:, None] * N + offsets_n[None, :]
#     tl.store(c_ptrs, c, mask=(offsets_m < M)[:, None] & (offsets_n < N))    

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



@triton.jit 
def batchnorm_kernel(
    x_ptr, m_ptr, v_ptr, w_ptr, b_ptr, y_ptr,
    M, N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr 
):
    # 每个program处理一个[BLOCK_SIZE_M, BLOCK_SIZE_N]的块
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    x_ptrs = x_ptr + offsets_m[:, None] * N + offsets_n[None, :] 
    m_ptrs = m_ptr + offsets_n
    v_ptrs = v_ptr + offsets_n
    w_ptrs = w_ptr + offsets_n 
    b_ptrs = b_ptr + offsets_n
    y_ptrs = y_ptr + offsets_m[:, None] * N + offsets_n[None, :] 

    x = tl.load(x_ptrs, mask=(offsets_m < M)[:, None] & (offsets_n < N))
    m = tl.load(m_ptrs, mask=offsets_n < N)
    v = tl.load(v_ptrs, mask=offsets_n < N)
    w = tl.load(w_ptrs, mask=offsets_n < N)
    b = tl.load(b_ptrs, mask=offsets_n < N)

    eps = 1e-5
    norm = (x - m[None, :]) / tl.sqrt(v + eps)
    y = norm * w + b 
    
    tl.store(y_ptrs, y, mask=(offsets_m < M)[:, None] & (offsets_n < N))

    
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 32 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32 , 'BLOCK_SIZE_N': 64 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit 
def bmm_kernel(
    a_ptr, b_ptr, c_ptr, 
    M, N, K, 
    stride_al, stride_am, stride_ak,
    stride_bl, stride_bk, stride_bn,
    stride_cl, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr, 
    GROUP_SIZE_M: tl.constexpr,
):
    pid_batch = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n 
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak + pid_batch * stride_al)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn + pid_batch * stride_bl)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak 
        b_ptrs += BLOCK_SIZE_K * stride_bk 
    
    c = acc.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn + pid_batch * stride_cl
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)

@triton.jit 
def log_softmax_kernel(
    i_ptr, o_ptr, 
    M, N, 
    i_row_stride, o_row_stride, 
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr
):
    row_start = tl.program_id(axis=0)
    row_step = tl.num_programs(axis=0)

    for row_idx in tl.range(row_start, M, row_step, num_stages=num_stages):
        row_start_ptr = i_ptr + row_idx * i_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N 
        i_ptrs = row_start_ptr + col_offsets

        row = tl.load(i_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        o = tl.log(numerator / denominator)

        o_row_start_ptr = o_ptr + row_idx * o_row_stride
        o_ptrs = o_row_start_ptr + col_offsets
        tl.store(o_ptrs, o, mask=mask)


# 有一定精度损失，精度能达到0.1
def linear(x, w, b):
    assert x.shape[1] == w.shape[1], "Incompatible dimensions"
    dout, din = w.shape 
    bs, din = x.shape

    w = w.transpose(0, 1)
    x_w = torch.empty((bs, dout), dtype=torch.float16, device=x.device)
    grid = lambda meta: (triton.cdiv(bs, meta["BLOCK_SIZE_M"]) * triton.cdiv(dout, meta["BLOCK_SIZE_N"]), )
    matmul_kernel[grid](
        x, w, x_w, 
        bs, dout, din,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        x_w.stride(0), x_w.stride(1)
    )
    
    y = torch.empty((bs, dout), dtype=torch.float16, device=x.device)
    grid = lambda meta: (triton.cdiv(bs, meta["BLOCK_SIZE_M"]), )
    add_kernel[grid](
        x_w, b, y, 
        bs, dout, 
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=triton.next_power_of_2(dout)
    )
    return y

# def conv1d(x, w, b):
#     assert x.shape[1] == w.shape[1], "Incompatible dimensions"
#     dout, din = w.shape
#     bs, din, d = x.shape 

#     x = x.transpose(2, 1).reshape(-1, din)
#     w = w.transpose(0, 1)
#     x_w = torch.empty((bs * d, dout), dtype=torch.float16, device=x.device)
#     grid = lambda meta: (triton.cdiv(bs * d, meta["BLOCK_SIZE_M"]) * triton.cdiv(dout, meta["BLOCK_SIZE_N"]),)
#     matmul_kernel[grid](
#         x, w, x_w, 
#         bs * d, dout, din, 
#         x.stride(0), x.stride(1),
#         w.stride(0), w.stride(1),
#         x_w.stride(0), x_w.stride(1)
#     )

#     y = torch.empty_like(x_w)
#     grid = lambda meta: (triton.cdiv(bs * d, meta["BLOCK_SIZE_M"]),)
#     add_kernel[grid](
#         x_w, b, y, 
#         bs * d, dout, 
#         BLOCK_SIZE_M=32, BLOCK_SIZE_N=triton.next_power_of_2(dout)
#     )
#     y = y.view(bs, d, -1).transpose(2, 1)
#     return y 
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
 
def batchnorm1d(x, m, v, w, b, relu=True):
    if len(x.shape) == 3:
        C = x.shape[1]
        nx = x.transpose(2, 1).reshape(-1, C)
    else: 
        nx = x 
    assert nx.is_contiguous(), "Matrix x must be contiguous"
    M, N = nx.shape 

    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]), triton.cdiv(N, meta["BLOCK_SIZE_N"]) )
    batchnorm_kernel[grid](
        nx, m, v, w, b, y, 
        M, N, 
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32
    )
    if len(x.shape) == 3:
        y = y.reshape(x.shape[0], x.shape[2], -1).transpose(2, 1)
    if relu:
        y = y * (y > 0)
    return y 

def bmm(a, b):
    assert a.shape[0] == b.shape[0], "Batchsize Mismatch"
    assert a.shape[2] == b.shape[1], "Mismatch"
    B, M, K = a.shape
    B, K, N = b.shape

    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (B, triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']))

    bmm_kernel[grid](
        a, b, c, 
        M, N, K, 
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
    )
    return c 

def log_softmax(x):
    M, N = x.shape 

    y = torch.empty_like(x)
    log_softmax_kernel[(M, )](
        x, y, 
        M, N, 
        x.stride(0), y.stride(0),
        BLOCK_SIZE=triton.next_power_of_2(N),
        num_stages=4
    )
    return y 


class Linear(nn.Module):
    def __init__(self, din, dout,params, keystr):
        super(Linear, self).__init__()
        self.din = din 
        self.dout = dout 
        # 加载参数
        self.weight = torch.tensor(params[keystr + ".weight"], dtype=torch.float16, device="cuda").reshape(dout, din)
        self.bias = torch.tensor(params[keystr + ".bias"], dtype=torch.float16, device="cuda").reshape(dout)

    def forward(self, x):
        return linear(x, self.weight, self.bias)


# BatchNorm1d + ReLU
class BatchNorm1d(nn.Module):
    def __init__(self, d, params, keystr):
        super(BatchNorm1d, self).__init__()
        self.d = d 
        # 加载参数
        self.mean = torch.tensor(params[keystr + ".running_mean"], dtype=torch.float16, device="cuda").reshape(d)
        self.var = torch.tensor(params[keystr + ".running_var"], dtype=torch.float16, device="cuda").reshape(d)
        self.weight = torch.tensor(params[keystr + ".weight"], dtype=torch.float16, device="cuda").reshape(d)
        self.bias = torch.tensor(params[keystr + ".bias"], dtype=torch.float16, device="cuda").reshape(d)

    def forward(self, x, relu=True):
        return batchnorm1d(x, self.mean, self.var, self.weight, self.bias, relu=relu)
    

class Conv1d(nn.Module):
    def __init__(self, din, dout, params, keystr):
        super(Conv1d, self).__init__()
        self.din = din 
        self.dout = dout 
        # 加载参数
        self.weight = torch.tensor(params[keystr + ".weight"], dtype=torch.float16, device="cuda").reshape(dout, din)
        self.bias = torch.tensor(params[keystr + ".bias"], dtype=torch.float16, device="cuda").reshape(dout, 1)

    def forward(self, x):
        return conv1d(x, self.weight, self.bias)
        

class STNkd(nn.Module):
    def __init__(self, k, params, keystr):
        super(STNkd, self).__init__()
        self.conv1 = Conv1d(k, 32, params=params, keystr=keystr + ".conv1")
        self.conv2 = Conv1d(32, 64, params=params, keystr=keystr + ".conv2")
        self.conv3 = Conv1d(64, 128, params=params, keystr=keystr + ".conv3")
        # self.conv1 = nn.Conv1d(k, 32, 1).half()
        # self.conv2 = nn.Conv1d(32, 64, 1).half()
        # self.conv3 = nn.Conv1d(64, 128, 1).half()
        # load_model_params_and_buffers_from_txt(self.conv1, "./params", keystr + ".conv1.")
        # load_model_params_and_buffers_from_txt(self.conv2, "./params", keystr + ".conv2.")
        # load_model_params_and_buffers_from_txt(self.conv3, "./params", keystr + ".conv3.")
        # self.conv1.cuda()
        # self.conv2.cuda()
        # self.conv3.cuda()

        self.fc1 = Linear(128, 64, params=params, keystr=keystr + ".fc1")
        self.fc2 = Linear(64, 32, params=params, keystr=keystr + ".fc2")
        self.fc3 = Linear(32, k * k, params=params, keystr=keystr + ".fc3")

        # self.bn1 = BatchNorm1d(32, params=params, keystr=keystr + ".bn1")
        # self.bn2 = BatchNorm1d(64, params=params, keystr=keystr + ".bn2")
        # self.bn3 = BatchNorm1d(128, params=params, keystr=keystr + ".bn3")
        self.bn1 = nn.BatchNorm1d(32).half()
        self.bn2 = nn.BatchNorm1d(64).half()
        self.bn3 = nn.BatchNorm1d(128).half()
        load_model_params_and_buffers_from_txt(self.bn1, "./params", keystr + ".bn1.")
        load_model_params_and_buffers_from_txt(self.bn2, "./params", keystr + ".bn2.")
        load_model_params_and_buffers_from_txt(self.bn3, "./params", keystr + ".bn3.")
        self.bn1.cuda()
        self.bn2.cuda()
        self.bn3.cuda()

        self.bn4 = BatchNorm1d(64, params=params, keystr=keystr + ".bn4")
        self.bn5 = BatchNorm1d(32, params=params, keystr=keystr + ".bn5")

        self.k = k

    def forward(self, x):
        batchsize = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # **************************************
        # !!! 重要 !!! 
        # batchnorm 对于 [B, C, L] 的实现有问题，需要修改

        # diff = x1 - x2 
        # print(diff)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 128)
        x = self.bn4(self.fc1(x))
        x = self.bn5(self.fc2(x))
        x = self.fc3(x)

        iden = torch.eye(self.k,dtype=torch.float16, device=x.device).view(1, self.k * self.k).repeat(batchsize, 1)
        x = x + iden 
        x = x.view(-1, self.k, self.k)
        return x 


class PointNetEncoder(nn.Module):
    def __init__(self, params, keystr, global_feat=True, feature_transform=True, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STNkd(channel, params=params, keystr=keystr + ".stn")
        self.conv1 = Conv1d(channel, 16, params=params, keystr=keystr + ".conv1")
        self.conv2 = Conv1d(16, 32, params=params, keystr=keystr + ".conv2")
        self.conv3 = Conv1d(32, 64, params=params, keystr=keystr + ".conv3")
        # self.conv1 = nn.Conv1d(channel, 16, 1).half()
        # self.conv2 = nn.Conv1d(16, 32, 1).half()
        # self.conv3 = nn.Conv1d(32, 64, 1).half()
        # load_model_params_and_buffers_from_txt(self.conv1, "./params", keystr + ".conv1.")
        # load_model_params_and_buffers_from_txt(self.conv2, "./params", keystr + ".conv2.")
        # load_model_params_and_buffers_from_txt(self.conv3, "./params", keystr + ".conv3.")
        # self.conv1.cuda()
        # self.conv2.cuda()
        # self.conv3.cuda()

        # self.bn1 = BatchNorm1d(16, params=params, keystr=keystr + ".bn1")
        # self.bn2 = BatchNorm1d(32, params=params, keystr=keystr + ".bn2")
        # self.bn3 = BatchNorm1d(64, params=params, keystr=keystr + ".bn3")
        self.bn1 = nn.BatchNorm1d(16).half()
        self.bn2 = nn.BatchNorm1d(32).half()
        self.bn3 = nn.BatchNorm1d(64).half()
        load_model_params_and_buffers_from_txt(self.bn1, "./params", keystr + ".bn1.")
        load_model_params_and_buffers_from_txt(self.bn2, "./params", keystr + ".bn2.")
        load_model_params_and_buffers_from_txt(self.bn3, "./params", keystr + ".bn3.")
        self.bn1.cuda()
        self.bn2.cuda()
        self.bn3.cuda()


        self.global_feat = global_feat 
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(16, params=params, keystr=keystr + ".fstn")

    def forward(self, x):
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = bmm(x, trans)
        x = x.transpose(2, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        
        trans_feat = self.fstn(x)
        x = x.transpose(2, 1)
        x = bmm(x, trans_feat)
        x = x.transpose(2, 1)
        
        x = F.relu(self.bn2(self.conv2(x)))
        # x = self.bn3(self.conv3(x), relu=False)
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 64)

        return x, trans, trans_feat 

class PointNet(nn.Module):
    def __init__(self, k, params, keystr=""):
        super(PointNet, self).__init__()
        channel = 3
        self.feat = PointNetEncoder(params=params, keystr=keystr + "feat", global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = Linear(64, 32, params=params, keystr=keystr + "fc1")
        self.fc2 = Linear(32, 16, params=params, keystr=keystr + "fc2")
        self.fc3 = Linear(16, k, params=params, keystr=keystr + "fc3")
        
        self.bn1 = BatchNorm1d(32, params=params, keystr=keystr + "bn1")
        self.bn2 = BatchNorm1d(16, params=params, keystr=keystr + "bn2")

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = self.bn1(self.fc1(x))
        x = self.bn2(self.fc2(x))
        x = self.fc3(x)
        x = log_softmax(x)
        return x, trans_feat 


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

def uniform_sampling_on_points(points, num=64):
    length = len(points)
    sample_indices = np.linspace(0, length - 1, num=num, dtype=int)
    return np.array([points[i] for i in sample_indices])

def read_h5_file(dataPath):
    list_of_points = []
    list_of_labels = []
    with h5py.File(dataPath,"r") as hf:
        for k in hf.keys():
            # list_of_points.append(hf[k]["points"][:].astype(np.float32)) #每个points是（N,3）的二维数组ndarray
            # list_of_points.append(hf[k]["points"][:].astype(np.float32).flatten()) #每个points是N*3的一维ndarray
            list_of_points.append(uniform_sampling_on_points(hf[k]["points"], num=64).astype(np.float16)) #每个points是(N,3)的ndarray
            list_of_labels.append(hf[k].attrs["label"])
    return torch.tensor(np.array(list_of_points), dtype=torch.float16), torch.tensor(np.array(list_of_labels))


def do_inference(list_of_points, list_of_labels, params): #请在本函数下使用triton实现推理操作
    for i,points in enumerate(list_of_points):
        # TODO ...在这里实现利用triton对点云数据进行深度学习的推理过程，当然，你也可以改进for循环以使用batch推理提速...
                # 打印每一帧的数据，仅用于调试！
    
        # print(f"=========Points : {i}=========")
        # # print(f"type {type(points)},shape {points.shape}")
        # point_num = int(len(points) / 3)
        # print(f"point_num={point_num}")
        # for point_i in range(point_num):
        #     print(f"x:{points[point_i*3+0]} y:{points[point_i*3+1]} z:{points[point_i*3+2]}")
        print(f"Label: {list_of_labels[i]}")

    # 示例triton程序
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    # output_triton = add(x, y)
    # output_triton2 = add(x, y)
    # output_triton3 = add(x, y)
    # output_triton4 = add(x, y)

    # print(output_triton)

    accuracy_rate = 0.0001
    return accuracy_rate


def main():
    # dir = os.path.dirname(__file__) # 保存模型参数文件(.txt)的文件夹路径
    dir = "./params"

    # 读取模型参数
    params = read_params(dir)
    # #示例，获取feat.stn.fc3.bias的参数
    # feat_stn_fc3_bias = params["feat.stn.fc3.bias"]
    # print(f"feat.stn.fc3.bias:")
    # for i in feat_stn_fc3_bias:
    #     print(i)

    # 读取训练集数据
    dataPath = "./data/test_point_clouds.h5"
    points, labels = read_h5_file(dataPath)

    # 转移到GPU
    points, labels = points.cuda(), labels.cuda()

    model = PointNet(k=num_class, params=params)
    model.eval()

    print(points.shape)
    # 开始计时
    start = time.time()
    # accuracy_rate = do_inference(list_of_points, list_of_labels, params)
    points = points.transpose(2, 1)
    pred, _ = model(points)

    # 结束计时
    end = time.time()
    ms = end - start

    print(pred.shape)

    pred_choice = pred.data.max(1)[1]

    print(pred_choice)

    correct = pred_choice.eq(labels).cpu().sum()
    accuracy_rate = correct / len(labels)

    print(accuracy_rate)

    exit()

    # 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    print(f"{ms:.4f}:{accuracy_rate:.4f}")

if __name__ == '__main__':
    main()