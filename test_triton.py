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

def conv1d(x, w, b):
    # [B, Cin, L] -> [Cout, Cin] -> [Cout] -> [B, Cout, L]
    assert x.shape[1] == w.shape[1], "Mismatch"
    assert w.shape[0] == b.shape[0], "Mismatch"
    B, Cin, L = x.shape 
    Cout, Cin = w.shape 

    y = torch.empty((B, Cout, L), dtype=torch.float16, device=x.device)
    for bid in range(B):
        tmp = torch.empty((Cout, L), dtype=torch.float16, device=x.device)
        grid = lambda meta: (triton.cdiv(Cout, meta["BLOCK_SIZE_M"]) * triton.cdiv(L, meta["BLOCK_SIZE_N"]), )
        matmul_kernel[grid](
            w, x[bid], tmp,
            Cout, L, Cin,
            w.stride(0), w.stride(1),
            x[bid].stride(0), x[bid].stride(1),
            tmp.stride(0), tmp.stride(1),
        )
        y[bid] = tmp + b
    return y 

def batchnorm1d(x, mean, var, weight, bias, relu=True):
    assert x.shape[1] == mean.shape[0] == var.shape[0] == weight.shape[0] == bias.shape[0], "Mismatch"
    x.contiguous()
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
    
    if relu: 
        y = torch.maximum(y, torch.tensor(0.0, device=x.device))
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

        self.fc1 = Linear(128, 64, params=params, keystr=keystr + ".fc1")
        self.fc2 = Linear(64, 32, params=params, keystr=keystr + ".fc2")
        self.fc3 = Linear(32, k * k, params=params, keystr=keystr + ".fc3")

        self.bn1 = BatchNorm1d(32, params=params, keystr=keystr + ".bn1")
        self.bn2 = BatchNorm1d(64, params=params, keystr=keystr + ".bn2")
        self.bn3 = BatchNorm1d(128, params=params, keystr=keystr + ".bn3")
        self.bn4 = BatchNorm1d(64, params=params, keystr=keystr + ".bn4")
        self.bn5 = BatchNorm1d(32, params=params, keystr=keystr + ".bn5")

        self.k = k

    def forward(self, x):
        batchsize = x.shape[0]
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
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
        self.bn1 = BatchNorm1d(16, params=params, keystr=keystr + ".bn1")
        self.bn2 = BatchNorm1d(32, params=params, keystr=keystr + ".bn2")
        self.bn3 = BatchNorm1d(64, params=params, keystr=keystr + ".bn3")

        self.global_feat = global_feat 
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(16, params=params, keystr=keystr + ".fstn")

    def forward(self, x):
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = bmm(x, trans)
        x = x.transpose(2, 1)
        
        x = self.bn1(self.conv1(x))
        
        trans_feat = self.fstn(x)
        x = x.transpose(2, 1)
        x = bmm(x, trans_feat)
        x = x.transpose(2, 1)
        
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x), relu=False)

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


def main():
    dir = os.path.dirname(__file__) # 保存模型参数文件(.txt)的文件夹路径
    # dir = "./params"

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

    points = points.transpose(2, 1)
    model(points)

    # 开始计时
    start = time.time()
    pred, _ = model(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(labels).cpu().sum()
    accuracy_rate = correct / len(labels)

    # 结束计时
    end = time.time()
    ms = end - start

    # 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    print(f"{ms:.4f}:{accuracy_rate:.4f}")

if __name__ == '__main__':
    main()