import torch 
import triton 
import triton.language as tl 
from triton.runtime import driver 

def naive_softmax(x):
    """
    softmax(xi) = exp(xi) / sum(exp(xj))
    """
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None] # x_max[:, None] 是在None所对应的位置上增加了一个维度，便于广播
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    ret = numerator / denominator[:, None]
    return ret 

@triton.jit 
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride, # row_stride表示按行遍历时需要跳过的元素数量，本质上等于每行元素个数，即列数
    n_rows, n_cols, 
    BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr 
):
    row_start = tl.program_id(axis=0) 
    row_step = tl.num_programs(axis=0) # num_programs()返回启动的program实例的数量，应该是依据grid中的配置，即block的数量

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages): # 每个program处理若干行
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE) # BLOCK_SIZE被设置为大于n_cols的最小的2的幂次，保证一个BLOCK可以装得下一行
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols

        row = tl.load(input_ptrs, mask=mask, other=-float("inf")) # other的作用是在mask掩码无效的位置指定一个备用值。
                                                                  # float("inf") 表示无穷大， -float("inf") 表示负无穷大，
                                                                  # 在最大化过程中使用，确保无效位置的值不会对最大值产生干扰
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator 

        output_row_start_ptr = output_ptr + row_idx * output_row_stride 
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
    
device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

def softmax(x):
    n_rows, n_cols = x.shape 

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 8 

    num_stages = 4 if SIZE_SMEM > 200000 else 2 

    y = torch.empty_like(x)

    # 预编译以获取寄存器使用和线程占用信息，帮助主程序运行提供合理配置
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, 
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    kernel._init_handles() # 提供后续访问kernel的metadata（元信息）的接口
    n_regs = kernel.n_regs # 获取寄存器使用量
    size_smem = kernel.metadata.shared # 获取共享内存使用量 

    #  根据寄存器和共享内存的使用量计算 kernel 的理论硬件占用率
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps) 
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    
    # 计算实际启动program的数量
    num_programs = NUM_SM * occupancy
    num_programs = min(num_programs, n_rows)

    # BLOCK_SIZE和num_stages等静态编译时信息在warmup阶段已经预编译到kernel实例中了
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols) 

    return y 

torch.manual_seed(0)
x = torch.randn(1823, 781, device='cuda') # 生成一个随机数张量
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
    



