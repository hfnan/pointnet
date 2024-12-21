import torch 

import triton
import triton.language as tl 

@triton.jit 
def add_kernel(
    x_ptr, y_ptr, output_ptr, # 向量指针
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE # triton以blcok为基本单元

    offsets = block_start + tl.arange(0, BLOCK_SIZE) # offsets是块内偏移量，是一个list; arange() 生成一系列连续值，左闭右开

    mask = offsets < n_elements # mask用于边界控制

    x = tl.load(x_ptr + offsets ,mask=mask) # load() 返回由指针指向内存（显存）的数据（张量）
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y 

    tl.store(output_ptr + offsets, output, mask=mask) # store() 将数据（张量）存入指针所指向的内存（显存）

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x) # 为输出张量预分配内存（显存），output应该与x在同一种设备上
    assert x.is_cuda and y.is_cuda and output.is_cuda
    
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), ) # meta传递kernel定义时配置的编译时参数，比如BLOCK_SIZE
    
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output 

torch.manual_seed(0)
size = 98432 
x = torch.rand(size, device="cuda")
y = torch.rand(size, device="cuda")
output_torch = x + y 
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')


