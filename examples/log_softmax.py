import torch 
import torch.nn.functional as F
import triton
import triton.language as tl

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

torch.manual_seed(0)
A = torch.randn((1000, 10), device="cuda", dtype=torch.float16)
# B = torch.randn((1000, 50, 512), device="cuda", dtype=torch.float16)
triton_output = log_softmax(A)
torch_output = F.log_softmax(A, dim=1)
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



