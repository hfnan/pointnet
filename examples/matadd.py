import torch 
import triton
import triton.language as tl 

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

def add(a, b):
    M, N = a.shape
    assert b.shape[0] == N, "?"

    c = torch.empty_like(a)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]), )
    add_kernel[grid](
        a, b, c,
        M, N,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=triton.next_power_of_2(N)
    )
    return c 


torch.manual_seed(0)
a = torch.randn((1000, 64), device="cuda", dtype=torch.float16)
b = torch.randn(64, device="cuda", dtype=torch.float16)

triton_output = add(a, b)
torch_output = a + b

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

