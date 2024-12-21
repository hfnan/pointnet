import torch
import triton
import triton.language as tl 

@triton.jit 
def test_kernel(
    in_ptr, out_ptr, 
    vec_length, 
    BLOCK_SIZE: tl.constexpr 
):
    pid = tl.program_id(axis=0)
    nump = tl.num_programs(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vec_length

    inv = tl.load(in_ptr + offsets, mask=mask)
    outv = tl.load(out_ptr + offsets, mask=mask)

    outv = inv + nump

    tl.store(out_ptr + offsets, outv, mask=mask)


def main():
    output = torch.empty(12).cuda()
    print(output)

    input = torch.arange(12).cuda()
    print(input)

    length = len(input)
    print(length)

    grid = lambda meta: (2, )

    test_kernel[grid](input, output, length, 4)
    print(output)
    

if __name__ == "__main__":
    main()