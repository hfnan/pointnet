# 这是附加题模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现triton的深度学习推理过程，请严格保持输出格式输出
import os
import h5py
import time
import numpy as np
import torch
import triton
import triton.language as tl


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

def read_h5_file(dataPath):
    list_of_points = []
    list_of_labels = []
    with h5py.File(dataPath,"r") as hf:
        for k in hf.keys():
            # list_of_points.append(hf[k]["points"][:].astype(np.float32)) #每个points是（N,3）的二维数组ndarray
            list_of_points.append(hf[k]["points"][:].astype(np.float32).flatten()) #每个points是N*3的一维ndarray
            list_of_labels.append(hf[k].attrs["label"])
    return list_of_points,list_of_labels

#示例triton函数
@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector`.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.`
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output

def do_inference(list_of_points,list_of_labels,params): #请在本函数下使用triton实现推理操作
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
    output_triton = add(x, y)
    # output_triton2 = add(x, y)
    # output_triton3 = add(x, y)
    # output_triton4 = add(x, y)

    # print(output_triton)

    accuracy_rate = 0.0001
    return accuracy_rate

if __name__ == '__main__':
    dir = os.path.dirname(__file__) # 保存模型参数文件(.txt)的文件夹路径

    # 读取模型参数
    params = read_params(dir)
    # #示例，获取feat.stn.fc3.bias的参数
    # feat_stn_fc3_bias = params["feat.stn.fc3.bias"]
    # print(f"feat.stn.fc3.bias:")
    # for i in feat_stn_fc3_bias:
    #     print(i)

    # 读取训练集数据
    dataPath = "./data/test_point_clouds.h5"
    list_of_points,list_of_labels = read_h5_file(dataPath)

    # 开始计时
    start = time.time()
    accuracy_rate = do_inference(list_of_points,list_of_labels,params)
    # 结束计时
    end = time.time()
    ms = end - start


    # 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    print(f"{ms:.4f}:{accuracy_rate:.4f}")