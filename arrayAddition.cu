#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// -------------------- CPU函数：数组相加 --------------------
void addArraysCPU(const float* A, const float* B, float* C, int size)
{
    for (int i = 0; i < size; i++)
    {
        C[i] = A[i] + B[i];
    }
}

// -------------------- GPU核函数：数组相加 --------------------
__global__ void addArraysKernel(const float* A, const float* B, float* C, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// -------------------- GPU函数封装：调用核函数 --------------------
void addArraysGPU(const float* h_A, const float* h_B, float* h_C, int size)
{
    // 定义指向设备端的指针
    float *d_A, *d_B, *d_C;
    float gpuTime = 0.0f;
    // CUDA事件
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // 分配显存
    cudaMalloc((void**)&d_A, size * sizeof(float));
    cudaMalloc((void**)&d_B, size * sizeof(float));
    cudaMalloc((void**)&d_C, size * sizeof(float));

    // 将数据从主机端传输到设备端
    cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);

    // 配置 kernel 的执行维度
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    cudaEventRecord(startEvent, 0);
    // 启动 kernel
    addArraysKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, size);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&gpuTime, startEvent, stopEvent);

    // 输出GPU时间
    std::cout << "GPU array addition time: " << gpuTime << " ms\n\n";

    // 将结果拷贝回主机端
    cudaMemcpy(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放显存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // 释放CUDA事件
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

int main()
{
    // 设置数组大小
    int size = 1 << 25; // 约0.5G
    size_t bytes = size * sizeof(float);

    // 分配主机内存
    std::vector<float> h_A(size), h_B(size), h_C_CPU(size), h_C_GPU(size);

    // 初始化数据
    for (int i = 0; i < size; i++)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // -------------------- 计时CPU版本 --------------------
    auto startCPU = std::chrono::high_resolution_clock::now();
    addArraysCPU(h_A.data(), h_B.data(), h_C_CPU.data(), size);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = endCPU - startCPU;
    std::cout << "CPU array addition time: " << cpuDuration.count() << " ms\n";

    // -------------------- 计时GPU版本 --------------------

    addArraysGPU(h_A.data(), h_B.data(), h_C_GPU.data(), size);

    // -------------------- 验证结果正确性 --------------------
    bool correct = true;
    for (int i = 0; i < size; i++)
    {
        if (fabs(h_C_CPU[i] - h_C_GPU[i]) > 1e-5)
        {
            correct = false;
            break;
        }
    }
    std::cout << (correct ? "Results are correct!" : "Results are incorrect!") << std::endl;

    return 0;
}
