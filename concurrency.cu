#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// -------------------- CPU 耗时函数 --------------------
// 模拟一个比较耗时的CPU计算（例如大循环 + 数学运算）
void heavyWorkCPU(int N)
{
    double sum = 0.0;
    for (int i = 0; i < N; ++i)
    {
        sum += std::sqrt(static_cast<double>(i));
    }
    // 打印，以防编译器对 sum 优化
    std::cout << "[CPU] sum = " << sum << std::endl;
}

// -------------------- GPU 核函数 (模拟耗时计算) --------------------
__global__ void heavyWorkKernel(const float* in, float* out, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        float val = in[idx];
        // 做一些重复运算来拉长计算时间
        for (int i = 0; i < 100; i++)
        {
            val = sqrtf(val + idx * 0.00001f);
        }
        out[idx] = val;
    }
}

// -------------------- 方法A：无并行（串行同步） --------------------
float runWithoutConcurrency(float* h_in, float* h_out, float* d_in, float* d_out, int N)
{
    // 计时开始
    auto start = std::chrono::high_resolution_clock::now();

    // 1) 主机->设备 拷贝（同步）
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    // 2) 启动 kernel（默认流），再同步
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;
    heavyWorkKernel<<<gridSize, blockSize>>>(d_in, d_out, N);
    cudaDeviceSynchronize(); // 等 GPU 计算结束

    // 3) 设备->主机 拷贝（同步）
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 4) CPU 耗时计算
    heavyWorkCPU(N);

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return static_cast<float>(duration.count());
}

// -------------------- 方法B：并行（异步 + 流） --------------------
float runWithConcurrency(float* h_in, float* h_out, float* d_in, float* d_out, int N)
{
    // 为了更好地展示CPU-GPU并行，使用 cudaStream_t
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 计时开始
    auto start = std::chrono::high_resolution_clock::now();

    // 1) 主机->设备 拷贝（异步）
    cudaMemcpyAsync(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice, stream);

    // 2) 启动 kernel（指定流）
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;
    heavyWorkKernel<<<gridSize, blockSize, 0, stream>>>(d_in, d_out, N);

    // 3) GPU 端执行完 kernel 后，设备->主机 拷贝（异步）
    cudaMemcpyAsync(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost, stream);

    // 4) **无等待**，直接在 CPU 干自己的 heavyWork
    //    GPU 在 stream 中排队执行 kernel
    heavyWorkCPU(N);


    // 5) 等待流中所有任务完成
    cudaStreamSynchronize(stream);

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    cudaStreamDestroy(stream);
    return static_cast<float>(duration.count());
}

int main()
{
    // 数组大小
    int N = 1 << 25; // 约 0.5G
    size_t bytes = N * sizeof(float);

    // -------------------- 分配主机内存（Pinned Memory更利于异步拷贝） --------------------
    float *h_in, *h_out;
    cudaMallocHost((void**)&h_in,  bytes); // 使用固定页内存
    cudaMallocHost((void**)&h_out, bytes);

    // 初始化数据
    for (int i = 0; i < N; i++)
    {
        h_in[i] = static_cast<float>(i * 0.1f);
    }

    // -------------------- 分配设备内存 --------------------
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in,  bytes);
    cudaMalloc((void**)&d_out, bytes);

    // -------------------- 测试 A：无并行（串行） --------------------
    float timeA = runWithoutConcurrency(h_in, h_out, d_in, d_out, N);
    std::cout << "\n[Without Concurrency] Total time: " << timeA << " ms\n";

    // -------------------- 测试 B：异步流（并行） --------------------
    float timeB = runWithConcurrency(h_in, h_out, d_in, d_out, N);
    std::cout << "[With Concurrency]   Total time: " << timeB << " ms\n\n";

    // 对比
    std::cout << "Speedup (A/B): ~" << timeA / timeB << "x\n\n";

    // -------------------- 清理资源 --------------------
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);

    return 0;
}
