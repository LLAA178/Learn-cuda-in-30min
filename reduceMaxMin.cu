#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <float.h>
#include <cuda_runtime.h>

// ------------------------------------------------------------
// 1. CPU版：求最大值和最小值
// ------------------------------------------------------------
void reduceMaxMinCPU(const float* arr, int n, float &outMax, float &outMin)
{
    float maxVal = -FLT_MAX;
    float minVal =  FLT_MAX;

    for (int i = 0; i < n; i++) {
        if (arr[i] > maxVal) maxVal = arr[i];
        if (arr[i] < minVal) minVal = arr[i];
    }
    outMax = maxVal;
    outMin = minVal;
}

// ------------------------------------------------------------
// 2. GPU版（共享内存 + 手动规约）
//   每个线程块输出一个部分 max/min 到 global memory；
//   最终在主机端做一次简单的合并。
// ------------------------------------------------------------
__global__ void reduceMaxMinShared(const float* in, float* blockMax, float* blockMin, int n)
{
    // 共享内存
    extern __shared__ float sdata[];  
    float* smax = sdata;                // 前半存储 max
    float* smin = sdata + blockDim.x;   // 后半存储 min

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x + tid;

    // 读入数据：超出 n 的线程读虚值
    float myMax = (idx < n) ? in[idx] : -FLT_MAX;
    float myMin = (idx < n) ? in[idx] :  FLT_MAX;

    // 写到共享内存
    smax[tid] = myMax;
    smin[tid] = myMin;
    __syncthreads();

    // 在共享内存里逐步规约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smax[tid] = fmaxf(smax[tid], smax[tid + s]);
            smin[tid] = fminf(smin[tid], smin[tid + s]);
        }
        __syncthreads();
    }

    // 本块的最终结果由 tid=0 写到 global memory
    if (tid == 0) {
        blockMax[blockIdx.x] = smax[0];
        blockMin[blockIdx.x] = smin[0];
    }
}

// ------------------------------------------------------------
// 3. GPU版（warp shuffle 规约）
//    不再借助共享内存做最后的循环规约，改用 __shfl_down_sync。
//    这里演示一个“分块规约 + warp shuffle”相结合的思路。
// ------------------------------------------------------------
__device__ float warpReduceMax(float val)
{
    // 典型的warp规约：连续二分法
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float warpReduceMin(float val)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void reduceMaxMinWarp(const float* in, float* blockMax, float* blockMin, int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x + tid;

    // 每个线程先读取一个元素
    float myMax = (idx < n) ? in[idx] : -FLT_MAX;
    float myMin = (idx < n) ? in[idx] :  FLT_MAX;

    // 首先在warp内做一次shuffle规约
    myMax = warpReduceMax(myMax);
    myMin = warpReduceMin(myMin);

    // 对于大于warpSize的block，需要多warp再继续规约
    // 将每个warp的结果写入共享内存，再由第0个warp继续规约
    __shared__ float sharedMax[32];  // 最多 32个warp（blockDim.x <= 1024）
    __shared__ float sharedMin[32];

    // warp级别信息
    int warpID = tid / warpSize;
    int lane   = tid % warpSize;

    // 让warp内线程0写入共享内存
    if (lane == 0) {
        sharedMax[warpID] = myMax;
        sharedMin[warpID] = myMin;
    }
    __syncthreads();

    // 由warp 0 来继续对 shared[] 做规约
    if (warpID == 0)
    {
        // 取当前线程所在的lane，对应在 shared 里的元素
        float valMax = (lane < (blockDim.x / warpSize + 1)) ? 
                        sharedMax[lane] : -FLT_MAX;
        float valMin = (lane < (blockDim.x / warpSize + 1)) ? 
                        sharedMin[lane] :  FLT_MAX;

        // 再用warp shuffle做一次规约
        valMax = warpReduceMax(valMax);
        valMin = warpReduceMin(valMin);

        // lane=0 最终得到本block结果
        if (lane == 0) {
            blockMax[blockIdx.x] = valMax;
            blockMin[blockIdx.x] = valMin;
        }
    }
}

// ----------------------------------------------------------------------
// 辅助函数：在完成各block部分结果后，CPU端合并得到最终结果
// blockMax/blockMin 数组长度为 gridSize
// ----------------------------------------------------------------------
void finalizeBlockResults(const float* blockMaxHost, const float* blockMinHost,
                          int gridSize, float &finalMax, float &finalMin)
{
    float tmpMax = -FLT_MAX;
    float tmpMin =  FLT_MAX;
    for (int i = 0; i < gridSize; i++) {
        if (blockMaxHost[i] > tmpMax) tmpMax = blockMaxHost[i];
        if (blockMinHost[i] < tmpMin) tmpMin = blockMinHost[i];
    }
    finalMax = tmpMax;
    finalMin = tmpMin;
}

// ----------------------------------------------------------------------
// 主函数
// ----------------------------------------------------------------------
int main()
{
    // 准备数据
    const int N = 1 << 25;  // 约0.5G
    std::vector<float> h_in(N);

    // 简单初始化：随机或者赋一些值
    for (int i = 0; i < N; i++) {
        // 让值在 [0,100) 之间浮动
        h_in[i] = static_cast<float>((rand() % 10000) / 100.0);
    }

    // 分配GPU内存
    float *d_in;
    cudaMalloc(&d_in, N * sizeof(float));
    // 拷贝数据到GPU（这里不记入计时，只演示核函数的比较）
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // CPU结果
    float cpuMax, cpuMin;
    {
        // 计时CPU
        double cpuTime = 0.0;
        auto startCPU = std::chrono::high_resolution_clock::now();
        reduceMaxMinCPU(h_in.data(), N, cpuMax, cpuMin);
        auto endCPU = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = endCPU - startCPU;
        cpuTime = duration.count();

        std::cout << "[CPU Version] Max = " << cpuMax << ", Min = " << cpuMin
                  << "  Time = " << cpuTime << " ms\n";
    }

    // 线程配置
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    // 分配存储“每个Block的部分结果”的内存
    float *d_blockMax, *d_blockMin;
    cudaMalloc(&d_blockMax, gridSize * sizeof(float));
    cudaMalloc(&d_blockMin, gridSize * sizeof(float));

    // Host 端存储最终block结果，用于合并
    std::vector<float> h_blockMax(gridSize), h_blockMin(gridSize);

    // ========== 1) 共享内存 + 手动规约 ==========
    {
        // 创建 CUDA 事件进行计时
        cudaEvent_t startEvent, stopEvent;
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);

        // 记录开始
        cudaEventRecord(startEvent, 0);

        // 启动 kernel
        // 这里要给共享内存分配 2*blockSize 的空间 (max + min)
        size_t smemSize = 2 * blockSize * sizeof(float);
        reduceMaxMinShared<<<gridSize, blockSize, smemSize>>>(d_in, d_blockMax, d_blockMin, N);

        // 记录结束
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);

        float kernelTime = 0.0f;
        cudaEventElapsedTime(&kernelTime, startEvent, stopEvent);

        // 拷回部分结果并合并(不计入kernel时间)
        cudaMemcpy(h_blockMax.data(), d_blockMax, gridSize*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_blockMin.data(), d_blockMin, gridSize*sizeof(float), cudaMemcpyDeviceToHost);

        float finalMax, finalMin;
        finalizeBlockResults(h_blockMax.data(), h_blockMin.data(), gridSize, finalMax, finalMin);

        std::cout << "[GPU SharedMem]  Max = " << finalMax 
                  << ", Min = " << finalMin
                  << "  KernelTime = " << kernelTime << " ms\n";

        // 销毁事件
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    // ========== 2) Warp Shuffle 规约 ==========
    {
        // 创建 CUDA 事件进行计时
        cudaEvent_t startEvent, stopEvent;
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);

        // 记录开始
        cudaEventRecord(startEvent, 0);

        // 启动 kernel
        reduceMaxMinWarp<<<gridSize, blockSize>>>(d_in, d_blockMax, d_blockMin, N);

        // 记录结束
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);

        float kernelTime = 0.0f;
        cudaEventElapsedTime(&kernelTime, startEvent, stopEvent);

        // 拷回部分结果并合并(不计入kernel时间)
        cudaMemcpy(h_blockMax.data(), d_blockMax, gridSize*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_blockMin.data(), d_blockMin, gridSize*sizeof(float), cudaMemcpyDeviceToHost);

        float finalMax, finalMin;
        finalizeBlockResults(h_blockMax.data(), h_blockMin.data(), gridSize, finalMax, finalMin);

        std::cout << "[GPU WarpShuffle] Max = " << finalMax 
                  << ", Min = " << finalMin
                  << "  KernelTime = " << kernelTime << " ms\n";

        // 销毁事件
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    // 清理资源
    cudaFree(d_in);
    cudaFree(d_blockMax);
    cudaFree(d_blockMin);

    return 0;
}
