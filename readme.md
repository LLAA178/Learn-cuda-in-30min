# CUDA 教程：数组相加、同步异步执行、Max/Min规约

欢迎来到本 CUDA 教程！在这个项目中，我们将学习如何使用 CUDA 来实现高效的并行计算，包括数组相加、同步与异步执行的性能比较，以及使用共享内存和 warp shuffle 实现并行规约。

## 视频链接

[点击这里观看教程](https://www.bilibili.com/video/BV1gzFTe8EpT)

## 代码介绍

### 1. 数组相加示例

这个例子演示了如何在 CUDA 中实现一个简单的数组相加操作。我们提供了 CPU 版本和 GPU 版本（使用核函数实现），并通过计时比较它们的性能。

### 2. 同步与异步执行

在这个例子中，我们展示了如何在 CUDA 中使用同步（`cudaMemcpy`）和异步（`cudaMemcpyAsync`）方式来传输数据，并通过多线程同时执行 CPU 与 GPU 任务，提升计算效率。

### 3. Max/Min 规约示例

我们实现了三种方法来求数组的最大值和最小值：
- **CPU 版**：直接在 CPU 上执行。
- **GPU 版（共享内存 + 手动规约）**：每个线程块输出一个部分的最大值和最小值，最后在主机端做一次合并。
- **GPU 版（warp shuffle 规约）**：使用 CUDA warp shuffle 特性来避免共享内存的使用，减少内存访问延迟，提高效率。

### 4. 性能比较

每个示例中都进行了性能计时，帮助你了解不同实现方式的效率差异，并加深对 CUDA 编程的理解。

## 如何运行代码

1. 克隆本项目：
   ```bash
   git clone https://github.com/LLAA178/Learn-cuda-in-30min.git
   ```
2. 在 CUDA 支持的环境中编译代码：
   ```bash
   nvcc your_code.cu -o your_program
   ```
3. 运行程序：
   ```bash
   ./your_program
   ```

## 相关资源

- [CUDA 官方文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#)
- [Youtube gpu painting](https://www.youtube.com/watch?v=-P28LKWTzrI)

---

感谢您的观看，祝您在学习 CUDA 编程的过程中取得优异的成果！