## 英特尔 oneAPI 黑客

### 活动简介

* 使用 oneMKL 工具，对 FFT 算法进行加速与优化。
* 问题描述

  * 调用 oneMKL 相应 API 函数， 产生 2048 * 2048 个 随机单精度实数()；
  * 根据 2 产生的随机数据作为输入，实现两维 Real to complex FFT 参考代码；
  * 根据 2 产生的随机数据作为输入， 调用 oneMKL API 计算两维 Real to complex FFT；
  * 结果正确性验证，对 3 和 4 计算的两维 FFT 输出数据进行全数据比对（允许适当精度误
    差）， 输出 “结果正确”或“结果不正确”信息；
  * 平均性能数据比对（比如运行 1000 次），输出 FFT 参考代码平均运行时间和 oneMKL
    FFT 平均运行时间。

### 实验过程

### 1. 数据生成与准备

这部分的代码负责准备数据，包括内存分配和随机数据的生成。

#### 1.1 `makeData()`

 **功能** ：为输入数据分配内存。

 **详细描述** ：在数字信号处理中，我们经常处理大量的数据。此函数通过 `malloc`为这些数据分配所需的内存，确保我们在后续处理中有足够的空间。选择正确的数据大小（在此为 `N * M`）对于确保FFT的准确性和效率至关重要。

```cpp
// 生成随机数
float* makeData() 
{
    return (float*)malloc(N * M * sizeof(float));
}
```

#### 1.2 `generateRandomData(float* data)`

 **功能** ：为输入数组生成随机数据。

 **详细描述** ：随机数据经常在测试和基准测试中使用，以确保算法在各种输入条件下的稳健性。此处，我们使用Intel's MKL提供的随机数生成函数为数据数组填充均匀分布的随机数。选择均匀分布是因为它能提供广泛的数据变化，确保我们的FFT实现在各种数据输入下都能正确工作。

```cpp
// 将随机数转成单精度
void generateRandomData(float* data) 
{
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 1);
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N * M, data, 0.0f, 1.0f);
    vslDeleteStream(&stream);
}
```

---

### 2. FFT计算

这部分的代码负责执行FFT计算，使用了两种不同的库：FFTW和oneMKL。

#### 2.1 `execute_FFTW(fftwf_plan fftw_plan)`

 **功能** ：使用FFTW库执行快速傅里叶变换并测量执行时间。

 **详细描述** ：FFTW是一个广泛使用的C库，专门用于计算快速傅里叶变换。这个函数不仅执行FFT，还测量其执行时间，这对于后续性能比较很有价值。使用 `QueryPerformanceCounter`确保测量的时间精度很高。

```cpp
// 计算快速傅里叶变换，返回耗时
double execute_FFTW(fftwf_plan fftw_plan) 
{
    LARGE_INTEGER start, end;
    QueryPerformanceCounter(&start);
    fftwf_execute(fftw_plan);
    QueryPerformanceCounter(&end);
    return (double)(end.QuadPart - start.QuadPart);
}
```

#### 2.2 `execute_oneMKL(DFTI_DESCRIPTOR_HANDLE hand, float* data, MKL_Complex8* mkl_output_data)`

 **功能** ：使用oneMKL库执行快速傅里叶变换并测量执行时间。

 **详细描述** ：Intel的oneMKL是为高性能计算优化的数学库。与FFTW类似，我们在此函数中执行FFT，并测量执行时间。性能的直接比较可以帮助我们决定在特定应用中使用哪个库。

```cpp
// 利用oneMKL加速FFT，返回耗时
double execute_oneMKL(DFTI_DESCRIPTOR_HANDLE hand, float* data, MKL_Complex8* mkl_output_data)
{
    LARGE_INTEGER start, end;
    QueryPerformanceCounter(&start);
    DftiComputeForward(hand, data, mkl_output_data);
    QueryPerformanceCounter(&end);
    return (double)(end.QuadPart - start.QuadPart);
}
```

#### 2.3 `createDftiDescriptor()`

 **功能** ：为oneMKL库的DFT操作创建和配置描述符。

 **详细描述** ：为了在oneMKL中执行FFT，我们需要一个配置好的描述符，它定义了如何在库内部执行变换。这个函数进行了所有必要的设置，确保我们的FFT操作既准确又高效。

```cpp
// 调用oneMKLAPI计算两维Real to complex FFT
DFTI_DESCRIPTOR_HANDLE createDftiDescriptor() 
{
    MKL_LONG size[2] = { M, N };

    // 定义描述符，作用是存储计算的中间结果
    MKL_LONG input_stride[3] = { 0, N, 1 };
    MKL_LONG output_stride[3] = { 0, N / 2 + 1, 1 };

    DFTI_DESCRIPTOR_HANDLE hand = NULL;
    DftiCreateDescriptor(&hand, DFTI_SINGLE, DFTI_REAL, 2, size);
    DftiSetValue(hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(hand, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(hand, DFTI_INPUT_STRIDES, input_stride);
    DftiSetValue(hand, DFTI_OUTPUT_STRIDES, output_stride);
    DftiCommitDescriptor(hand);

    return hand;
}
```

---

### 3. 结果验证

在进行FFT之后，我们需要确保所得的结果是正确的。这部分代码就是用来验证这一点。

#### 3.1 `areOutputsMatching(const MKL_Complex8* mkl_output_data, const fftwf_complex* fftw_output_data)`

 **功能** ：比较FFTW和oneMKL的输出数据是否匹配。

 **详细描述** ：为了确保我们的FFT实现是正确的，我们比较了两个不同库的输出。由于这两个库在内部可能使用了不同的算法和优化，轻微的差异是可以预期的。然而，任何大的不一致都可能指示一个问题。这个函数通过对比输出来帮助我们捕捉到这些问题。

```cpp
bool areOutputsMatching(const MKL_Complex8* mkl_output_data, const fftwf_complex* fftw_output_data)
{
    for (int i = 0; i < (N / 2 + 1) * M; i++)
    {
        if (fabs(mkl_output_data[i].real - fftw_output_data[i][0]) > 1e-6 ||
            fabs(mkl_output_data[i].imag - fftw_output_data[i][1]) > 1e-6) 
        {
            return false;
        }
    }
    return true;
}
```

---

### 4. 主程序

#### `main()`

 **功能** ：主执行流程。

 **详细描述** ：
此函数是整个程序的起点和核心。它按照以下步骤组织了上述所有函数的执行：

1. 循环执行 `iter_count`次，每次：
   * 生成和准备数据。
   * 使用FFTW和oneMKL分别执行FFT。
   * 比较两者的输出结果，并标记任何不一致。
   * 释放本次迭代使用的所有资源。
2. 最后，根据所有迭代的结果，打印出对比结果和两个库的平均执行时间。

```cpp
int main() 
{
    double fftw_total_time = 0.0;
    double mkl_total_time = 0.0;
    bool allResultsCorrect = true;

    for (int i = 0; i < iter_count; i++) 
    {
        float* data = makeData();
        generateRandomData(data);

        // FFTW 计划
        // 为FFTW库执行快速傅里叶变换的输出结果分配适当大小的对齐内存空间
        // fftwf_complex为FFTW库的一个复数类型
        fftwf_complex* fftw_output_data = (fftwf_complex*)fftwf_malloc((N / 2 + 1) * M * sizeof(fftwf_complex));

        // 为FFTW库创建一个策略，用于执行二维实数到复数的快速傅里叶变换。
        fftwf_plan fftw_plan = fftwf_plan_dft_r2c_2d(N, M, data, fftw_output_data, FFTW_ESTIMATE);

        // 配置一个傅里叶变换描述符
        DFTI_DESCRIPTOR_HANDLE hand = createDftiDescriptor();
        // 分配 OneMKL 输出数组
        MKL_Complex8* mkl_output_data = (MKL_Complex8*)malloc((N / 2 + 1) * M * sizeof(MKL_Complex8));

        fftw_total_time += execute_FFTW(fftw_plan);
        mkl_total_time += execute_oneMKL(hand, data, mkl_output_data);

        if (!areOutputsMatching(mkl_output_data, fftw_output_data)) 
        {
            printf("Iteration %d - 结果不正确！\n", i + 1);
            allResultsCorrect = false;
        }

        fftwf_destroy_plan(fftw_plan);
        fftwf_free(fftw_output_data);
        DftiFreeDescriptor(&hand);
        free(data);
        free(mkl_output_data);
    }

    printf("执行 %d次 的比较结果：%s\n", iter_count, allResultsCorrect ? "结果正确" : "结果不正确");
    if (fftw_total_time > 0 && mkl_total_time > 0) 
    {
        printf("FFTW3 FFT 平均执行时间：%.3fμs\n", fftw_total_time / iter_count);
        printf("OneMKL FFT 平均执行时间：%.3fμs\n", mkl_total_time / iter_count);
    }

    return 0;
}
```

### 5.参赛经验

* 通过本次大赛我第一次了解到了傅里叶变换，并且学习了oneAPI的使用，对于出现的BUG自己则尝试从多种渠道，例如阅读官方文档等方式进行解决，编程能力得到了提高。
* oneAPI库给我最大的感受就是线代数学函数库的高精度，有效和优越性，对于复杂的运算直接调用接口就能得到精确的结果。
