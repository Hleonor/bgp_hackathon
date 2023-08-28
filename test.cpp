#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <mkl.h>
#include <math.h>
#include <windows.h>

#define N 2048
#define M 2048
const int iter_count = 1000;

// 生成随机数
float* makeData() 
{
    return (float*)malloc(N * M * sizeof(float));
}

// 计算快速傅里叶变换，返回耗时
double execute_FFTW(fftwf_plan fftw_plan) 
{
    LARGE_INTEGER start, end;
    QueryPerformanceCounter(&start);
    fftwf_execute(fftw_plan);
    QueryPerformanceCounter(&end);
    return (double)(end.QuadPart - start.QuadPart);
}

// 利用oneMKL加速FFT，返回耗时
double execute_oneMKL(DFTI_DESCRIPTOR_HANDLE hand, float* data, MKL_Complex8* mkl_output_data)
{
    LARGE_INTEGER start, end;
    QueryPerformanceCounter(&start);
    DftiComputeForward(hand, data, mkl_output_data);
    QueryPerformanceCounter(&end);
    return (double)(end.QuadPart - start.QuadPart);
}

// 将随机数转成单精度
void generateRandomData(float* data) 
{
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 1);
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N * M, data, 0.0f, 1.0f);
    vslDeleteStream(&stream);
}

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
