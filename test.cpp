#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <mkl.h>
#include <math.h>
#include <windows.h>

#define N 2048
#define M 2048
const int iter_count = 1000;

// ���������
float* makeData() 
{
    return (float*)malloc(N * M * sizeof(float));
}

// ������ٸ���Ҷ�任�����غ�ʱ
double execute_FFTW(fftwf_plan fftw_plan) 
{
    LARGE_INTEGER start, end;
    QueryPerformanceCounter(&start);
    fftwf_execute(fftw_plan);
    QueryPerformanceCounter(&end);
    return (double)(end.QuadPart - start.QuadPart);
}

// ����oneMKL����FFT�����غ�ʱ
double execute_oneMKL(DFTI_DESCRIPTOR_HANDLE hand, float* data, MKL_Complex8* mkl_output_data)
{
    LARGE_INTEGER start, end;
    QueryPerformanceCounter(&start);
    DftiComputeForward(hand, data, mkl_output_data);
    QueryPerformanceCounter(&end);
    return (double)(end.QuadPart - start.QuadPart);
}

// �������ת�ɵ�����
void generateRandomData(float* data) 
{
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 1);
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N * M, data, 0.0f, 1.0f);
    vslDeleteStream(&stream);
}

// ����oneMKLAPI������άReal to complex FFT
DFTI_DESCRIPTOR_HANDLE createDftiDescriptor() 
{
    MKL_LONG size[2] = { M, N };
	
    // �����������������Ǵ洢������м���
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

        // FFTW �ƻ�
        // ΪFFTW��ִ�п��ٸ���Ҷ�任�������������ʵ���С�Ķ����ڴ�ռ�
        // fftwf_complexΪFFTW���һ����������
        fftwf_complex* fftw_output_data = (fftwf_complex*)fftwf_malloc((N / 2 + 1) * M * sizeof(fftwf_complex));

        // ΪFFTW�ⴴ��һ�����ԣ�����ִ�ж�άʵ���������Ŀ��ٸ���Ҷ�任��
        fftwf_plan fftw_plan = fftwf_plan_dft_r2c_2d(N, M, data, fftw_output_data, FFTW_ESTIMATE);

        // ����һ������Ҷ�任������
        DFTI_DESCRIPTOR_HANDLE hand = createDftiDescriptor();
        // ���� OneMKL �������
        MKL_Complex8* mkl_output_data = (MKL_Complex8*)malloc((N / 2 + 1) * M * sizeof(MKL_Complex8));

        fftw_total_time += execute_FFTW(fftw_plan);
        mkl_total_time += execute_oneMKL(hand, data, mkl_output_data);

        if (!areOutputsMatching(mkl_output_data, fftw_output_data)) 
        {
            printf("Iteration %d - �������ȷ��\n", i + 1);
            allResultsCorrect = false;
        }

        fftwf_destroy_plan(fftw_plan);
        fftwf_free(fftw_output_data);
        DftiFreeDescriptor(&hand);
        free(data);
        free(mkl_output_data);
    }

    printf("ִ�� %d�� �ıȽϽ����%s\n", iter_count, allResultsCorrect ? "�����ȷ" : "�������ȷ");
    if (fftw_total_time > 0 && mkl_total_time > 0) 
    {
        printf("FFTW3 FFT ƽ��ִ��ʱ�䣺%.3f��s\n", fftw_total_time / iter_count);
        printf("OneMKL FFT ƽ��ִ��ʱ�䣺%.3f��s\n", mkl_total_time / iter_count);
    }

    return 0;
}
