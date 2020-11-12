#include "main.h"

//´úÂë£ºhttps://www.cnblogs.com/aiguona/p/9479881.html

#define pi 3.1415926535
#define LENGTH 100 //signal sampling points

void mainfft()
{
    // data gen
    float Data[LENGTH] = { 1,2,3,4 };
    float fs = 1000000.000;//sampling frequency
    float f0 = 200000.00;// signal frequency
    for (int i = 0; i < LENGTH; i++)
    {
        Data[i] = (float)(1.35 * cos(2 * pi * f0 * i / fs));//signal gen,
    }

    cufftComplex* CompData = (cufftComplex*)malloc(LENGTH * sizeof(cufftComplex));//allocate memory for the data in host
    int i;
    for (i = 0; i < LENGTH; i++)
    {
        CompData[i].x = Data[i];
        CompData[i].y = 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    }

    cufftComplex* d_fftData;
    cudaMalloc((void**)&d_fftData, LENGTH * sizeof(cufftComplex));// allocate memory for the data in device
    cudaMemcpy(d_fftData, CompData, LENGTH * sizeof(cufftComplex), cudaMemcpyHostToDevice);// copy data from host to device

    cufftHandle plan;// cuda library function handle
    cufftPlan1d(&plan, LENGTH, CUFFT_C2C, 1);//declaration
    cufftExecC2C(plan, (cufftComplex*)d_fftData, (cufftComplex*)d_fftData, CUFFT_FORWARD);//execute
    cudaDeviceSynchronize();//wait to be done
    cudaMemcpy(CompData, d_fftData, LENGTH * sizeof(cufftComplex), cudaMemcpyDeviceToHost);// copy the result from device to host

    for (i = 0; i < LENGTH / 2; i++)
    {
        printf("i=%d\tf= %6.1fHz\tRealAmp=%3.1f\t", i, fs * i / LENGTH, CompData[i].x * 2.0 / LENGTH);
        printf("ImagAmp=+%3.1fi", CompData[i].y * 2.0 / LENGTH);
        printf("\n");
    }
    cufftDestroy(plan);
    free(CompData);
    cudaFree(d_fftData);

}
