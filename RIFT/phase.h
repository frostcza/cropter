#ifndef __PHASE_H__
#define __PHASE_H__

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/time.h>
#include <jetson-utils/cudaMappedMemory.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include "cudaFunc.h"

//yjt daima
class PhaseCongruency
{
public:
    // PhaseCongruency(cv::Size _img_size, size_t _nscale, size_t _norient);
    PhaseCongruency();
    ~PhaseCongruency() {}
    void Init(cv::Size _img_size, size_t _nscale, size_t _norient);
    // void calc_eo(cv::InputArray _src);
    void cudatest(cv::InputArray _src);
    std::vector<cv::Mat> eo;
    cv::Mat maxMoment;
    double * Max_moment;

private:
    cv::Size size;
    size_t norient;
    size_t nscale;
    std::vector<cv::Mat> filter; //原来代码使用的filter cv版本

    std::vector<cv::Mat> filter_cpu; //实数filter 
    
    double ** filter_host; //filter 指针数组版本
    double ** filter_gpu; //filter gpu版本 

    cufftDoubleComplex ** filtered_gpu;//滤波之后的图像
    cufftDoubleComplex ** ifft_img_gpu;//反变换之后的图像 实际上两者留一个就好

    double ** An; //也就是 eo_mag gpu版本 申请为 cudaallocmapped
    double ** An_cpu; //eo_mag cpu版本

    double* host_data;
    double * idata; //输入图像
    cufftDoubleComplex * img_complex_gpu; //输入图像补0
    cufftDoubleComplex* odata; //输出fft图像

    double* sum_re;
    double* sum_im;
    double* sum_an;
    double* max_an;
    double* complex0;
    double* complex1;
    double* energy;
    double ** pc_gpu;
    double * convx;
    double * convy;
    double * convx2;
    double * convy2;
    double * convxy;

    

    cufftHandle plan_forward; // fft 变换句柄
    cufftHandle plan_inverse;

    double sigma;
    double mult;  //1.6
    double minwavelength;
    double epsilon;
    double cutOff;
    double g;  //3
    double k;  //1
};

class MyTime
{
public:
    static struct timeval start, end;
    void static inline tic(){ gettimeofday(&start,NULL); }
    void static inline toc(std::string s) 
    {
        gettimeofday(&end,NULL);
        std::cout << s << "'s run time is " << (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000 << "s." << std::endl;
    }

};

#endif // PHASE_H