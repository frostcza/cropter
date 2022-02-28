#ifndef PHASE_H
#define PHASE_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/time.h>

class PhaseCongruency
{
public:
    // PhaseCongruency(cv::Size _img_size, size_t _nscale, size_t _norient);
    PhaseCongruency();
    ~PhaseCongruency() {}
    void Init(cv::Size _img_size, size_t _nscale, size_t _norient);
    void calc(cv::InputArray _src, std::vector<cv::Mat> &_pc);
    void feature(std::vector<cv::Mat> &_pc, cv::OutputArray _edges, cv::OutputArray _corners);
    void feature(cv::InputArray _src, cv::OutputArray _edges, cv::OutputArray _corners);
    std::vector<cv::Mat> eo; //有s*o个
    cv::Mat maxMoment;

private:
    cv::Size size;
    size_t norient; // 方向数
    size_t nscale; // 尺度数
    std::vector<cv::Mat> filter; // 有s*o个

    // some constants
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
