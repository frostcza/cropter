#ifndef RIFT_NO_ROTATION_INVARIANCE_H
#define RIFT_NO_ROTATION_INVARIANCE_H

#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/cudaimgproc.hpp>
#include "phase.h"

class RIFT{
public:
    RIFT(int _s, int _o, int _patch_size, int _thre, float _RATIO, cv::Size img_size);
    void Inference(cv::Mat im1, cv::Mat im2, float transMat[3][3], int id);
    cv::Mat RIFT_no_rotation_invariance(cv::Mat img, std::vector<cv::KeyPoint> &kps);
    cv::Mat RIFT_descriptor_no_rotation_invariance(cv::Mat im, std::vector<cv::KeyPoint> keypoints,std::vector<cv::KeyPoint> &kps, std::vector<cv::Mat> eo);
    cv::Mat CalcMatch(cv::Mat& des_m1, cv::Mat& des_m2, std::vector<cv::KeyPoint> &kps1, std::vector<cv::KeyPoint> &kps2, std::vector<cv::DMatch>& InlierMatches);


private:
    int s;
    int o;
    int patch_size;
    int thre;
    float RATIO;
    PhaseCongruency pc;

};

#endif