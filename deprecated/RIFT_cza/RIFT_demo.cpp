#include <iostream>
#include <opencv2/opencv.hpp>
#include "rift_no_rotation_invariance.h"

#include <vector>
#include <string>
#include <fstream>
#include <iomanip>

using namespace cv;
using namespace std;


/* RIFT配准主要步骤：
 * phase congruency 相位一致性 计算PC图，得到类似边缘检测的结果
 * FAST dectector 提取特特征点，边缘图->特征点的x,y坐标
 * RIFT descriptor 描述子，每个特征点对应一个特征向量
 * BFmatcher 用暴力算法求解匹配，计算两个图中所有特征点之间的欧氏距离，取距离最小的
 * RANSAC 假设两图符合单应变换模型，利用RANSAC进行误匹配剔除，得到最优的变换矩阵H
 * warpPerspective 把所得的H和原图A相乘，得到变换后的图A
 * fusion 融合规则不属于配准该管的事，这里使用加权平均规则
 * 所有测试均在/dataset/IR/00.jpg，/dataset/VI/00.jpg进行，注释中记录的数据是这对图像的结果
 */

int main(int argc, char* argv[])
{
	RIFT rift(4, 6, 96, 5, 1.0f, cv::Size(640,512)); // im1.size(), 00.jpg->(256,256) else->(640,512)
	for (int i=1;i<20;i++)
	{
		printf("pic num: %d\n", i);
		char irPath[100];
		sprintf(irPath, "/home/cza/cropter/deprecated/RIFT_cza/dataset/IR/%02d.jpg",i);
		char viPath[100];
		sprintf(viPath, "/home/cza/cropter/deprecated/RIFT_cza/dataset/VI/%02d.jpg",i);
		cv::Mat im1 = cv::imread(irPath);
    	cv::Mat im2 = cv::imread(viPath);
		float transMat[3][3];
		MyTime::tic();
		cv::resize(im2, im2, im1.size());
		rift.Inference(im1, im2, transMat, i);
		MyTime::toc("RIFT total");


		// Homography matrix evaluation: reject wrong result according to experience
		// https://zhuanlan.zhihu.com/p/74597564
		// 	transMat =	[a	 b	 c;
		//     			 d	 e	 f;
		// 				 g	 h	 i]
		// 1. 认为两个相机满足仿射变换而不是单应变换，g,h应接近于0（使用findHomography的原因是opencv没提供仿射接口）
		// 2. 旋转不可能大于90度，且不存在翻转，a,e应同时大于零
		// 3. 图像本身是640x512大小，c,f是平移，不可能很大，一般绝对值在150以内
		// 4. 由于我们的VI摄像头在左，IR摄像头在右，所以把IR往VI上配时，一般X方向上的位移c>0

		bool trust = true;
		if ( std::abs(transMat[2][0]) > 0.001 || std::abs(transMat[2][1]) > 0.001
				|| transMat[0][0] <= 0 || transMat[1][1] <= 0 
				|| std::abs(transMat[0][2]) > 150 || std::abs(transMat[1][2]) > 150 )
		{
			trust = false;
		}
		if (trust == false)
		{
			transMat[0][0] = 1.15; transMat[0][1] = 0; transMat[0][2] = 48;
			transMat[1][0] = 0; transMat[1][1] = 1.25; transMat[1][2] = -45;
			transMat[2][0] = 0; transMat[2][1] = 0; transMat[2][2] = 1;
		}

		
		// // CV perspective
		MyTime::tic();
		cv::Mat output;
		cv::warpPerspective(im1, output, cv::Mat(3, 3, CV_32FC1, transMat), im1.size());
		MyTime::toc("perspective");
		cv::cvtColor(output, output, cv::COLOR_BGR2GRAY);
		// char transformed_name[50];
		// sprintf(transformed_name, "../result/transformed_cv.jpg");
		// cv::imwrite(transformed_name, output);

		cv::cvtColor(im2, im2, cv::COLOR_BGR2GRAY);
		cv::addWeighted(im2, 0.5, output, 0.5, 0.0, output);
		char fused_name[50];
		sprintf(fused_name, "../result/fused%d.jpg",i);
		cv::imwrite(fused_name, output);
	}

    return EXIT_SUCCESS;
}