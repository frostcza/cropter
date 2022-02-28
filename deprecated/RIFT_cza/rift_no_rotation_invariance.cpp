#include <vector>
#include <list>
#include <algorithm>
#include "rift_no_rotation_invariance.h"
#include "omp.h"

// print some elements in the Mat to debug
void displayMat(cv::Mat &mat_to_print, std::string name,int row = 0, int col = 0)
{
    printf("%s\n",name.c_str());
    for(int ii=col;ii<8+col;ii++)
    {
        printf("%f\t", mat_to_print.at<float>(row, ii));
    }
    printf("\n");
}

// // new version 
// struct KeypointResponseGreater
// {
//     inline bool operator()(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2) const
//     {
//         return kp1.response > kp2.response;
//     }
// };

// // old version 
bool sortFunction(cv::KeyPoint fir, cv::KeyPoint sec)
{
    return (fir.response > sec.response);  //from high value to low value
}

RIFT::RIFT(int _s, int _o, int _patch_size, int _thre, float _RATIO, cv::Size img_size){
    s = _s;
    o = _o;
    patch_size = _patch_size;
    thre = _thre;
    RATIO = _RATIO;
    pc.Init(img_size, s, o);
}

void RIFT::Inference(cv::Mat im1, cv::Mat im2, float transMat[3][3], int id)
{
    cv::Mat des_m1, des_m2;
    cv::Mat Homography;
    std::vector<cv::KeyPoint> kps1;
    std::vector<cv::KeyPoint> kps2;
    auto img_size = im1.size();

    std::vector<cv::DMatch> inlierMatches;
    des_m1 = RIFT_no_rotation_invariance(im1, kps1);
    des_m2 = RIFT_no_rotation_invariance(im2, kps2);
    
    // MyTime::tic();
    Homography = CalcMatch(des_m1, des_m2, kps1, kps2, inlierMatches);
    // MyTime::toc("Matcher + RANSAC");
    for (int i=0; i<3; i++)
    {
        for (int j=0; j<3; j++)
        {
            transMat[i][j] = Homography.at<double>(i,j);
        }
    }

    // cv::Mat img_matches;
    // cv::drawMatches(im1, kps1, im2, kps2, inlierMatches, img_matches, cv::Scalar(0, 255, 0), 2);
    // char name[50];
    // sprintf(name, "../result/img_match%d.jpg",id);
    // cv::imwrite(name, img_matches);

}


cv::Mat RIFT::RIFT_no_rotation_invariance(cv::Mat img, std::vector<cv::KeyPoint> &kps)
{
    cv::Mat img_gray, M_BGR;
    cv::Mat img_edges, img_corners;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    // MyTime::tic();
    pc.feature(img_gray, img_edges, img_corners);
    // MyTime::toc("get PC map");

    // 从pc获取maxMoment, 对M进行归一化操作
    cv::Mat M = pc.maxMoment;
    double minv = 0, maxv = 0;
    cv::minMaxIdx(M, &minv, &maxv);
    M = (M-minv)/(maxv-minv);
    // displayMat(M,"M");
    // printf("normalized M test %f \n", M.at<double>(0, 0));
    // 0.608627        0.190558        0.132847        0.144938        0.216635        0.263065        0.229081        0.336070

    cv::Mat M_8U;
    M.convertTo(M_8U, CV_8U, 255, 0);

    // FAST detector
    // cv::cuda::FastFeatureDetector runs slower than the version without cuda 
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(thre);
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(M_8U, keypoints);

    // 如果大于1000个点则清除，否则全部保留
    // 没必要用std::sort，只需获取response排名前1000的，而不要求这1000个keypoint有序
    // cv::KeyPointsFilter::retainBest()内部使用std::nth_element()，只找到第1000大的数，放在[999]位置，并把所有大于它的放在左边，所有小于它的放在右边
    // 存在一些response大小相等，retainBest()会把并列的也加进来，所以得到的keypoint.size()会大于1000
    // 好像没排序对结果有一定影响 why?

    // // new version, faster 
    // if(keypoints.size() > 1000)
    // {
    //     std::nth_element(keypoints.begin(), keypoints.begin() + 1000 - 1, keypoints.end(), KeypointResponseGreater());
    //     keypoints.resize(1000);
    // }

    // // old version, don't delete, useful for debug
    if(keypoints.size()>1000)
    {
        std::sort(keypoints.begin(), keypoints.end(), sortFunction); 
        keypoints.erase(keypoints.begin()+1000, keypoints.end());
    }
    // std::cout<<keypoints.size()<<std::endl;

    // RIFT 描述子构建
    cv::Mat des_im;
    des_im = RIFT_descriptor_no_rotation_invariance(img_gray, keypoints, kps, pc.eo);

    return des_im;
}

cv::Mat RIFT::RIFT_descriptor_no_rotation_invariance(cv::Mat im, std::vector<cv::KeyPoint> keypoints,std::vector<cv::KeyPoint> &kps, std::vector<cv::Mat> eo)
{
    /* Matlab in RIFT_descriptor_no_rotation_invariance.m
    CS = zeros(yim, xim, o); %convolution sequence
    for j=1:o
        for i=1:s
            CS(:,:,j)=CS(:,:,j)+abs(eo{i,j});
        end
    end
    [~, MIM] = max(CS,[],3); % MIM maximum index map
    */
    // 根据phase.cpp，eo是一个包含s*o个矩阵的向量
    // 旧版: eo中每个矩阵是包含实部和虚部的二通道矩阵
    // 新版: eo中每个矩阵是IDFT结果(复数)的模
    // Matlab中循环的含义是先取eo{i,j}中每个复数的模，然后将其沿s方向累加，原本s*o个Mat变为o个Mat，再沿o方向找到最大值所在的index，存入MIM
    // MIM中元素的取值只可能是[1,6]之间的整数
	
    // phase改了eo.push_back附近，本函数全部重写


    // MyTime::tic();
    // cv::cuda::GpuMat::add() runs slower than cpu version, maybe .upload() and .download() is time consuming

    cv::Mat sum;
    std::vector<cv::Mat> sum_s; // 沿s维累加所得的矩阵，一共o个
    for(int j=0;j<o;j++)
    {
        sum = cv::Mat::zeros(im.rows, im.cols, CV_64F);
        for(int i=0;i<s;i++)
        {
            cv::add(sum, eo[j*s+i], sum, cv::Mat(), CV_64F);
        }
        sum_s.push_back(sum.clone());
    }

    // 建立channel数为o的三维矩阵CS
    cv::Mat CS;
    cv::merge(sum_s, CS);
    // printf("%f\n",CS.ptr<double>(0,0)[1]);
    // std::cout<<CS.at<cv::Vec6d>(0,0)<<std::endl;
    // CS 24.0728 12.0530 测试正确
    
    // MyTime::toc("[RIFT descriptor] eo->CS");

    // MyTime::tic();

    // 求MIM索引图
    // 找最大值索引--用cv::minMaxIdx 速度不如std::max_element快

    cv::Mat MIM = cv::Mat(im.rows, im.cols, CV_8U);
    int index_max;
    double vec_for_max[o]; 
    for (int i = 0; i<im.rows; i++)
    {
        for (int j = 0; j<im.cols; j++)
        {
            // cv::minMaxIdx(CS.at<cv::Vec6d>(i,j), NULL, NULL, NULL, &index_max);
            // MIM.at<uchar>(i,j) = index_max + 1;
            for (int k=0;k<o;k++)
            {
                vec_for_max[k] = CS.ptr<double>(i,j)[k];
            }
            index_max = std::max_element(vec_for_max, vec_for_max+o) - vec_for_max;
            *MIM.ptr<uchar>(i,j) = (uchar)index_max + 1;
        }
    }

    // MyTime::toc("[RIFT descriptor] MIM");

    //MIM测试通过
    //displayMat(MIM, "MIM", 247,245);
    //printf("MIM test %d %d %d \n", MIM.at<uchar>(0, 0), MIM.at<uchar>(0, 5), MIM.at<uchar>(99, 45));

    // MyTime::tic();
    int size_ss = keypoints.size();
    int ns = 6; //这里的ns和nscales不是一个东西。后面会把patch再分成ns*ns的
    int size_R_des[] = {o, ns, ns};
    cv::Mat des(ns*ns*o, size_ss, CV_32F, cv::Scalar::all(0.0)); // 每个点的descriptor是216维的向量
    int count_kps_saved = 0; // 被保留下来的点数，以当前特征点为中心开一个(patch_size, patch_size)的窗，窗的某个边缘跑到原图之外的点会被舍弃
    cv::Mat kps_to_ignore = cv::Mat::zeros(1, size_ss, CV_8U);

    // matlab中最后一个大循环
#pragma omp parallel num_threads(4)
    #pragma omp for reduction(+:count_kps_saved) 
    for (int k=0; k<size_ss; k++)
    {
        cv::Mat RIFT_des = cv::Mat(3, size_R_des, CV_16SC(o), cv::Scalar::all(0.0));
        // cv::KeyPoint内部数据是float格式
        int x = (int)round(keypoints[k].pt.x); //pt.x is the cols, y is the cols
        int y = (int)round(keypoints[k].pt.y); //pt.y is the rows, x is the rows

        // c++下标从0开始
        int half_patch = floor(patch_size/2);
        int x1 = std::max(0, x-half_patch);
        int y1 = std::max(0, y-half_patch);
        int x2 = std::min(x+half_patch, im.cols-1);
        int y2 = std::min(y+half_patch, im.rows-1);

        //如果特征点周围这么大的地方两个边界都需要裁减  则忽略这个特征点
        if ((y2-y1)!=patch_size || (x2-x1)!=patch_size)
        {
            kps_to_ignore.at<uchar>(0,k) = 1;
            continue;
        }
        count_kps_saved ++;


        cv::Mat patch = MIM(cv::Range(y1,y2),cv::Range(x1,x2)); //裁剪 y1~y2行 x1-x2列 Range(a,b) means [a,b)
        int ys = patch.rows;
        int xs = patch.cols;

        // 直方图统计
        // rewrote hist by opencv
        // tried cv::cuda::histEven(), but it runs slower
        // to use cv::cuda, #include <opencv2/cudaimgproc.hpp> 

        cv::Mat hist;
        cv::Mat clip;
        float range[] = { 1,7 };
	    const float* histRanges = { range };
        for (int i=0; i<ns; i++)
        {
            for (int j=0; j<ns; j++)
            {
                clip = patch(cv::Range((int)(round(i*ys/ns)), (int)(round((i+1)*ys/ns))),
                                 cv::Range((int)(round(j*xs/ns)), (int)(round((j+1)*xs/ns)))); // Range(a,b) means [a,b)
                
                cv::calcHist(&clip, 1, 0, cv::Mat(), hist, 1, &o, &histRanges);
                hist.convertTo(hist, CV_16S);
                typedef cv::Vec<short, 6> Vec6s;
                RIFT_des.at<Vec6s>(i,j) = hist.clone();

                // // hist 测试通过 139     0       0       0       117     0
                // if (k==2 && i==0 && j==0)
                // {
                //     for(int hi=0;hi<6;hi++)
                //     {
                //         printf("%d\t",hist.at<short>(hi,0));
                //     }
                //     printf("\n");
                //     printf("%d,%d\n",hist.rows,hist.cols);
                //     std::cout<< hist.type()<<std::endl;;
                // }
            }
        }
        
        // // RIFT_des 测试通过 244     0       0       12      0       0
        // if (k==2)
        // {
        //     for(int hi=0;hi<6;hi++)
        //     {
        //         printf("%d\t",RIFT_des.at<cv::Vec<short, 6>>(4,4)[hi]);
        //     }
        //     printf("\n");
        // }

        // 优化了Mat访问方式和norm的求法
        short des_value;
        int des_sqare_sum = 0;
        for (int ch = 0; ch < o; ch++)
        {
            for (int j = 0; j < ns; j++)
            {
                for (int i = 0; i < ns; i++)
                {
                    /* old version
                     * des.ptr<float>(ch*ns*ns+j*ns+i)[k] = RIFT_des.ptr<short>(i)[j*o+ch];
                     */
                    des_value = RIFT_des.ptr<short>(i)[j*o+ch];
                    des.ptr<float>(ch*ns*ns+j*ns+i)[k] = des_value;
                    des_sqare_sum += des_value * des_value;
                }
            }
        }
        /* old version
         * des.col(k) /= cv::norm(des.col(k));
         */
        des.col(k) /= std::sqrt(des_sqare_sum);
    } // cycle k


    cv::Mat des_final(count_kps_saved, o*ns*ns, CV_32F, cv::Scalar::all(0.0));
    int index = -1;
    for(int k=0; k<size_ss; k++)
    {
        if(kps_to_ignore.at<uchar>(0,k) == 0)
        {
            index++;
            for(int i=0;i<ns*ns*o;i++)
            {
                des_final.ptr<float>(index)[i] = des.ptr<float>(i)[k];
            }
            kps.push_back(keypoints[k]);
        }
    }

    // 0.100092    0.002160    0.008641    0.058327    0.141857    0.184343    0.150498    0.059767
    // displayMat(des_final,"des_final");
    // printf("%d\n", count_kps_saved);
    // printf("%d,%d\n",des_final.rows,des_final.cols);
    // printf("%d\n",(int)kps.size());

    // MyTime::toc("[RIFT descriptor] big cycle");

    return des_final;
    // 返回的特征描述对应des_final的一行
}

cv::Mat RIFT::CalcMatch(cv::Mat& des_m1, cv::Mat& des_m2, std::vector<cv::KeyPoint> &kps1, std::vector<cv::KeyPoint> &kps2, std::vector<cv::DMatch>& inlierMatches)
{
    // MyTime::tic();

    // Matcher有BFMatch和FlannBasedMatcher两种可用
    // BFMatch对应Matlab中matchFeatures('Method'='Exhaustive'), FlannBasedMatcherFlann对应Matlab中matchFeatures('Method'='Approximate')
    // Matlab代码使用前者
    cv::BFMatcher matcher;
    
    // vector<DMatch> matches是存储匹配结果的数据结构。关键用法如下
    // matches.size() 返回匹配对数
    // matches[i].queryIdx : 查询点的索引（当前要寻找匹配结果的点在它所在图片上的索引）。对应第一张图中的点
    // matches[i].trainIdx : 被匹配到的点的索引。对应第二张图中的点
    // matches[i].distance ：两个描述子之间的距离
    std::vector<cv::DMatch> matches; 
    
    // 匹配方法有mathcer.match()和matcher.knnMatch()，区别是后者可以指定返回k个最好的匹配，前者k=1
    // Matlab代码用的是mathcer.match()，只寻找一个最好的

    matcher.match(des_m1, des_m2, matches);
    // MyTime::toc("Matcher");

    // 这里要用特征的index 求原图中点的坐标
    // 举例：A图第1个特征对应B图第1822个特征，A图第1个特征的对应点坐标为(119,272)，B图第1822个特征的对应点坐标为(184,279)
    // 则matchedPoints1第1行存(119,272)，matchedPoints2第1行存(184,279)

    // MyTime::tic();
   	std::vector<int> queryIdxs(matches.size());
    std::vector<int> trainIdxs(matches.size());
	for (size_t i = 0; i < matches.size(); i++)
	{
		queryIdxs[i] = matches[i].queryIdx; //取出m1图片中匹配的点对的索引即id号
		trainIdxs[i] = matches[i].trainIdx; //取出m2图片中匹配的点对的索引即id号
	}

    std::vector<cv::Point2f> matchedPoints1; 
	cv::KeyPoint::convert(kps1, matchedPoints1, queryIdxs);//KeyPoint根据索引转point2f坐标
	std::vector<cv::Point2f> matchedPoints2; 
	cv::KeyPoint::convert(kps2, matchedPoints2, trainIdxs);

    std::vector<char> matchesMask(matches.size(), 0); //作为内点标记使用，1表示内点
    cv::Mat H;
	H = cv::findHomography(matchedPoints1, matchedPoints2, cv::RANSAC, 3, matchesMask, 2000); //计算单应矩阵，其中包含RANSAC
    // 适当减少RANSAC迭代次数可节省大量时间
    // H = cv::findHomography(matchedPoints1, matchedPoints2, cv::RANSAC, 3, matchesMask, 10000);

    // printf("homography matrix\n");
    // for (int i=0; i<3; i++)
    // {
    //     for (int j=0; j<3; j++)
    //     {
    //         printf("%f\t",H.at<double>(i,j));
    //     }
    //     printf("\n");
    // }

    for (int i = 0; i < matches.size(); i++)
    {
        if (matchesMask[i] != 0)
        {
            inlierMatches.push_back(matches[i]);
        }
    }
    // MyTime::toc("RANSAC");
    // printf("Matches num %d\n", (unsigned int)matches.size());
    // printf("inlierMatches num %d\n", (unsigned int)inlierMatches.size());

    return H;
}
