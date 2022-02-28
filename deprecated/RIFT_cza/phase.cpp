#include "phase.h"
#include "bfprt.h"
#include <string>

#define MAT_TYPE CV_64FC1
#define MAT_TYPE_CNV CV_64F

// using namespace cv;

struct timeval MyTime::start, MyTime::end;

// // print some elements in the Mat to debug
// void displayMat(cv::Mat &mat_to_print, std::string name,int row = 0, int col = 0)
// {
//     printf("%s\n",name.c_str());
//     for(int ii=col;ii<8+col;ii++)
//     {
//         printf("%f\t", mat_to_print.at<double>(row, ii));
//     }
//     printf("\n");
// }


// Rearrange the quadrants of Fourier image so that the origin is at the image center
void shiftDFT(cv::InputArray _src, cv::OutputArray _dst)
{
    // cv::InputArray 和 cv::OutputArray是代理类
    // 做cv2函数时要用cv::InputArray指定输入矩阵的形参类型, cv::OutputArray指定输出矩阵的形参类型
    // 这样cv::InputArray可以接收cv::Mat, cv::Matx, std::vector<int>等各种类型的数据, cv::OutputArray同理
    
    cv::Mat src = _src.getMat(); // 对于输入, 先要用.getMat()将cv::InputArray转换为Mat
    cv::Size size = src.size(); 

    _dst.create(size, src.type());
    auto dst = _dst.getMat(); // 对于输出, 先要用.create()创建一个Mat

    const int cx = size.width / 2;
    const int cy = size.height / 2; // image center

    cv::Mat s1 = src(cv::Rect(0, 0, cx, cy));
    cv::Mat s2 = src(cv::Rect(cx, 0, cx, cy));
    cv::Mat s3 = src(cv::Rect(cx, cy, cx, cy));
    cv::Mat s4 = src(cv::Rect(0, cy, cx, cy));

    cv::Mat d1 = dst(cv::Rect(0, 0, cx, cy));
    cv::Mat d2 = dst(cv::Rect(cx, 0, cx, cy));
    cv::Mat d3 = dst(cv::Rect(cx, cy, cx, cy));
    cv::Mat d4 = dst(cv::Rect(0, cy, cx, cy));

    cv::Mat tmp;
    s3.copyTo(tmp);
    s1.copyTo(d3);
    tmp.copyTo(d1);

    s4.copyTo(tmp);
    s2.copyTo(d4);
    tmp.copyTo(d2);
}

PhaseCongruency::PhaseCongruency(){
    sigma = -1.0 / (2.0 * log(0.75) * log(0.75));
    mult = 1.6;  //1.6
    minwavelength = 3;
    epsilon = 0.0001;
    cutOff = 0.5;
    g = 3;  //3
    k = 1.0;  //1
}

// Making filters
void PhaseCongruency::Init(cv::Size _size, size_t _nscale, size_t _norient)
{
    size = _size;
    nscale = _nscale;
    norient = _norient;

    filter.resize(nscale * norient);

    const int dft_M = cv::getOptimalDFTSize(_size.height); //nrows
    const int dft_N = cv::getOptimalDFTSize(_size.width);  //ncols

    cv::Mat radius = cv::Mat::zeros(dft_M, dft_N, MAT_TYPE);
    cv::Mat normradius = cv::Mat::zeros(dft_M, dft_N, MAT_TYPE);
    cv::Mat matAr[2];
    matAr[0] = cv::Mat::zeros(dft_M, dft_N, MAT_TYPE);
    matAr[1] = cv::Mat::zeros(dft_M, dft_N, MAT_TYPE);
    cv::Mat lp = cv::Mat::zeros(dft_M, dft_N, MAT_TYPE);
    cv::Mat angular = cv::Mat::zeros(dft_M, dft_N, MAT_TYPE);
    std::vector<cv::Mat> gabor(nscale);

    //=============================Initialization of radius(Edited)============================

    const int dft_M_2 = floor(dft_M / 2);
    const int dft_N_2 = floor(dft_N / 2);

    const double dr_y = 1.0 / static_cast<double>(dft_M);
    const double dr_x = 1.0 / static_cast<double>(dft_N);
    for (int row = 0; row < dft_M; row++)
    {
        auto radius_row = radius.ptr<double>(row); // cv::Mat.ptr<type>(a)[b]是一个指针, 类型为type, 指向Mat中第a行第b个元素
        for (int col = 0; col < dft_N; col++)
        {
            double m = (row - dft_M_2);
            double n = (col - dft_N_2);
            m = m * dr_y;
            n = n * dr_x;
            /* Matlab 162~177
            Set up X and Y matrices with ranges normalised to +/- 0.5
            [x,y] = meshgrid(xrange, yrange); 
            */

            radius_row[col] = sqrt(static_cast<double>(m * m + n * n));
            /* Matlab 179
            radius = sqrt(x.^2 + y.^2); 
            % Matrix values contain *normalised* radius from centre. 
            ??? where is the ifftshift
            */
        }
    }

    radius.at<double>(dft_M_2, dft_N_2) = 1.0; 
    shiftDFT(radius, radius);
    /* Matlab 186
    radius(1,1) = 1; 
    按照上面对radius的计算, 在0频率(中心位置)的值为0, 人为赋1.0防止后续log计算出错
    */

    normradius = radius * 1.0; //abs(x).max()*2, abs(x).max() is 0.5
    lp = normradius / 0.45;
    cv::pow(lp, 30.0, lp);
    lp += 1.0;
    /* Matlab 73 in lowpassfilter.m
    f = ifftshift( 1.0 ./ (1.0 + (radius ./ cutoff).^(2*n)) );   % The filter 
    cutoff = 0.45; n = 15;
    */


    // The following implements the log-gabor transfer function.
    double mt = 1.0f; 
    for (int scale = 0; scale < nscale; scale++)
    {
        const double wavelength = minwavelength * mt;  //wavelength = minWavelength*mult*s
        gabor[scale] = radius * wavelength;
        cv::log(gabor[scale], gabor[scale]);
        cv::pow(gabor[scale], 2.0, gabor[scale]); //log(radius/fo)**2
        gabor[scale] *= sigma;
        cv::exp(gabor[scale], gabor[scale]);

        cv::divide(gabor[scale], lp, gabor[scale]);  //logGabor*lowpassbutterworth
        gabor[scale].at<double>(0, 0) = 0.0;
        // for(int ii=0;ii<5;ii++)
        // {
        //     printf("gabor: %f\t",gabor[scale].at<double>(30, ii));
        // }
        // printf("\n");
        //gabor[scale].at<double>(dft_M_2, dft_N_2) = 0.0;
        //In python lp is reversed and here not, so use divide.
        mt = mt * mult;
    }
    /* Matlab 211~218
    for s = 1:nscale
        wavelength = minWaveLength*mult^(s-1);
        fo = 1.0/wavelength;                  % Centre frequency of filter.
        logGabor{s} = exp((-(log(radius/fo)).^2) / (2 * log(sigmaOnf)^2));  
        logGabor{s} = logGabor{s}.*lp;        % Apply low-pass filter
        logGabor{s}(1,1) = 0;                 % Set the value at the 0 frequency point of the filter
                                              % back to zero (undo the radius fudge).
    end 
    */

    const double angle_const = static_cast<double>(M_PI) / static_cast<double>(norient);  //pi/6
    for (int ori = 0; ori < norient; ori++)
    {
        double angl = (double)ori * angle_const;
        //Now we calculate the angular component that controls the orientation selectivity of the filter.
        for (int i = 0; i < dft_M; i++)  //nrows
        {
            auto angular_row = angular.ptr<double>(i); // angular <=> spread
            for (int j = 0; j < dft_N; j++)  //ncols
            {
                double m = atan2(-((double)i / (double)dft_M - 0.5), (double)j / (double)dft_N - 0.5); 
                /*
                Matlab 180 theta = atan2(-y,x); 
                ??? where is the ifftshift
                */
                double s = sin(m);
                double c = cos(m);
                m = s * cos(angl) - c * sin(angl); //ds
                double n = c * cos(angl) + s * sin(angl); //dc
                s = fabs(atan2(m, n)); //dtheta
                /* Matlab 228~230
                ds = sintheta * cos(angl) - costheta * sin(angl);    % Difference in sine.
                dc = costheta * cos(angl) + sintheta * sin(angl);    % Difference in cosine.
                dtheta = abs(atan2(ds,dc));                          % Absolute angular distance.
                */
                angular_row[j] = (cos(std::min(s * (double)norient * 0.5, M_PI)) + 1.0) * 0.5; 
                /* Matlab 232~235
                dtheta = min(dtheta*norient/2,pi);
                spread = (cos(dtheta)+1)/2; 
                */
            }
        }
        
        shiftDFT(angular, angular);

        for (int scale = 0; scale < nscale; scale++)
        {
            cv::multiply(gabor[scale], angular, matAr[0]); //Product of the two components. the corresponding "filter"
            cv::merge(matAr, 2, filter[nscale * ori + scale]); 
            // merge()将这个指针上的Mat在channel维堆叠, 就是把全0的矩阵放在channel=1上
            // 原因是filter在calc中与原图相乘后参与了dft运算, cv2做dft时必须在其后pad一个全零的通道, 用于同时接收实部和虚部
            // Here, problem with merge()
            // matAr[0] and [1] represent the real part and imag part

            /* Matlab 243
            filter = logGabor{s} .* spread;
            */
        }//scale
    }//orientation
    //Filter ready
    // std::cout << filter.size() << std::endl;
}


//Phase congruency calculation
void PhaseCongruency::calc(cv::InputArray _src, std::vector<cv::Mat> &_pc)
{
    cv::Mat src = _src.getMat();

    CV_Assert(src.size() == size);

    const int width = size.width, height = size.height;

    cv::Mat src64;
    src.convertTo(src64, MAT_TYPE_CNV, 1.0 / 1.0);

    const int dft_M_r = cv::getOptimalDFTSize(src.rows) - src.rows; // actually, it is the padding size 
    const int dft_N_c = cv::getOptimalDFTSize(src.cols) - src.cols;

    _pc.resize(norient);

    //std::vector<Mat> eo(nscale);
    // THE eo. And here I expand it to a (o,s) vector according to former matlab version.
    //std::vector<vector<Mat> >eo(norient, vector<Mat>(nscale));
    // New update: defined in header file.
    //std::vector<vector<cv::Mat> >eo = std::vector<vector<cv::Mat> >(6, vector<Mat>(4));
    std::vector<cv::Mat> trans(nscale);

    cv::Mat complex[2];
    cv::Mat sumAn;
    cv::Mat sumRe;
    cv::Mat sumIm;
    cv::Mat maxAn;
    cv::Mat xEnergy;
    cv::Mat tmp;
    cv::Mat tmp1;
    cv::Mat tmp2;
    cv::Mat energy = cv::Mat::zeros(size, MAT_TYPE);

    //expand input image to optimal size
    cv::Mat padded;
    cv::copyMakeBorder(src64, padded, 0, dft_M_r, 0, dft_N_c, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = { cv::Mat_<double>(padded), cv::Mat::zeros(padded.size(), MAT_TYPE_CNV) };

    cv::Mat dft_A;
    cv::merge(planes, 2, dft_A);         // Add to the expanded another plane with zeros
    cv::dft(dft_A, dft_A);
    //shiftDFT(dft_A, dft_A);
    eo.clear();
    
    double noise = 0;
    for (unsigned int o = 0; o < norient; o++)
    {
        for (unsigned int scale = 0; scale < nscale; scale++)
        {
            cv::Mat filtered;
            cv::mulSpectrums(dft_A, filter[nscale * o + scale], filtered, 0); // Convolution
            cv::dft(filtered, filtered, cv::DFT_INVERSE + cv::DFT_SCALE);
            // deprecate: 旧版把trans[scale]存入eo，包含实部和虚部2个通道
            trans[scale] = filtered(cv::Rect(0, 0, width, height)).clone();
            //displayMat(trans[scale], "trans[scale]");
            //eo.push_back(trans[scale]);

            cv::split(trans[scale], complex);
            cv::Mat eo_mag;
            //sumAn_ThisOrient
            cv::magnitude(complex[0], complex[1], eo_mag);
            // new: 新版把复数的模存入eo
            eo.push_back(eo_mag.clone());

            /* Matlab 243~252
            EO{s,o} = ifft2(imagefft .* filter);      
            An = abs(EO{s,o});
            */
            if (scale == 0)
            {
                // noise threshold calculation
                // note: original operation is MEDIAN, but it is time consuming, so we consider to replace it by MEAN
                # if 1
                //MyTime::tic();
                std::vector<double> eo_mag_vec = (std::vector<double>)(eo_mag.reshape(1, 1));
                BFPRT bfprt(eo_mag_vec, eo_mag_vec.size()/2);
                double tau = bfprt.KthMin();
                tau = tau / sqrt(log(4.0));
                //MyTime::toc("median");

                auto mt = 1.0 * pow(mult, nscale);
                auto totalTau = tau * (1.0 - 1.0 / mt) / (1.0 - 1.0 / mult);
                # else
                auto tau = cv::mean(eo_mag);
                tau.val[0] = tau.val[0] / sqrt(log(4.0));
                auto mt = 1.0 * pow(mult, nscale);
                auto totalTau = tau.val[0] * (1.0 - 1.0 / mt) / (1.0 - 1.0 / mult);
                # endif
                

                auto m = totalTau * sqrt(M_PI / 2.0);
                //EstNoiseEnergyMean
                auto n = totalTau * sqrt((4 - M_PI) / 2.0);
                //EstNoiseEnergySigma: values of noise energy
                noise = m + k * n;
                // T: noise threshold
                /* Matlab
                if noiseMethod == -1     % Use median to estimate noise statistics
                    tau = median(sumAn_ThisOrient(:))/sqrt(log(4));   
                totalTau = tau * (1 - (1/mult)^nscale)/(1-(1/mult));
                EstNoiseEnergyMean = totalTau*sqrt(pi/2);        % Expected mean and std
                EstNoiseEnergySigma = totalTau*sqrt((4-pi)/2);   % values of noise energy
                T =  EstNoiseEnergyMean + k*EstNoiseEnergySigma;
                */

                eo_mag.copyTo(maxAn);
                eo_mag.copyTo(sumAn);
                complex[0].copyTo(sumRe);
                complex[1].copyTo(sumIm);
            }
            else
            {
                cv::add(sumAn, eo_mag, sumAn);
                cv::max(eo_mag, maxAn, maxAn);
                cv::add(sumRe, complex[0], sumRe);
                cv::add(sumIm, complex[1], sumIm);
                /* Matlab 250~252,268
                sumAn_ThisOrient = sumAn_ThisOrient + An;  % Sum of amplitude responses.
                sumE_ThisOrient = sumE_ThisOrient + real(EO{s,o}); % Sum of even filter convolution results.
                sumO_ThisOrient = sumO_ThisOrient + imag(EO{s,o}); % Sum of odd filter convolution results.
                maxAn = max(maxAn,An); 
                */
            }
        } // next scale

        cv::magnitude(sumRe, sumIm, xEnergy);
        xEnergy += epsilon;
        cv::divide(sumIm, xEnergy, sumIm);
        cv::divide(sumRe, xEnergy, sumRe);
        /* Matlab 280~282
        XEnergy = sqrt(sumE_ThisOrient.^2 + sumO_ThisOrient.^2) + epsilon;   
        MeanE = sumE_ThisOrient ./ XEnergy; 
        MeanO = sumO_ThisOrient ./ XEnergy;
        */
        energy.setTo(0);
        for (int scale = 0; scale < nscale; scale++)
        {
            cv::split(trans[scale], complex);
            cv::multiply(complex[0], sumIm, tmp1);
            cv::multiply(complex[1], sumRe, tmp2);
            cv::absdiff(tmp1, tmp2, tmp);
            cv::subtract(energy, tmp, energy);
            cv::multiply(complex[0], sumRe, complex[0]);
            cv::add(energy, complex[0], energy);
            cv::multiply(complex[1], sumIm, complex[1]);
            cv::add(energy, complex[1], energy);

            /* Matlab 290~292
            E = real(EO{s,o}); O = imag(EO{s,o});
            Energy = Energy + E.*MeanE + O.*MeanO - abs(E.*MeanO - O.*MeanE);
            */

        } //next scale
        trans.clear();

        energy -= noise; // -noise
        cv::max(energy, 0.0, energy);
        maxAn += epsilon;
        cv::divide(sumAn, maxAn, tmp);
        tmp = -1.0 / (static_cast<double>(nscale) - 1.0) * (tmp - 1.0);
        tmp += cutOff;
        tmp = tmp * g;
        cv::exp(tmp, tmp);
        tmp += 1.0; // tmp = 1/weight
        /* Matlab 336~347
        Energy = max(Energy - T, 0);         
        width = (sumAn_ThisOrient./(maxAn + epsilon) - 1) / (nscale-1);    
        weight = 1.0 ./ (1 + exp( (cutOff - width)*g));
        */

        //PC
        cv::multiply(tmp, sumAn, tmp);
        cv::divide(energy, tmp, _pc[o]);
        //displayMat(_pc[o],"_pc[o]");
        /* Matlab
        weight = 1.0 ./ (1 + exp( (cutOff - width)*g)); 
        PC{o} = weight.*Energy./sumAn_ThisOrient;
        */
    }//orientation
}

//Build up covariance data for every point
void PhaseCongruency::feature(std::vector<cv::Mat>& _pc, cv::OutputArray _edges, cv::OutputArray _corners)
{
    _edges.create(size, CV_8UC1);
    _corners.create(size, CV_8UC1);
    auto edges = _edges.getMat();
    auto corners = _corners.getMat();

    cv::Mat covx2 = cv::Mat::zeros(size, MAT_TYPE);
    cv::Mat covy2 = cv::Mat::zeros(size, MAT_TYPE);
    cv::Mat covxy = cv::Mat::zeros(size, MAT_TYPE);
    cv::Mat cos_pc, sin_pc, mul_pc;

    const double angle_const = M_PI / static_cast<double>(norient);

    for (unsigned o = 0; o < norient; o++)
    {
        auto angl = static_cast<double>(o) * angle_const;
        cos_pc = _pc[o] * cos(angl);
        sin_pc = _pc[o] * sin(angl);
        cv::multiply(cos_pc, sin_pc, mul_pc);
        cv::add(covxy, mul_pc, covxy);
        cv::pow(cos_pc, 2, cos_pc);
        cv::add(covx2, cos_pc, covx2);
        cv::pow(sin_pc, 2, sin_pc);
        cv::add(covy2, sin_pc, covy2);
        /* Matlab
        covx = PC{o}*cos(angl);
        covy = PC{o}*sin(angl);
        covx2 = covx2 + covx.^2;
        covy2 = covy2 + covy.^2;
        covxy = covxy + covx.*covy;
        */
    } // next orientation

      //Edges calculations
    covx2 *= 2.0 / static_cast<double>(norient);
    covy2 *= 2.0 / static_cast<double>(norient);
    covxy *= 4.0 / static_cast<double>(norient);
    cv::Mat sub;
    cv::subtract(covx2, covy2, sub);

    //denom += Scalar::all(epsilon);
    cv::Mat denom;
    cv::magnitude(sub, covxy, denom); // denom;
    denom += epsilon;
    cv::Mat sum;
    cv::add(covy2, covx2, sum);
    cv::Mat minMoment;
    cv::subtract(sum, denom, minMoment);
    cv::add(sum, denom, maxMoment);
    /* Matlab
    covx2 = covx2/(norient/2);
    covy2 = covy2/(norient/2);
    covxy = 4*covxy/norient;   % This gives us 2*covxy/(norient/2)
    denom = sqrt(covxy.^2 + (covx2-covy2).^2)+epsilon;
    M = (covy2+covx2 + denom)/2;          % Maximum moment
    m = (covy2+covx2 - denom)/2; 
    */
    maxMoment *= 0.5;
    minMoment *= 0.5;
    //displayMat(maxMoment,"maxMoment");
    maxMoment.convertTo(edges, CV_8U, 255);
    minMoment.convertTo(corners, CV_8U, 255);

}

//Build up covariance data for every point
void PhaseCongruency::feature(cv::InputArray _src, cv::OutputArray _edges, cv::OutputArray _corners)
{
    std::vector<cv::Mat> pc;
    calc(_src, pc);
    feature(pc, _edges, _corners);
}