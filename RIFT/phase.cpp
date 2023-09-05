#include "phase.h"
#include "omp.h"

#define MAT_TYPE CV_64FC1
#define MAT_TYPE_CNV CV_64F

struct timeval MyTime::start, MyTime::end;

void shiftDFT(cv::InputArray _src, cv::OutputArray _dst)
{
    cv::Mat src = _src.getMat();
    cv::Size size = src.size();

    _dst.create(size, src.type());
    auto dst = _dst.getMat();

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

PhaseCongruency::PhaseCongruency() {
    sigma = -1.0 / (2.0 * log(0.75) * log(0.75));//这里的sigma 是后面为了减少计算量改出来的
    mult = 1.6;  //1.6
    minwavelength = 3;
    epsilon = 0.0001;
    cutOff = 0.5; //注意不要将此cutoff和低通滤波器弄混
    g = 3;  //3
    k = 1.0;  //1
    std::cout << "PhaseCongruency params done" << std::endl;
}

// 初始化 以及warm up
void PhaseCongruency::Init(cv::Size _size, size_t _nscale, size_t _norient)
{
    size = _size;
    nscale = _nscale;
    norient = _norient;

    filter.resize(nscale * norient);
    filter_cpu.resize(nscale * norient);
    eo.resize(nscale*norient);
    const int dft_M = cv::getOptimalDFTSize(_size.height); //nrows
    const int dft_N = cv::getOptimalDFTSize(_size.width);  //ncols

//===============================================gpu malloc===========================================================//
    timeval s1,e1;
    cufftPlan2d(&plan_forward,dft_M,dft_N,CUFFT_Z2Z);
    cufftPlan2d(&plan_inverse, dft_M, dft_N, CUFFT_Z2Z);

    filter_host = (double **)malloc(sizeof(double*)*nscale * norient);
    filter_gpu = (double **)malloc(sizeof(double*)*nscale * norient);
    filtered_gpu = (cufftDoubleComplex **)malloc(sizeof(cufftDoubleComplex*)*nscale*norient);
    ifft_img_gpu = (cufftDoubleComplex **)malloc(sizeof(cufftDoubleComplex*)*nscale*norient);
    An =(double**)malloc(sizeof(double*)*nscale*norient);
    An_cpu=(double**)malloc(sizeof(double*)*nscale*norient);
    // 把所有的malloc放到这里并且进行同步 warmup
    host_data = (double *)malloc(sizeof(double)*dft_M* dft_N);
    cudaAllocMapped((void**)&idata, 300*sizeof(double)* dft_M* dft_N);
    cudaAllocMapped((void**)&odata, sizeof(cufftDoubleComplex)* dft_M* dft_N);
    cudaAllocMapped((void**)&img_complex_gpu, sizeof(cufftDoubleComplex)*dft_M* dft_N);

    cudaAllocMapped((void**)&sum_re,sizeof(double)*dft_M* dft_N);
    cudaAllocMapped((void**)&sum_im,sizeof(double)*dft_M* dft_N);
    cudaAllocMapped((void**)&sum_an,sizeof(double)*dft_M* dft_N);
    cudaAllocMapped((void**)&max_an,sizeof(double)*dft_M* dft_N);
    cudaAllocMapped((void**)&complex0,sizeof(double)*dft_M* dft_N);
    cudaAllocMapped((void**)&complex1,sizeof(double)*dft_M* dft_N);
    cudaAllocMapped((void**)&energy,sizeof(double)*dft_M* dft_N);
    cudaAllocMapped((void**)&convx,sizeof(double)*dft_M* dft_N);
    cudaAllocMapped((void**)&convy,sizeof(double)*dft_M* dft_N);
    cudaAllocMapped((void**)&convx2,sizeof(double)*dft_M* dft_N);
    cudaAllocMapped((void**)&convy2,sizeof(double)*dft_M* dft_N);
    cudaAllocMapped((void**)&convxy,sizeof(double)*dft_M* dft_N);
    cudaAllocMapped((void**)&Max_moment,sizeof(double)*dft_M* dft_N);
    pc_gpu = (double **)malloc(sizeof(double*)*norient);
    
    double noise;
    gettimeofday(&s1, NULL);
    for (unsigned int o = 0; o < norient; o++)
    {

        cudaAllocMapped((void **)&pc_gpu[o],sizeof(double)*dft_M*dft_N);
        for (unsigned int scale = 0; scale < nscale; scale++){
            cudaAllocMapped((void**)&filter_gpu[nscale * o + scale], sizeof(double)* dft_M* dft_N);
            cudaAllocMapped((void**)&filtered_gpu[nscale * o + scale],sizeof(cufftDoubleComplex)*dft_M*dft_N);   
            cudaAllocMapped((void**)&ifft_img_gpu[nscale * o + scale],sizeof(cufftDoubleComplex)*dft_M*dft_N);        
            cudaAllocMapped((void **)&An[nscale * o + scale],sizeof(double)*dft_M*dft_N);

//=============================warm up 使用假数据进行热身 方便测试时间 实际使用可以关闭 =============================== 
            cuda_mulSpectrums(odata,filter_gpu[nscale * o + scale],filtered_gpu[nscale * o + scale],dft_N,dft_M);
            cufftExecZ2Z(plan_inverse,(cufftDoubleComplex *)filtered_gpu[nscale * o + scale],(cufftDoubleComplex *)ifft_img_gpu[nscale * o + scale],CUFFT_INVERSE);    
            cuda_calc_eo(ifft_img_gpu[nscale * o + scale],dft_N,dft_M);
            // cudaDeviceSynchronize();
            // cudaDeviceSynchronize();
            cuda_calc_eomag(ifft_img_gpu[nscale * o + scale],An[nscale * o + scale],dft_N,dft_M);
            cudaDeviceSynchronize();
            eo[nscale * o + scale] =cv::Mat(dft_M,dft_N,CV_64F, (double*)An[nscale * o + scale]);
            if(scale==0){
                auto tau = mean(eo[nscale * o + scale]);
                tau.val[0] = tau.val[0] / sqrt(log(4.0));
                auto mt = 1.0 * pow(mult, nscale);
                auto totalTau = tau.val[0] * (1.0 - 1.0 / mt) / (1.0 - 1.0 / mult);
                auto m = totalTau * sqrt(M_PI / 2.0);
                auto n = totalTau * sqrt((4 - M_PI) / 2.0);
                noise = m + k * n;
                cuda_copy_to(An[nscale * o],max_an,dft_N,dft_M);
                cuda_copy_to(An[nscale * o],sum_an,dft_N,dft_M);
                cuda_splitZ(ifft_img_gpu[nscale * o],sum_re,sum_im,dft_N,dft_M);               
            }
            else{
                cuda_splitZ(ifft_img_gpu[nscale * o + scale],complex0,complex1,dft_N,dft_M);
                cuda_self_add(sum_an,An[nscale * o + scale],dft_N,dft_M);
                cuda_self_add(sum_re,complex0,dft_N,dft_M);
                cuda_self_add(sum_im,complex1,dft_N,dft_M);
                cuda_self_max(max_an,An[nscale * o + scale],dft_N,dft_M);             
            }
           
        }
        cuda_norm_orient(sum_re,sum_im,epsilon,dft_N,dft_M);
        // sum_re=mean E  sum_im=mean_O
        cuda_zero(energy,dft_N,dft_M);
        for (int scale = 0; scale < nscale; scale++)
        {
            cuda_splitZ(ifft_img_gpu[nscale * o + scale],complex0,complex1,dft_N,dft_M);
            cuda_clac_energy(sum_re,sum_im,complex0,complex1,energy,dft_N,dft_M);          
        } 
        cuda_max_R(energy,noise,dft_N,dft_M);
        cuda_calc_pc(sum_an,max_an,energy,pc_gpu[o],cutOff,g,epsilon,nscale,dft_N,dft_M);
        // cudaDeviceSynchronize();
    }// for o
    cuda_zero(convx2,dft_N,dft_M);
    cuda_zero(convy2,dft_N,dft_M);
    cuda_zero(convxy,dft_N,dft_M);
    double angle = M_PI / static_cast<double>(norient);
    for (unsigned int o = 0; o < norient; o++)
    {
        auto angl = static_cast<double>(o) * angle;
        cuda_calc_convx(pc_gpu[o],angl,convx,dft_N,dft_M);
        cuda_calc_convy(pc_gpu[o],angl,convy,dft_N,dft_M);
        cuda_add_convx2(convx,convx2,norient,dft_N,dft_M);
        cuda_add_convy2(convy,convy2,norient,dft_N,dft_M);
        cuda_add_convxy(convx,convy,convxy,norient,dft_N,dft_M);
    } 
    // cudaDeviceSynchronize();
    cuda_calc_M(convx2,convy2,convxy,epsilon,Max_moment,dft_N,dft_M);
    // cudaDeviceSynchronize();
    gettimeofday(&e1, NULL);
    double timeUsed;
    timeUsed = 1000000 * (e1.tv_sec - s1.tv_sec) + e1.tv_usec - s1.tv_usec;
    std::cout << "[RIFT-PC] warm up Time = " << timeUsed / 1000 << " ms" << std::endl;


//===============================================构造滤波器幅度===============================================//
    cv::Mat radius = cv::Mat::zeros(dft_M, dft_N, MAT_TYPE);
    cv::Mat normradius = cv::Mat::zeros(dft_M, dft_N, MAT_TYPE);
    cv::Mat matAr[2];
    matAr[0] = cv::Mat::zeros(dft_M, dft_N, MAT_TYPE);
    matAr[1] = cv::Mat::zeros(dft_M, dft_N, MAT_TYPE);
    cv::Mat lp = cv::Mat::zeros(dft_M, dft_N, MAT_TYPE);
    cv::Mat angular = cv::Mat::zeros(dft_M, dft_N, MAT_TYPE);
    std::vector<cv::Mat> gabor(nscale);

    const int dft_M_2 = floor(dft_M / 2);  //cy
    const int dft_N_2 = floor(dft_N / 2);  //cx

    const double dr_y = 1.0 / static_cast<double>(dft_M);
    const double dr_x = 1.0 / static_cast<double>(dft_N);
    for (int row = 0; row < dft_M; row++)
    {
        auto radius_row = radius.ptr<double>(row);
        for (int col = 0; col < dft_N; col++)
        {
            double m = (row - dft_M_2);  //y-cy
            double n = (col - dft_N_2);  //x-cx
            m = m * dr_y;
            n = n * dr_x;
            radius_row[col] = sqrt(static_cast<double>(m * m + n * n));
        }
    }

    shiftDFT(radius, radius);
    radius.at<double>(0, 0) = 1.0; 

    normradius = radius * 1.0; //abs(x).max()*2, abs(x).max() is 0.5

    lp = normradius / 0.45;
    pow(lp, 30.0, lp);
    lp += cv::Scalar::all(1.0);

    double mt = 1.0f;
    for (int scale = 0; scale < nscale; scale++)
    {
        const double wavelength = minwavelength * mt;  //wavelength = minWavelength*mult^(s-1) 
        //省略了f0计算步骤
        gabor[scale] = radius * wavelength;
        log(gabor[scale], gabor[scale]);
        pow(gabor[scale], 2.0, gabor[scale]); //log(radius/fo)**2
        gabor[scale] *= sigma;
        exp(gabor[scale], gabor[scale]);

        divide(gabor[scale], lp, gabor[scale]);  //logGabor*lowpassbutterworth
        //gabor[scale].at<double>(0, 0) = 0.0;
        gabor[scale].at<double>(0, 0) = 0.0;

        mt = mt * mult;
       
    }

//==================================================构造滤波器角度部分===============================// 
    const double angle_const = static_cast<double>(M_PI) / static_cast<double>(norient);  //pi/6
    for (int ori = 0; ori < norient; ori++)
    {
        double angl = (double)ori * angle_const;
        for (int i = 0; i < dft_M; i++)  //nrows
        {
            auto angular_row = angular.ptr<double>(i);
            for (int j = 0; j < dft_N; j++)  //ncols
            {
                double m = atan2(-((double)i / (double)dft_M - 0.5), (double)j / (double)dft_N - 0.5);
                double s = sin(m);
                double c = cos(m);
                m = s * cos(angl) - c * sin(angl);
                //ds
                double n = c * cos(angl) + s * sin(angl);
                //dc
                s = fabs(atan2(m, n));
                //dtheta
                angular_row[j] = (cos(std::min(s * (double)norient * 0.5, M_PI)) + 1.0) * 0.5;
            }
        }
        shiftDFT(angular, angular);

        for (int scale = 0; scale < nscale; scale++)
        {
            multiply(gabor[scale], angular, matAr[0]); //Product of the two components.
            
            filter_cpu[nscale * ori + scale]=matAr[0];

            merge(matAr, 2, filter[nscale * ori + scale]);

            filter_host[nscale * ori + scale] = (double *)malloc(sizeof(double)*dft_M*dft_N);
            filter_host[nscale * ori + scale] = (double *)filter_cpu[nscale * ori + scale].data;

            cudaMemcpy(filter_gpu[nscale * ori + scale], filter_host[nscale * ori + scale], dft_M* dft_N * sizeof(double), cudaMemcpyHostToHost);

        }//scale
    }//orientation
    //Filter ready
}


//和初始化对应
void PhaseCongruency::DeInit(){
    free(filter_host);
    free(filter_gpu);
    free(filtered_gpu);
    free(ifft_img_gpu);
    free(An);
    free(An_cpu);
    free(host_data);
    cudaFree(idata);
    cudaFree(odata);
    cudaFree(img_complex_gpu);
    cudaFree(sum_re);
    cudaFree(sum_im);
    cudaFree(sum_an);
    cudaFree(max_an);
    cudaFree(complex0);
    cudaFree(complex1);
    cudaFree(energy);
    cudaFree(convx);
    cudaFree(convy);
    cudaFree(convx2);
    cudaFree(convy2);
    cudaFree(convxy);
    cudaFree(Max_moment);
    free(pc_gpu);
    //指针置空
    filter_host=nullptr; filter_gpu=nullptr;filtered_gpu=nullptr; ifft_img_gpu=nullptr; An=nullptr;
    An_cpu=nullptr; host_data=nullptr; idata=nullptr;odata=nullptr;img_complex_gpu=nullptr;
    sum_re=nullptr; sum_im=nullptr;sum_an=nullptr;max_an=nullptr;complex0=nullptr;complex1=nullptr;
    energy=nullptr;convx=nullptr;convy=nullptr;convx2=nullptr;convy2=nullptr;convxy=nullptr;Max_moment=nullptr;pc_gpu=nullptr;   
}


//cuda 代码 多线程 经过测试 多线程不能够对gpu加速 因此不使用多线程
void PhaseCongruency::cudatest(cv::InputArray _src) {
    // struct timeval s1,e1;
    // gettimeofday(&s1, NULL);
    cv::Mat src = _src.getMat();
    CV_Assert(src.size() == size);
    const int width = size.width, height = size.height;
    cv::Mat src64;
    src.convertTo(src64, MAT_TYPE_CNV, 1.0 / 1.0);
    const int dft_M_r = cv::getOptimalDFTSize(src.rows) - src.rows;
    const int dft_N_c = cv::getOptimalDFTSize(src.cols) - src.cols;

    cv::Mat padded;
    copyMakeBorder(src64, padded, 0, dft_M_r, 0, dft_N_c, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    int NX= cv::getOptimalDFTSize(src.rows); //x是高度
    int NY= cv::getOptimalDFTSize(src.cols); //y是宽度

	host_data = (double *)padded.data;
    cudaMemcpy(idata, host_data, NX* NY * sizeof(double), cudaMemcpyHostToHost);

    cuda_padcomplex(idata,img_complex_gpu,NY,NX);
    cufftExecZ2Z(plan_forward,img_complex_gpu,(cufftDoubleComplex*)odata,CUFFT_FORWARD);

    //多流尝试 没用
    // cudaDeviceProp prop;
	// int deviceID;
	// cudaGetDevice(&deviceID);
	// cudaGetDeviceProperties(&prop, deviceID);
	// //检查设备是否支持重叠功能  
	// if (!prop.deviceOverlap)
	// {
	// 	printf("No device will handle overlaps. so no speed up from stream.\n");
	// }
 
    double noise=0;
    // 多个线程执行一个
    for (unsigned int o = 0; o < norient; o++)
    {      
        for (unsigned int scale = 0; scale < nscale; scale++)
        {
            cuda_mulSpectrums(odata,filter_gpu[nscale * o + scale],filtered_gpu[nscale * o + scale],NY,NX);
            cufftExecZ2Z(plan_inverse,(cufftDoubleComplex *)filtered_gpu[nscale * o + scale],(cufftDoubleComplex *)ifft_img_gpu[nscale * o + scale],CUFFT_INVERSE);
            cuda_calc_eo(ifft_img_gpu[nscale * o + scale],NY,NX);      
            cuda_calc_eomag(ifft_img_gpu[nscale * o + scale],An[nscale * o + scale],NY,NX);
        }
           
    }
    // cudaDeviceSynchronize();

    // for (unsigned int o = 0; o < norient; o++)
    // {      
    //     for (unsigned int scale = 0; scale < nscale; scale++)
    //     {           
    //         cufftExecZ2Z(plan_inverse,(cufftDoubleComplex *)filtered_gpu[nscale * o + scale],(cufftDoubleComplex *)ifft_img_gpu[nscale * o + scale],CUFFT_INVERSE);  
    

    //     }
           
    // }
    // cudaDeviceSynchronize();

    // for (unsigned int o = 0; o < norient; o++)
    // {      
    //     for (unsigned int scale = 0; scale < nscale; scale++)
    //     {     
    //         cuda_calc_eo(ifft_img_gpu[nscale * o + scale],NY,NX);      
           
    //     }
           
    // }
    // cudaDeviceSynchronize();
    // for (unsigned int o = 0; o < norient; o++)
    // {      
    //     for (unsigned int scale = 0; scale < nscale; scale++)
    //     {     
    //         cuda_calc_eomag(ifft_img_gpu[nscale * o + scale],An[nscale * o + scale],NY,NX);    
           
    //     }
           
    // }
    // cudaDeviceSynchronize(); 

    for (unsigned int o = 0; o < norient; o++)
    {      
        for (unsigned int scale = 0; scale < nscale; scale++)
        {
            
            cudaDeviceSynchronize();          
            //转为本地 此时 eo是eo_mag
            cv::Mat eo_padded = cv::Mat(NX,NY,CV_64F, (double*)An[nscale * o + scale]);
            eo[nscale * o + scale] = eo_padded(cv::Range(0, height),cv::Range(0, width));
            // eo[nscale * o + scale] = cv::Mat(NX,NY,CV_64F, (double*)An[nscale * o + scale]);
            if(scale==0){
                //计算noise 正确
                auto tau = mean(eo[nscale * o + scale]);
                tau.val[0] = tau.val[0] / sqrt(log(4.0));
                auto mt = 1.0 * pow(mult, nscale);
                auto totalTau = tau.val[0] * (1.0 - 1.0 / mt) / (1.0 - 1.0 / mult);
                auto m = totalTau * sqrt(M_PI / 2.0);
                auto n = totalTau * sqrt((4 - M_PI) / 2.0);
                noise = m + k * n;
                // printf("noise %f \n",noise);
                cuda_copy_to(An[nscale * o],max_an,NY,NX);
                cuda_copy_to(An[nscale * o],sum_an,NY,NX);
                cuda_splitZ(ifft_img_gpu[nscale * o],sum_re,sum_im,NY,NX);               
            }
            else{
                cuda_splitZ(ifft_img_gpu[nscale * o + scale],complex0,complex1,NY,NX);
                cuda_self_add(sum_an,An[nscale * o + scale],NY,NX);
                cuda_self_add(sum_re,complex0,NY,NX);
                cuda_self_add(sum_im,complex1,NY,NX);
                cuda_self_max(max_an,An[nscale * o + scale],NY,NX);             
            }
           
        }
        cuda_norm_orient(sum_re,sum_im,epsilon,NY,NX);

        cuda_zero(energy,NY,NX);
        for (int scale = 0; scale < nscale; scale++)
        {
            cuda_splitZ(ifft_img_gpu[nscale * o + scale],complex0,complex1,NY,NX);
            cuda_clac_energy(sum_re,sum_im,complex0,complex1,energy,NY,NX);          
        } 
        cuda_max_R(energy,noise,NY,NX);
        //计算pc 输入:sum_an max_an energy res g epslion nscale pc[o]
        cuda_calc_pc(sum_an,max_an,energy,pc_gpu[o],cutOff,g,epsilon,nscale,NY,NX);

        // cudaDeviceSynchronize();
        
        // double * realimg;
        // double * falimg;
        // cudaMalloc((double**)&realimg, sizeof(double)* NX* NY);
        // cudaMalloc((double**)&falimg, sizeof(double)* NX* NY);
        // cuda_splitZ((cufftDoubleComplex*)odata, (double*) realimg,( double*) falimg, NY,NX);
        // double * real;
        // double * fal;
        // real=(double *)malloc(sizeof(double)*NX*NY);
        // fal=(double *)malloc(sizeof(double)*NX*NY);
        // cudaMemcpy(real, realimg, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);
        // cudaMemcpy(fal, falimg, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);
        // for (size_t i =0; i < 10; i++)
        // {
        //     std::cout << "gpu test: " << i << "  " << real[i] <<" " <<fal[i]<< std::endl;
        // }
    }

    // add 之前可能需要置零
    cuda_zero(convx2,NY,NX);
    cuda_zero(convy2,NY,NX);
    cuda_zero(convxy,NY,NX);
    const double angle_const = M_PI / static_cast<double>(norient);
    for (unsigned int o = 0; o < norient; o++)
    {
        auto angl = static_cast<double>(o) * angle_const;
        cuda_calc_convx(pc_gpu[o],angl,convx,NY,NX);
        cuda_calc_convy(pc_gpu[o],angl,convy,NY,NX);
        cuda_add_convx2(convx,convx2,norient,NY,NX);
        cuda_add_convy2(convy,convy2,norient,NY,NX);
        cuda_add_convxy(convx,convy,convxy,norient,NY,NX);
    } 
    //计算M
    cuda_calc_M(convx2,convy2,convxy,epsilon,Max_moment,NY,NX);
    cudaDeviceSynchronize();
    //转为本地 
    maxMoment=cv::Mat(NX,NY,CV_64F,(double*)Max_moment);

    // gettimeofday(&e1, NULL);     
    // double timeUsed;
    // timeUsed = 1000000 * (e1.tv_sec - s1.tv_sec) + e1.tv_usec - s1.tv_usec;
    // std::cout << " test cuda function Time=" << timeUsed / 1000 << " ms" << std::endl;

    //// ==测试代码
    // double * realimg;
    // double * falimg;
    // cudaMalloc((double**)&realimg, sizeof(double)* NX* NY);
    // cudaMalloc((double**)&falimg, sizeof(double)* NX* NY);
    // cuda_splitZ((cufftDoubleComplex*)Max_moment, (double*) realimg,( double*) falimg, NY,NX);
    // double * real;
    // double * fal;
    // real=(double *)malloc(sizeof(double)*NX*NY);
    // fal=(double *)malloc(sizeof(double)*NX*NY);
    // cudaMemcpy(real, realimg, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(fal, falimg, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);
    // for (size_t i =0; i < 10; i++)
    // {
    //     std::cout << "gpu: " << i << "  " << real[i] <<" " <<fal[i]<< std::endl;
    // }

}
