#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/cudaFilterMode.cuh>
#include <jetson-utils/cudaVector.h>
#include <cufft.h>
#include <cuComplex.h>

//复数分离函数 感觉没啥问题呀
__global__ void splitZ_kernel(cufftDoubleComplex* A, double* B, double* res, int iWidth, int iHeight)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if( m >= iWidth || n >= iHeight )
		return;

	int idx = n * iWidth + m;

	B[idx]=cuCreal(A[idx]);
	res[idx]=cuCimag(A[idx]);
}

//变换滤波实数乘上复数//
__global__ void mulSpectrums_kernel(cufftDoubleComplex* A, double* B, cufftDoubleComplex* res, int iWidth, int iHeight)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if( m >= iWidth || n >= iHeight )
		return;

	int idx = n * iWidth + m;

	res[idx].x=cuCreal(A[idx])*B[idx];
	res[idx].y=cuCimag(A[idx])*B[idx];
}


//输入源图像转换成复数形式 使用C2C进行fft变换
__global__ void com_padding_kernel(double* A, cufftDoubleComplex* res, int iWidth, int iHeight)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if( m >= iWidth || n >= iHeight )
		return;

	int idx = n * iWidth + m;

	res[idx].x=A[idx];
	res[idx].y=(double)0.0;
}

// eo除以点数
__global__ void calc_eo_kernel(cufftDoubleComplex* A,int iWidth, int iHeight)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if( m >= iWidth || n >= iHeight )
		return;

	int idx = n * iWidth + m;
	A[idx].x=A[idx].x /(iWidth*iHeight);
	A[idx].y=A[idx].y/(iWidth*iHeight);
}




//根据eo计算eo_mag(同时除以傅里叶点数)
__global__ void calc_eomag_kernel(cufftDoubleComplex* A, double* res, int iWidth, int iHeight)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if( m >= iWidth || n >= iHeight )
		return;

	int idx = n * iWidth + m;

	res[idx]=cuCabs(A[idx]);
}

//self_add(sumAn, eo_mag, sumAn); sumAn = eo_mag + sumAn 这样能够省去声明一个
__global__ void self_add_kernel(double* res, double* A, int iWidth, int iHeight)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if( m >= iWidth || n >= iHeight )
		return;

	int idx = n * iWidth + m;

	res[idx]+=A[idx];
}

//self_max
__global__ void self_max_kernel(double* res, double* A, int iWidth, int iHeight)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if( m >= iWidth || n >= iHeight )
		return;

	int idx = n * iWidth + m;

	if(res[idx]<A[idx]){
		res[idx]=A[idx];
	}
}



//copy to 使用A初始化res
__global__ void copy_to_kernel(double* A, double* res, int iWidth, int iHeight)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if( m >= iWidth || n >= iHeight )
		return;

	int idx = n * iWidth + m;

	res[idx]=A[idx];
}


__global__ void norm_orient_kernel(double* A, double* B, double epsilon, int iWidth, int iHeight)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if( m >= iWidth || n >= iHeight )
		return;

	int idx = n * iWidth + m;
	double temp=sqrt(A[idx]*A[idx]+B[idx]*B[idx])+epsilon;
	A[idx]=A[idx]/temp;
	B[idx]=B[idx]/temp;
}


// E+= D*B +C*A-abs(AD-BC)
__global__ void calc_energy_kernel(double* A, double* B, double* C, double * D, double* res,  int iWidth, int iHeight)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if( m >= iWidth || n >= iHeight )
		return;

	int idx = n * iWidth + m;

	res[idx]=res[idx]+D[idx]*B[idx] +C[idx]*A[idx]-abs(A[idx]*D[idx]-B[idx]*C[idx]);
}

__global__ void zero_kernel(double* res, int iWidth, int iHeight)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if( m >= iWidth || n >= iHeight )
		return;

	int idx = n * iWidth + m;

	res[idx]=0.0;
}

__global__ void max_R_kernel(double* res, double noise, int iWidth, int iHeight)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if( m >= iWidth || n >= iHeight )
		return;

	int idx = n * iWidth + m;

	res[idx]=max(0.0,res[idx]-noise);
}

__global__ void calc_pc_kernel(double* sum_an,double* max_an, double* energy,double* res,double cutoff, double g,double epsilon,double nscale, int iWidth, int iHeight)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if( m >= iWidth || n >= iHeight )
		return;

	//输入的是double类型 会不会存在问题呢 会不会等待同步取用数据？ 设置成const 试试？ 有待确定
	int idx = n * iWidth + m;
	double width=(sum_an[idx]/(max_an[idx]+epsilon)-1)/(nscale-1);
	double temp=1+exp(g*(cutoff-width));
	res[idx]=energy[idx]/(sum_an[idx]*temp);
}

__global__ void calc_convx_kernel(double* pc, double angle, double* res, int iWidth, int iHeight)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if( m >= iWidth || n >= iHeight )
		return;

	//输入的是double类型 会不会存在问题呢 会不会等待同步取用数据？ 设置成const 试试？ 有待确定
	int idx = n * iWidth + m;

	res[idx]=pc[idx]*cos(angle);
}



__global__ void calc_convy_kernel(double* pc, double angle, double* res, int iWidth, int iHeight)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if( m >= iWidth || n >= iHeight )
		return;

	//输入的是double类型 会不会存在问题呢 会不会等待同步取用数据？ 设置成const 试试？ 有待确定
	int idx = n * iWidth + m;

	res[idx]=pc[idx]*sin(angle);
}

__global__ void add_convx2_kernel(double* convx, double* res, double norient,int iWidth, int iHeight)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if( m >= iWidth || n >= iHeight )
		return;

	//输入的是double类型 会不会存在问题呢 会不会等待同步取用数据？ 设置成const 试试？ 有待确定
	int idx = n * iWidth + m;

	res[idx]+=convx[idx]*convx[idx]*2/norient;
}


__global__ void add_convy2_kernel(double* convy, double* res, double norient,int iWidth, int iHeight)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if( m >= iWidth || n >= iHeight )
		return;

	//输入的是double类型 会不会存在问题呢 会不会等待同步取用数据？ 设置成const 试试？ 有待确定
	int idx = n * iWidth + m;

	res[idx]+=convy[idx]*convy[idx]*2/norient;
}

__global__ void add_convxy_kernel(double* convx,double* convy, double* res,double norient, int iWidth, int iHeight)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if( m >= iWidth || n >= iHeight )
		return;

	//输入的是double类型 会不会存在问题呢 会不会等待同步取用数据？ 设置成const 试试？ 有待确定
	int idx = n * iWidth + m;

	res[idx]+=convx[idx]*convy[idx]*4/norient;
}

__global__ void calc_M_kernel(double* convx2, double* convy2,double* convxy, double epsilon, double* res, int iWidth, int iHeight)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if( m >= iWidth || n >= iHeight )
		return;

	//输入的是double类型 会不会存在问题呢 会不会等待同步取用数据？ 设置成const 试试？ 有待确定
	int idx = n * iWidth + m;

	res[idx]=(convx2[idx]+convy2[idx]+sqrt(convxy[idx]*convxy[idx]+(convx2[idx]-convy2[idx])*(convx2[idx]-convy2[idx]))+epsilon)/2;
}


cudaError_t cuda_splitZ(cufftDoubleComplex* A, double* B, double* res, size_t iWidth, size_t iHeight)
{
	if( !A || !B || !res )
		return cudaErrorInvalidDevicePointer;

	if( iWidth==0 || iHeight==0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(8,8);
	const dim3 gridDim(iDivUp(iWidth,blockDim.x), iDivUp(iHeight,blockDim.y));
	splitZ_kernel<<<gridDim,blockDim>>>(A, B, res, iWidth, iHeight);
	return CUDA(cudaGetLastError());
}

cudaError_t cuda_mulSpectrums(cufftDoubleComplex* A, double* B, cufftDoubleComplex* res, int iWidth, int iHeight)
{
	if( !A || !B || !res )
		return cudaErrorInvalidDevicePointer;

	if( iWidth==0 || iHeight==0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(8,8);
	const dim3 gridDim(iDivUp(iWidth,blockDim.x), iDivUp(iHeight,blockDim.y));
	mulSpectrums_kernel<<<gridDim,blockDim>>>(A, B, res, iWidth, iHeight);
	return CUDA(cudaGetLastError());
}

cudaError_t cuda_padcomplex(double* A, cufftDoubleComplex* res, int iWidth, int iHeight)
{
	if( !A || !res )
		return cudaErrorInvalidDevicePointer;

	if( iWidth==0 || iHeight==0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(8,8);
	const dim3 gridDim(iDivUp(iWidth,blockDim.x), iDivUp(iHeight,blockDim.y));
	com_padding_kernel<<<gridDim,blockDim>>>(A,res, iWidth, iHeight);
	return CUDA(cudaGetLastError());
}

//不能够这样写 报错 参数过少
cudaError_t cuda_calc_eo(cufftDoubleComplex* A, int iWidth, int iHeight)
{
	if( !A )
		return cudaErrorInvalidDevicePointer;

	if( iWidth==0 || iHeight==0 )
		return cudaErrorInvalidValue;

	// launch kernels
	const dim3 blockDim(8,8);
	const dim3 gridDim(iDivUp(iWidth,blockDim.x), iDivUp(iHeight,blockDim.y));
	calc_eo_kernel<<<gridDim,blockDim>>>(A, iWidth, iHeight);
	return CUDA(cudaGetLastError());
}


cudaError_t cuda_calc_eomag(cufftDoubleComplex* A, double* res, int iWidth, int iHeight)
{
	if( !A || !res )
		return cudaErrorInvalidDevicePointer;

	if( iWidth==0 || iHeight==0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(8,8);
	const dim3 gridDim(iDivUp(iWidth,blockDim.x), iDivUp(iHeight,blockDim.y));
	calc_eomag_kernel<<<gridDim,blockDim>>>(A,res, iWidth, iHeight);
	return CUDA(cudaGetLastError());
}

cudaError_t cuda_self_add(double* res, double* A, int iWidth, int iHeight)
{
	if( !A || !res )
		return cudaErrorInvalidDevicePointer;

	if( iWidth==0 || iHeight==0 )
		return cudaErrorInvalidValue;
	// launch kernel
	const dim3 blockDim(8,8);
	const dim3 gridDim(iDivUp(iWidth,blockDim.x), iDivUp(iHeight,blockDim.y));
	self_add_kernel<<<gridDim,blockDim>>>(res,A, iWidth, iHeight);
	return CUDA(cudaGetLastError());
}


cudaError_t cuda_self_max(double* res, double* A, int iWidth, int iHeight)
{
	if( !A || !res )
		return cudaErrorInvalidDevicePointer;

	if( iWidth==0 || iHeight==0 )
		return cudaErrorInvalidValue;
	// launch kernel
	const dim3 blockDim(8,8);
	const dim3 gridDim(iDivUp(iWidth,blockDim.x), iDivUp(iHeight,blockDim.y));
	self_max_kernel<<<gridDim,blockDim>>>(res,A, iWidth, iHeight);
	return CUDA(cudaGetLastError());
}


cudaError_t cuda_copy_to(double* A, double* res, int iWidth, int iHeight)
{
	if( !A || !res )
		return cudaErrorInvalidDevicePointer;

	if( iWidth==0 || iHeight==0 )
		return cudaErrorInvalidValue;
	// launch kernel
	const dim3 blockDim(8,8);
	const dim3 gridDim(iDivUp(iWidth,blockDim.x), iDivUp(iHeight,blockDim.y));
	copy_to_kernel<<<gridDim,blockDim>>>(A,res, iWidth, iHeight);
	return CUDA(cudaGetLastError());
}

//计算xenerngy时候
cudaError_t cuda_norm_orient(double* A, double* B, double epsilon, int iWidth, int iHeight)
{
	if( !A || !B )
		return cudaErrorInvalidDevicePointer;

	if( iWidth==0 || iHeight==0 )
		return cudaErrorInvalidValue;
	// launch kernel
	const dim3 blockDim(8,8);
	const dim3 gridDim(iDivUp(iWidth,blockDim.x), iDivUp(iHeight,blockDim.y));
	norm_orient_kernel<<<gridDim,blockDim>>>(A, B, epsilon, iWidth, iHeight);
	return CUDA(cudaGetLastError());
}

//试一试初始化的内存是不是0
cudaError_t cuda_clac_energy(double* A, double* B, double* C, double * D, double* res,  int iWidth, int iHeight)
{
	if( !A || !B || !C || !D || !res )
		return cudaErrorInvalidDevicePointer;

	if( iWidth==0 || iHeight==0 )
		return cudaErrorInvalidValue;
	// launch kernel
	const dim3 blockDim(8,8);
	const dim3 gridDim(iDivUp(iWidth,blockDim.x), iDivUp(iHeight,blockDim.y));
	calc_energy_kernel<<<gridDim,blockDim>>>(A, B, C,D,res, iWidth, iHeight);
	return CUDA(cudaGetLastError());
}

cudaError_t cuda_zero(double* res,  int iWidth, int iHeight)
{
	if( !res )
		return cudaErrorInvalidDevicePointer;

	if( iWidth==0 || iHeight==0 )
		return cudaErrorInvalidValue;
	// launch kernel
	const dim3 blockDim(8,8);
	const dim3 gridDim(iDivUp(iWidth,blockDim.x), iDivUp(iHeight,blockDim.y));
	zero_kernel<<<gridDim,blockDim>>>(res,iWidth, iHeight);
	return CUDA(cudaGetLastError());
}

cudaError_t cuda_max_R(double* res, double noise, int iWidth, int iHeight)
{
	if( !res )
		return cudaErrorInvalidDevicePointer;

	if( iWidth==0 || iHeight==0 )
		return cudaErrorInvalidValue;
	// launch kernel
	const dim3 blockDim(8,8);
	const dim3 gridDim(iDivUp(iWidth,blockDim.x), iDivUp(iHeight,blockDim.y));
	max_R_kernel<<<gridDim,blockDim>>>(res, noise, iWidth, iHeight);
	return CUDA(cudaGetLastError());
}

// sum_an max_an energy g esplion nscale pc[o] 计算pc 输入
cudaError_t cuda_calc_pc(double* sum_an,double* max_an, double* energy,double* res,double cutoff, double g,double epsilon,double nscale, int iWidth, int iHeight)
{
	if( !res )
		return cudaErrorInvalidDevicePointer;

	if( iWidth==0 || iHeight==0 )
		return cudaErrorInvalidValue;
	// launch kernel
	const dim3 blockDim(8,8);
	const dim3 gridDim(iDivUp(iWidth,blockDim.x), iDivUp(iHeight,blockDim.y));
	calc_pc_kernel<<<gridDim,blockDim>>>(sum_an,max_an,energy,res,cutoff,g,epsilon,nscale, iWidth, iHeight);
	return CUDA(cudaGetLastError());
}

cudaError_t cuda_calc_convx(double* pc, double angle, double* res, int iWidth, int iHeight)
{
	if( !res )
		return cudaErrorInvalidDevicePointer;

	if( iWidth==0 || iHeight==0 )
		return cudaErrorInvalidValue;
	// launch kernel
	const dim3 blockDim(8,8);
	const dim3 gridDim(iDivUp(iWidth,blockDim.x), iDivUp(iHeight,blockDim.y));
	calc_convx_kernel<<<gridDim,blockDim>>>(pc,angle,res,iWidth, iHeight);
	return CUDA(cudaGetLastError());
}

cudaError_t cuda_calc_convy(double* pc, double angle, double* res, int iWidth, int iHeight)
{
	if( !res )
		return cudaErrorInvalidDevicePointer;

	if( iWidth==0 || iHeight==0 )
		return cudaErrorInvalidValue;
	// launch kernel
	const dim3 blockDim(8,8);
	const dim3 gridDim(iDivUp(iWidth,blockDim.x), iDivUp(iHeight,blockDim.y));
	calc_convy_kernel<<<gridDim,blockDim>>>(pc,angle,res,iWidth, iHeight);
	return CUDA(cudaGetLastError());
}



cudaError_t cuda_add_convx2(double* convx, double* res,double norient, int iWidth, int iHeight)
{
	if( !res )
		return cudaErrorInvalidDevicePointer;

	if( iWidth==0 || iHeight==0 )
		return cudaErrorInvalidValue;
	// launch kernel
	const dim3 blockDim(8,8);
	const dim3 gridDim(iDivUp(iWidth,blockDim.x), iDivUp(iHeight,blockDim.y));
	add_convx2_kernel<<<gridDim,blockDim>>>(convx,res,norient,iWidth, iHeight);
	return CUDA(cudaGetLastError());
}

cudaError_t cuda_add_convy2(double* convy, double* res, double norient,int iWidth, int iHeight)
{
	if( !res )
		return cudaErrorInvalidDevicePointer;

	if( iWidth==0 || iHeight==0 )
		return cudaErrorInvalidValue;
	// launch kernel
	const dim3 blockDim(8,8);
	const dim3 gridDim(iDivUp(iWidth,blockDim.x), iDivUp(iHeight,blockDim.y));
	add_convy2_kernel<<<gridDim,blockDim>>>(convy,res,norient,iWidth, iHeight);
	return CUDA(cudaGetLastError());
}

cudaError_t cuda_add_convxy(double* convx, double* convy, double* res,double norient, int iWidth, int iHeight)
{
	if( !res )
		return cudaErrorInvalidDevicePointer;

	if( iWidth==0 || iHeight==0 )
		return cudaErrorInvalidValue;
	// launch kernel
	const dim3 blockDim(8,8);
	const dim3 gridDim(iDivUp(iWidth,blockDim.x), iDivUp(iHeight,blockDim.y));
	add_convxy_kernel<<<gridDim,blockDim>>>(convx,convy,res,norient,iWidth, iHeight);
	return CUDA(cudaGetLastError());
}



cudaError_t cuda_calc_M(double* convx2, double* convy2,double* convxy, double epsilon, double* res, int iWidth, int iHeight)
{
	if( !res )
		return cudaErrorInvalidDevicePointer;

	if( iWidth==0 || iHeight==0 )
		return cudaErrorInvalidValue;
	// launch kernel
	const dim3 blockDim(8,8);
	const dim3 gridDim(iDivUp(iWidth,blockDim.x), iDivUp(iHeight,blockDim.y));
	calc_M_kernel<<<gridDim,blockDim>>>(convx2,convy2,convxy,epsilon,res,iWidth, iHeight);
	return CUDA(cudaGetLastError());
}
