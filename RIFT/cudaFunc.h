#ifndef __CUDA_FUNC_H__
#define __CUDA_FUNC_H__

#include <jetson-utils/cudaUtility.h>

cudaError_t cuda_splitZ(cufftDoubleComplex* A, double* B, double* res, size_t iWidth, size_t iHeight);
cudaError_t cuda_mulSpectrums(cufftDoubleComplex* A, double* B, cufftDoubleComplex* res, int iWidth, int iHeight);
cudaError_t cuda_padcomplex(double* A, cufftDoubleComplex* res, int iWidth, int iHeight);
cudaError_t cuda_calc_eo(cufftDoubleComplex* A, int iWidth, int iHeight);
cudaError_t cuda_calc_eomag(cufftDoubleComplex* A, double* res, int iWidth, int iHeight);
cudaError_t cuda_self_add(double* res, double* A, int iWidth, int iHeight);
cudaError_t cuda_self_max(double* res, double* A, int iWidth, int iHeight);
cudaError_t cuda_copy_to(double* A, double* res, int iWidth, int iHeight);
cudaError_t cuda_norm_orient(double* A, double* B, double epsilon, int iWidth, int iHeight);
cudaError_t cuda_clac_energy(double* A, double* B, double* C, double * D, double* res,  int iWidth, int iHeight);
cudaError_t cuda_zero(double* res,  int iWidth, int iHeight);
cudaError_t cuda_max_R(double* res, double noise, int iWidth, int iHeight);
cudaError_t cuda_calc_pc(double* sum_an,double* max_an, double* energy,double* res,double cutoff,double g,double epsilon,double nscale, int iWidth, int iHeight);
cudaError_t cuda_calc_convx(double* pc, double angle, double* res, int iWidth, int iHeight);
cudaError_t cuda_calc_convy(double* pc, double angle, double* res, int iWidth, int iHeight);
cudaError_t cuda_add_convx2(double* convx, double* res, double norient,int iWidth, int iHeight);
cudaError_t cuda_add_convy2(double* convx, double* res,double norient, int iWidth, int iHeight);
cudaError_t cuda_add_convxy(double* convx, double* convy, double* res, double norient,int iWidth, int iHeight);
cudaError_t cuda_calc_M(double* convx2, double* convy2,double* convxy, double epsilon, double* res, int iWidth, int iHeight);
#endif
