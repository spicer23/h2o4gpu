/*!
 * Modifications Copyright 2017 H2O.ai, Inc.
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<iostream>
#include<iomanip>
#include<stdlib.h>
#include<stdio.h>
#include<assert.h>
#include<math.h>

#include <cusolverDn.h>
#include <cuda_runtime_api.h>

#include "../matrix/utilities.cuh"
#include "svd.h"

namespace svd {
	template<typename T>
	int svd_fit(cusolverDnHandle_t handle,
			signed char jobu,
			signed char jobvt,
			int m,
			int n,
			double *A,
			int lda,
			double *S,
			double *U,
			int ldu,
			double *VT,
			int ldvt,
			double *work,
			int lwork,
			double *rwork,
			int *info ){

	//    // --- gesvd only supports m >= n
	//    // --- column major memory ordering
	//
	//    const int m = 7;
	//    const int n = 5;
	//
	//    // --- cuSOLVE input/output parameters/arrays
	//    int lwork = 0;
	//    int *info;           gpuErrchk(cudaMalloc(&info,          sizeof(int)));

		// --- CUDA solver initialization
		cusolverDnCreate(&handle);

		// --- Setting the host, m x n matrix
		double *h_A = (double *)malloc(m * n * sizeof(double));
		for(int j = 0; j < m; j++)
			for(int i = 0; i < n; i++)
				h_A[j + i*m] = (i + j*j) * sqrt((double)(i + j));

		// --- Setting the device matrix and moving the host matrix to the device
		double *d_A;            gpuErrchk(cudaMalloc(&d_A,      m * n * sizeof(double)));
		gpuErrchk(cudaMemcpy(d_A, h_A, m * n * sizeof(double), cudaMemcpyHostToDevice));

		// --- host side SVD results space
		double *h_U = (double *)malloc(m * m     * sizeof(double));
		double *h_V = (double *)malloc(n * n     * sizeof(double));
		double *h_S = (double *)malloc(min(m, n) * sizeof(double));

		// --- device side SVD workspace and matrices
		double *d_U;            gpuErrchk(cudaMalloc(&d_U,  m * m     * sizeof(double)));
		double *d_V;            gpuErrchk(cudaMalloc(&d_V,  n * n     * sizeof(double)));
		double *d_S;            gpuErrchk(cudaMalloc(&d_S,  min(m, n) * sizeof(double)));

		// --- CUDA SVD initialization
		cusolveSafeCall(cusolverDnDgesvd_bufferSize(handle, m, n, &lwork));
		gpuErrchk(cudaMalloc(&work, lwork * sizeof(double)));

		// --- CUDA SVD execution
		cusolveSafeCall(cusolverDnDgesvd(handle, 'A', 'A', m, n, d_A, m, d_S, d_U, m, d_V, n, work, lwork, NULL, info));
		int info_h = 0;  gpuErrchk(cudaMemcpy(&info_h, info, sizeof(int), cudaMemcpyDeviceToHost));
		if (info_h != 0) std::cout   << "Unsuccessful SVD execution\n\n";

		// --- Moving the results from device to host
		gpuErrchk(cudaMemcpy(h_S, d_S, min(m, n) * sizeof(double), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(h_U, d_U, m * m     * sizeof(double), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(h_V, d_V, n * n     * sizeof(double), cudaMemcpyDeviceToHost));

		std::cout << "Singular values\n";
		for(int i = 0; i < min(m, n); i++)
			std::cout << "d_S["<<i<<"] = " << std::setprecision(15) << h_S[i] << std::endl;

		std::cout << "\nLeft singular vectors - For y = A * x, the columns of U span the space of y\n";
		for(int j = 0; j < m; j++) {
			printf("\n");
			for(int i = 0; i < m; i++)
				printf("U[%i,%i]=%f\n",i,j,h_U[j*m + i]);
		}

		std::cout << "\nRight singular vectors - For y = A * x, the columns of V span the space of x\n";
		for(int i = 0; i < n; i++) {
			printf("\n");
			for(int j = 0; j < n; j++)
				printf("V[%i,%i]=%f\n",i,j,h_V[j*n + i]);
		}

		cusolverDnDestroy(handle);

		return 0;
	}

	template<typename T> int makePtr_dense(
			cusolverDnHandle_t handle,
						signed char jobu,
						signed char jobvt,
						int m,
						int n,
						double *A,
						int lda,
						double *S,
						double *U,
						int ldu,
						double *VT,
						int ldvt,
						double *work,
						int lwork,
						double *rwork,
						int *info ) {
		return svd_fit(handle,
			jobu,
			jobvt,
			m,
			n,
			*A,
			lda,
			*S,
			*U,
			ldu,
			*VT,
			ldvt,
			*work,
			lwork,
			*rwork,
			*info );
	}

	template int makePtr_dense<float>(
			cusolverDnHandle_t handle,
						signed char jobu,
						signed char jobvt,
						int m,
						int n,
						double *A,
						int lda,
						double *S,
						double *U,
						int ldu,
						double *VT,
						int ldvt,
						double *work,
						int lwork,
						double *rwork,
						int *info );

	template int makePtr_dense<double>(
			cusolverDnHandle_t handle,
						signed char jobu,
						signed char jobvt,
						int m,
						int n,
						double *A,
						int lda,
						double *S,
						double *U,
						int ldu,
						double *VT,
						int ldvt,
						double *work,
						int lwork,
						double *rwork,
						int *info );

	template int svd_fit<float>(
			cusolverDnHandle_t handle,
						signed char jobu,
						signed char jobvt,
						int m,
						int n,
						double *A,
						int lda,
						double *S,
						double *U,
						int ldu,
						double *VT,
						int ldvt,
						double *work,
						int lwork,
						double *rwork,
						int *info );

	template int
	svd_fit<double>(
			cusolverDnHandle_t handle,
						signed char jobu,
						signed char jobvt,
						int m,
						int n,
						double *A,
						int lda,
						double *S,
						double *U,
						int ldu,
						double *VT,
						int ldvt,
						double *work,
						int lwork,
						double *rwork,
						int *info );
}

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Interface for other languages
 */

int make_ptr_float_svd(
		cusolverDnHandle_t handle,
					signed char jobu,
					signed char jobvt,
					int m,
					int n,
					double *A,
					int lda,
					double *S,
					double *U,
					int ldu,
					double *VT,
					int ldvt,
					double *work,
					int lwork,
					double *rwork,
					int *info ) {
  return svd::makePtr_dense<float>(
		  	handle,
			jobu,
			jobvt,
			m,
			n,
			*A,
			lda,
			*S,
			*U,
			ldu,
			*VT,
			ldvt,
			*work,
			lwork,
			*rwork,
			*info );
}

int make_ptr_double_svd(
		cusolverDnHandle_t handle,
					signed char jobu,
					signed char jobvt,
					int m,
					int n,
					double *A,
					int lda,
					double *S,
					double *U,
					int ldu,
					double *VT,
					int ldvt,
					double *work,
					int lwork,
					double *rwork,
					int *info ) {
  return svd::makePtr_dense<double>(
		    handle,
			jobu,
			jobvt,
			m,
			n,
			*A,
			lda,
			*S,
			*U,
			ldu,
			*VT,
			ldvt,
			*work,
			lwork,
			*rwork,
			*info );
}

#ifdef __cplusplus
}
#endif


