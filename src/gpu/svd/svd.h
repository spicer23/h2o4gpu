/*
 * svd.h
 *
 *  Created on: Sep 26, 2017
 *      Author: navdeep
 */

#ifndef SVD_H_
#define SVD_H_

template<typename T>
T svd(cusolverDnHandle_t handle,
		signed char jobu,
		signed char jobvt,
		int m, int n,
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

#endif /* SVD_H_ */

