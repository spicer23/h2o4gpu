/*!
 * Copyright 2017 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif
double solver_controller_double(const char solver, const char family, int dopredict, int sourceDev,
		int datatype, int sharedA, int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord,
		size_t mTrain, size_t n, size_t mValid, int intercept, int standardize,
		double lambda_max, double lambda_min_ratio, int nLambdas, int nFolds,
		int nAlphas, double alpha_min, double alpha_max, double *alphas,
		double *lambdas, double tol, double tolseekfactor, int lambdastopearly, int glmstopearly,
		double glmstopearlyerrorfraction, int max_iterations, int verbose,
		void *trainXptr, void *trainYptr, void *validXptr, void *validYptr,
		void *weightptr, int givefullpath, double **Xvsalphalambda,
		double **Xvsalpha, double **validPredsvsalphalambda,
		double **validPredsvsalpha, size_t *countfull, size_t *countshort,
		size_t *countmore);
double solver_controller_float(const char solver, const char family, int dopredict, int sourceDev,
		int datatype, int sharedA, int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord,
		size_t mTrain, size_t n, size_t mValid, int intercept, int standardize,
		double lambda_max, double lambda_min_ratio, int nLambdas, int nFolds,
		int nAlphas, double alpha_min, double alpha_max, float *alphas,
		float *lambdas, double tol, double tolseekfactor, int lambdastopearly, int glmstopearly,
		double glmstopearlyerrorfraction, int max_iterations, int verbose,
		void *trainXptr, void *trainYptr, void *validXptr, void *validYptr,
		void *weightptr, int givefullpath, float **Xvsalphalambda,
		float **Xvsalpha, float **validPredsvsalphalambda,
		float **validPredsvsalpha, size_t *countfull, size_t *countshort,
		size_t *countmore);

#ifdef __cplusplus
}
#endif
