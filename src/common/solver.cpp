/*!
 * Copyright 2017 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#include "solver.h"
#include "elastic_net_ptr.h"

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
		size_t *countmore) {
  if (solver == 'e') {//Elasticnet
		if (dopredict == 0) {
			return h2o4gpu::ElasticNetptr_fit(family, sourceDev, datatype, sharedA, nThreads, gpu_id, nGPUs, totalnGPUs,
					ord, mTrain, n, mValid, intercept, standardize,
					lambda_max, lambda_min_ratio, nLambdas, nFolds,
					nAlphas, alpha_min, alpha_max,
					alphas, lambdas,
					tol, tolseekfactor,
					lambdastopearly, glmstopearly, glmstopearlyerrorfraction, max_iterations, verbose, trainXptr,
					trainYptr, validXptr, validYptr, weightptr, givefullpath,
					Xvsalphalambda, Xvsalpha, validPredsvsalphalambda,
					validPredsvsalpha, countfull, countshort, countmore);
		} else {
			return h2o4gpu::ElasticNetptr_predict(family, sourceDev, datatype, sharedA, nThreads, gpu_id, nGPUs, totalnGPUs,
					ord, mTrain, n, mValid, intercept, standardize,
					lambda_max, lambda_min_ratio, nLambdas, nFolds,
					nAlphas, alpha_min, alpha_max,
					alphas, lambdas,
					tol, tolseekfactor,
					lambdastopearly, glmstopearly, glmstopearlyerrorfraction, max_iterations, verbose, trainXptr,
					trainYptr, validXptr, validYptr, weightptr, givefullpath,
					Xvsalphalambda, Xvsalpha, validPredsvsalphalambda,
					validPredsvsalpha, countfull, countshort, countmore);
		}
  } else{//Something else (SVD, etc)
	  printf("Wrong family");
  }
}

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
		size_t *countmore) {
  if (solver == 'e') {//Elasticnet
		if (dopredict == 0) {
			return h2o4gpu::ElasticNetptr_fit(family, sourceDev, datatype, sharedA, nThreads, gpu_id, nGPUs, totalnGPUs,
					ord, mTrain, n, mValid, intercept, standardize,
					lambda_max, lambda_min_ratio, nLambdas, nFolds,
					nAlphas, alpha_min, alpha_max,
					alphas, lambdas,
					tol, tolseekfactor,
					lambdastopearly, glmstopearly, glmstopearlyerrorfraction, max_iterations, verbose, trainXptr,
					trainYptr, validXptr, validYptr, weightptr, givefullpath,
					Xvsalphalambda, Xvsalpha, validPredsvsalphalambda,
					validPredsvsalpha, countfull, countshort, countmore);
		} else {
			return h2o4gpu::ElasticNetptr_predict(family, sourceDev, datatype, sharedA, nThreads, gpu_id, nGPUs, totalnGPUs,
					ord, mTrain, n, mValid, intercept, standardize,
					lambda_max, lambda_min_ratio, nLambdas, nFolds,
					nAlphas, alpha_min, alpha_max,
					alphas, lambdas,
					tol, tolseekfactor,
					lambdastopearly, glmstopearly, glmstopearlyerrorfraction, max_iterations, verbose, trainXptr,
					trainYptr, validXptr, validYptr, weightptr, givefullpath,
					Xvsalphalambda, Xvsalpha, validPredsvsalphalambda,
					validPredsvsalpha, countfull, countshort, countmore);
		}
  } else{//Something else (SVD, etc)
	  printf("Wrong family");
  }
}
