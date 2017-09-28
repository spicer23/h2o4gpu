/*!
 * Copyright 2017 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once
#include <stddef.h>
#include <stdio.h>
#include <limits>
#include <vector>
#include <cassert>
#include <iostream>
#include <random>

#include "matrix/matrix_dense.h"
#include "h2o4gpuglm.h"
#include "timer.h"
#include <omp.h>
#include <cmath>

namespace h2o4gpu {
// SVD

double SVDPtr(int sourceDev, int datatype, int sharedA,
			int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord, size_t mTrain, size_t n,
			int standardize, int verbose, void *matrixPtr);

double SVDPtr_fit(int sourceDev, int datatype, int sharedA,
			int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord, size_t mTrain, size_t n,
			int standardize, int verbose, void *matrixPtr);

int makePtr_dense(int sourceDev, int datatype, int sharedA,
			int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord, size_t mTrain, size_t n,
			int standardize, int verbose, void *matrixPtr);

#ifdef __cplusplus
extern "C" {
#endif
double svd_ptr_double(int sourceDev, int datatype, int sharedA,
		int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord, size_t mTrain, size_t n,
		int standardize, int verbose, void *matrixPtr);

double svd_ptr_float(int sourceDev, int datatype, int sharedA,
		int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord, size_t mTrain, size_t n,
		int standardize, int verbose, void *matrixPtr);

#ifdef __cplusplus
}
#endif
}
