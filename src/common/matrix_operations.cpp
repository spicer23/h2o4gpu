/*!
 * Copyright 2017 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#include "matrix_operations.h"
#include <float.h>
#include "../include/util.h"
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h> //  our new library

using namespace std;

namespace h2o4gpu {

// SVD

template<typename T>
double SVDPtr(
		int sourceDev, int datatype, int sharedA,
		int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord, size_t mTrain, size_t n,
		int standardize, int verbose, void *matrixPtr) {

	return SVDPtr_fit(sourceDev, datatype, sharedA,
				nThreads, gpu_id, nGPUs, totalnGPUs, ord, mTrain, n,
				standardize, verbose, matrixPtr);

}

template<typename T>
double SVDPtr_fit(int sourceDev, int datatype, int sharedA,
		int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord, size_t mTrain, size_t n,
		int standardize, int verbose, void *matrixPtr) {

		// number of openmp threads = number of cuda devices to use
		#ifdef _OPENMP
			int omt=omp_get_max_threads();
			//      omp_set_num_threads(MIN(omt,nGPUs));  // not necessary, but most useful mode so far
			omp_set_num_threads(nThreads);// not necessary, but most useful mode so far
			int nth=omp_get_max_threads();
			//      nGPUs=nth; // openmp threads = cuda/cpu devices used
			omp_set_dynamic(0);
		#if(USEMKL==1)
			mkl_set_dynamic(0);
		#endif
			omp_set_nested(1);
			omp_set_max_active_levels(2);
		#ifdef DEBUG
			cout << "Number of original threads=" << omt << " Number of final threads=" << nth << endl;
		#endif
		#endif

		int sourceme = sourceDev;
		h2o4gpu::MatrixDense<T> Asource_(sharedA, sourceme, datatype, ord, mTrain, n, reinterpret_cast<T *>(matrixPtr));

	#define MAX(a,b) ((a)>(b) ? (a) : (b))
		double t = timer<double>();
		double t1me0;
		////////////////////////////////
		// PARALLEL REGION
	#pragma omp parallel proc_bind(master)
			{
		#ifdef _OPENMP
				int me = omp_get_thread_num();
				//https://software.intel.com/en-us/node/522115
				int physicalcores=omt;///2; // asssume hyperthreading Intel processor (doens't improve much to ensure physical cores used0
				// set number of mkl threads per openmp thread so that not oversubscribing cores
				int mklperthread=MAX(1,(physicalcores % nThreads==0 ? physicalcores/nThreads : physicalcores/nThreads+1));
		#if(USEMKL==1)
				//mkl_set_num_threads_local(mklperthread);
				mkl_set_num_threads_local(mklperthread);
				//But see (hyperthreading threads not good for MKL): https://software.intel.com/en-us/forums/intel-math-kernel-library/topic/288645
		#endif
		#else
				int me = 0;
		#endif

				int blasnumber;
		#ifdef HAVECUDA
				blasnumber=CUDA_MAJOR;
		#else
				blasnumber = mklperthread; // roughly accurate for openblas as well
		#endif

				// choose GPU device ID for each thread
				int wDev = gpu_id + (nGPUs > 0 ? me % nGPUs : 0);
				wDev = wDev % totalnGPUs;

				FILE *fil = NULL;

				////////////
				//
				// create class objects that creates cuda memory, cpu memory, etc.
				//
				////////////
				double t0 = timer<double>();
				DEBUG_FPRINTF(fil, "Moving data to the GPU. Starting at %21.15g\n", t0);
		#pragma omp barrier // not required barrier
				h2o4gpu::MatrixDense<T> A_(sharedA, me, wDev, Asource_);
		#pragma omp barrier // required barrier for wDev=sourceDev so that Asource_._data (etc.) is not overwritten inside h2o4gpu_data(wDev=sourceDev) below before other cores copy data
				h2o4gpu::H2O4GPUDirect<T, h2o4gpu::MatrixDense<T> > h2o4gpu_data(
						sharedA, me, wDev, A_);
		#pragma omp barrier // not required barrier
				double t1 = timer<double>();
				if (me == 0) { //only thread=0 times entire post-warmup procedure
					t1me0 = t1;
				}
				DEBUG_FPRINTF(fil, "Done moving data to the GPU. Stopping at %21.15g\n",
						t1);
				DEBUG_FPRINTF(fil, "Done moving data to the GPU. Took %g secs\n",
						t1 - t0);

				///////////////////////////////////////////////////
				// BEGIN SVD
				A_.svd1();
		}
		return 0;
	}
}
