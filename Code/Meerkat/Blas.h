#ifndef __MEERKAT_BLAS_H__
#define __MEERKAT_BLAS_H__
#include "cblas.h"

namespace DeepLearning
{
	typedef void (*cblas_sgemm_type)(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
	extern cblas_sgemm_type cblas_sgemm_ptr;

	template<typename T>
	void dl_gemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		OPENBLAS_CONST T alpha, OPENBLAS_CONST T *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST T *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST T beta, T *C, OPENBLAS_CONST blasint ldc);

	template<>
	inline
	void dl_gemm<float>(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc)
	{
		cblas_sgemm_ptr(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
	}

	void InitBlas();
}

#endif