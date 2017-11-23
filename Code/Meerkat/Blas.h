#ifndef __MEERKAT_BLAS_H__
#define __MEERKAT_BLAS_H__
#include "cblas.h"
#include <cstring>

namespace DeepLearning
{
	typedef void (*cblas_sgemm_type)(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
	extern cblas_sgemm_type cblas_sgemm_ptr;

	template<typename T>
	void dl_gemm_cpu(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N, const blasint K,
		const T alpha, const T *A, const blasint lda, const T *B, const blasint ldb, const T beta, T *C, const blasint ldc);

	template<>
	inline
	void dl_gemm_cpu<float>(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N, const blasint K,
		const float alpha, const float *A, const blasint lda, const float *B, const blasint ldb, const float beta, float *C, const blasint ldc)
	{
		cblas_sgemm_ptr(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
	}

	template<typename T>
	void dl_memcpy_cpu(T* dest, const T* src, const blasint size, const blasint repeat)
	{
		for (blasint i = 0; i < repeat; ++i)
		{
			std::memcpy(dest, src, sizeof(T)*size);
			dest += size;
		}
	}

	void InitBlas();
}

#endif