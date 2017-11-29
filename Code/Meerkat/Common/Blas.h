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
		const T alpha, const T *A, const T *B, const T beta, T *C);

	template<>
	inline void dl_gemm_cpu<float>(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N, const blasint K,
			const float alpha, const float *A, const float *B, const float beta, float *C)
	{
		const blasint lda = (TransA == CblasNoTrans) ? K : M;
		const blasint ldb = (TransB == CblasNoTrans) ? N : K;
		const blasint ldc = N;
		cblas_sgemm_ptr(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
	}

	typedef void (*cblas_saxpy_type)(OPENBLAS_CONST blasint n, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx, float *y, OPENBLAS_CONST blasint incy);
	extern cblas_saxpy_type cblas_saxpy_ptr;

	template<typename T>
	void dl_axpy_cpu(const blasint n, const T alpha, const T *x, const blasint incx, T *y, const blasint incy);

	template<>
	inline void dl_axpy_cpu<float>(const blasint n, const float alpha, const float *x, const blasint incx, float *y, const blasint incy)
	{
		cblas_saxpy_ptr(n, alpha, x, incx, y, incy);
	}


	typedef void (*cblas_scopy_type)(OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx, float *y, OPENBLAS_CONST blasint incy);
	extern cblas_scopy_type cblas_scopy_ptr;

	template<typename T>
	void dl_copy_cpu(const blasint n, const T *x, const blasint incx, T *y, const blasint incy);

	template<>
	inline void dl_copy_cpu<float>(const blasint n, const float *x, const blasint incx, float *y, const blasint incy)
	{
		cblas_scopy_ptr(n, x, incx, y, incy);
	}

	typedef float (*cblas_sdot_type)(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST float  *y, OPENBLAS_CONST blasint incy);
	extern cblas_sdot_type cblas_sdot_ptr;

	template<typename T>
	T dl_dot_cpu(const blasint n, const T *x, const blasint incx, const T *y, const blasint incy);

	template<>
	inline float dl_dot_cpu<float>(const blasint n, const float *x, const blasint incx, const float *y, const blasint incy)
	{
		return cblas_sdot_ptr(n, x, incx, y, incy);
	}

	typedef void (*cblas_sscal_type)(OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, float *X, OPENBLAS_CONST blasint incX);
	extern cblas_sscal_type cblas_sscal_ptr;

	template<typename T>
	void dl_scal_cpu(const blasint N, const T alpha, T *X, const blasint incX);

	template<>
	inline void dl_scal_cpu<float>(const blasint N, const float alpha, float *X, const blasint incX)
	{
		cblas_sscal_ptr(N, alpha, X, incX);
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