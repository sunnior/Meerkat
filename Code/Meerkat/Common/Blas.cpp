#include "Blas.h"
#include <Windows.h>

namespace DeepLearning
{
	cblas_sgemm_type cblas_sgemm_ptr;
	cblas_saxpy_type cblas_saxpy_ptr;
	cblas_scopy_type cblas_scopy_ptr;
	cblas_sdot_type cblas_sdot_ptr;

	void InitBlas()
	{
		HINSTANCE hDLL = LoadLibrary("libopenblas.dll");
		cblas_sgemm_ptr = (cblas_sgemm_type)GetProcAddress(hDLL, "cblas_sgemm");
		cblas_saxpy_ptr = (cblas_saxpy_type)GetProcAddress(hDLL, "cblas_saxpy");
		cblas_scopy_ptr = (cblas_scopy_type)GetProcAddress(hDLL, "cblas_scopy");
		cblas_sdot_ptr = (cblas_sdot_type)GetProcAddress(hDLL, "cblas_sdot");
	}


}
