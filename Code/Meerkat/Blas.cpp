#include "Blas.h"
#include <Windows.h>

namespace Meerkat
{
	cblas_sgemm_type cblas_sgemm_ptr;

	void InitBlas()
	{
		HINSTANCE hDLL = LoadLibrary("libopenblas.dll");
		cblas_sgemm_ptr = (cblas_sgemm_type)GetProcAddress(hDLL, "cblas_sgemm");
	}
}
