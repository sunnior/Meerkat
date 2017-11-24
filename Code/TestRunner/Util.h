#include "UnitTest++.h"
#include "Meerkat.h"



#define DL_CHECK_CLOSE(a, b) CHECK_CLOSE(a, b, 0.0000001f)



namespace DeepLearning

{

	bool dl_check_cpumem_close(Tensor* pA, Tensor* pB)

	{

		if (pA->GetSize() != pB->GetSize())

		{

			return false;

		}



		dl_size size = pA->GetSize();

		dl_tensor* pDataA = pA->GetCpuData();

		dl_tensor* pDataB = pB->GetCpuData();



		dl_tensor tolerance = 0.00001f;

		for (dl_size i = 0; i < size; ++i)

		{

			if ((pDataA[i] >= (pDataB[i] - tolerance)) && (pDataA[i] <= (pDataB[i] + tolerance))) {

				continue;

			}

			return false;

		}

		return true;

	}

}