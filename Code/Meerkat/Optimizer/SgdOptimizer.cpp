#include "SgdOptimizer.h"
#include "Common/Blas.h"
#include "Common/Tensor.h"

namespace DeepLearning
{
	void SgdOptimizer::Update()
	{
		dl_tensor learning_rate = m_learning_rate_base / (1 + m_eval*0.0001f);
		++m_eval;
		for (dl_uint32 i = 0; i < m_count; ++i)
		{
			dl_size nelement = m_params[i]->GetSize();
			dl_axpy_cpu<dl_tensor>(nelement, -1.0f*learning_rate, m_grad_params[i]->GetData(), 1, m_params[i]->GetData(), 1);
			m_grad_params[i]->Zeros();
		}

	}

}