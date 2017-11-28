#include "Layer.h"
#include "Optimizer/Optimizer.h"

namespace DeepLearning
{

	Layer::~Layer()
	{
		while (m_learnable_param_count)
		{
			--m_learnable_param_count;
			DL_SAFE_DELETE(m_learnable_params[m_learnable_param_count]);
			if (m_if_train)
			{
				DL_SAFE_DELETE(m_learnable_param_grads[m_learnable_param_count]);
			}
		}
	}

	void Layer::_CreateLearnableTensor(Tensor*& param, Tensor*& param_grad, std::initializer_list<dl_uint32> shape)
	{
		Tensor* tensor = DL_NEW(Tensor)(m_type, shape);
		m_learnable_params[m_learnable_param_count] = tensor;
		tensor->Zeros();
		param = tensor;

		if (m_if_train)
		{
			Tensor* tensor_grad = DL_NEW(Tensor)(m_type, shape);
			m_learnable_param_grads[m_learnable_param_count] = tensor_grad;
			tensor_grad->Zeros();
			param_grad = tensor_grad;
		}
		++m_learnable_param_count;
	}

}