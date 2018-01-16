#include "Layer.h"
#include "Optimizer/Optimizer.h"

namespace DeepLearning
{

	Layer::~Layer()
	{
		dl_uint32 param_size = m_learnable_params.size();
		for (dl_uint32 i = 0; i < param_size; ++i)
		{
			DL_SAFE_DELETE(m_learnable_params[i]);
			if (m_if_train)
			{
				DL_SAFE_DELETE(m_learnable_param_grads[i]);

			}

		}

		m_learnable_params.clear();
		m_learnable_param_grads.clear();
	}

	void Layer::_CreateLearnableTensor(Tensor*& param, Tensor*& param_grad, dl_tensor_shape shape)
	{
		Tensor* tensor = DL_NEW(Tensor)(m_type, shape);
		m_learnable_params.push_back(tensor);
		tensor->Zeros();
		param = tensor;

		if (m_if_train)
		{
			Tensor* tensor_grad = DL_NEW(Tensor)(m_type, shape);
			m_learnable_param_grads.push_back(tensor_grad);
			tensor_grad->Zeros();
			param_grad = tensor_grad;
		}
	}

}