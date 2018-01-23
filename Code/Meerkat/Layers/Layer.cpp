#include "Layer.h"
#include "Optimizer/Optimizer.h"

namespace DeepLearning
{

	Layer::~Layer()
	{
		dl_uint32 tensor_count = m_tensors.size();
		for (auto& it : m_tensors)
		{
			DL_SAFE_DELETE(it);
		}

		m_tensors.clear();
	}

	void Layer::Optimize(class Optimizer* opti)
	{
		dl_vector<::std::pair<Tensor*, Tensor*>> params;
		_GetLearnableTensor(params);
		opti->Update(params);
	}

	Tensor* Layer::_CreateTensor(const dl_tensor_shape& shape)
	{
		Tensor* tensor = DL_NEW(Tensor)(m_type, shape);
		m_tensors.push_back(tensor);

		tensor->Zeros();
		return tensor;
	}

}