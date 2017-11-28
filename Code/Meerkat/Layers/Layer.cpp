#include "Layer.h"
#include "Optimizer/Optimizer.h"

namespace DeepLearning
{

	void Layer::Update(Optimizer* optimizer)
	{
		Tensor* param;
		Tensor* grad_param;
		while (GetNextLearnablePair(param, grad_param))
		{
			optimizer->Optimize(param, grad_param);
		}
	}

}