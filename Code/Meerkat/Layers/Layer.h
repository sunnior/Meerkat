#ifndef __MEERKAT_LAYER_H__
#define __MEERKAT_LAYER_H__

#include "Common/Platform.h"
#include "Common/Tensor.h"
#include "Util/dl_stl.h"
#include "Reflection/Reflection.h"

namespace DeepLearning
{
	class Layer
	{
		DL_REFL_DECLARE(Layer);
	public:
		Layer(ComputeType type)
			: m_type(type)
		{};

		virtual ~Layer();

		void Forward(const Tensor* input, Tensor* output)
		{
			(m_type == ComputeType_CPU) ? _ForwardCpu(input, output) : _ForwardGpu(input, output);
		}

		void Backward(const Tensor* input, const Tensor* grad_input, Tensor* grad_output)
		{
			(m_type == ComputeType_CPU) ? _BackwardCpu(input, grad_input, grad_output) : _BackwardGpu(input, grad_input, grad_output);
		}

		virtual dl_tensor_shape GetOutputShape(const dl_tensor_shape& input_shape) = 0;

		virtual void CreateTrainData(dl_uint32 batch_size) {};

		void Optimize(class Optimizer* opti);
	protected:
		virtual void _ForwardGpu(const Tensor* input, Tensor* output) = 0;
		virtual void _ForwardCpu(const Tensor* input, Tensor* output) = 0;

		virtual void _BackwardGpu(const Tensor* input, const Tensor* grad_input, Tensor* grad_output) = 0;
		virtual void _BackwardCpu(const Tensor* input, const Tensor* grad_input, Tensor* grad_output) = 0;

		Tensor* _CreateTensor(const dl_tensor_shape& shape);

		virtual void _GetLearnableTensor(dl_vector<::std::pair<Tensor*, Tensor*>>& params) {};
	protected:
		ComputeType m_type;

	private:
		dl_vector<Tensor*> m_tensors;
	};

}
#endif