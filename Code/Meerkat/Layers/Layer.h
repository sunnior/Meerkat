#ifndef __MEERKAT_LAYER_H__
#define __MEERKAT_LAYER_H__

#include "Common/Platform.h"
#include "Common/Tensor.h"
#include "Util/dl_stl.h"

namespace DeepLearning
{
	class Layer
	{
	public:
		Layer(ComputeType type, bool if_train)
			: m_type(type)
			, m_if_train(if_train)
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

		void GetLearnableParam(const dl_vector<Tensor*>*& params, const dl_vector<Tensor*>*& param_grads)
		{
			params = &m_learnable_params;
			param_grads = &m_learnable_param_grads;
		}

		virtual dl_tensor_shape GetOutputShape(const dl_tensor_shape& input_shape) = 0;

	protected:
		virtual void _ForwardGpu(const Tensor* input, Tensor* output) = 0;
		virtual void _ForwardCpu(const Tensor* input, Tensor* output) = 0;

		virtual void _BackwardGpu(const Tensor* input, const Tensor* grad_input, Tensor* grad_output) = 0;
		virtual void _BackwardCpu(const Tensor* input, const Tensor* grad_input, Tensor* grad_output) = 0;

		void _CreateLearnableTensor(Tensor*& param, Tensor*& param_grad, dl_tensor_shape shape);
	protected:
		ComputeType m_type;
		const bool  m_if_train;

	private:
		dl_vector<Tensor*> m_learnable_params;
		dl_vector<Tensor*> m_learnable_param_grads;
	};

}
#endif