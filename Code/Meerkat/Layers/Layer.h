#ifndef __MEERKAT_LAYER_H__
#define __MEERKAT_LAYER_H__

#include "Common/Platform.h"
#include "Common/Tensor.h"

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

		void GetLearnableParam(Tensor**& params, Tensor**& param_grads, dl_uint32& count)
		{
			params = m_learnable_params;
			param_grads = m_learnable_param_grads;
			count = m_learnable_param_count;
		}
	protected:
		virtual void _ForwardGpu(const Tensor* input, Tensor* output) = 0;
		virtual void _ForwardCpu(const Tensor* input, Tensor* output) = 0;

		virtual void _BackwardGpu(const Tensor* input, const Tensor* grad_input, Tensor* grad_output) = 0;
		virtual void _BackwardCpu(const Tensor* input, const Tensor* grad_input, Tensor* grad_output) = 0;

		void _CreateLearnableTensor(Tensor*& param, Tensor*& param_grad, std::initializer_list<dl_uint32> shape);
	protected:
		ComputeType m_type;
		const bool  m_if_train;

	private:
		static const dl_uint32 s_max_learnable_param_count = 16;
		Tensor* m_learnable_params[s_max_learnable_param_count]{ nullptr };
		Tensor* m_learnable_param_grads[s_max_learnable_param_count]{ nullptr };
		dl_uint32 m_learnable_param_count{ 0 };
	};

}
#endif