#ifndef __MEERKAT_LAYER_H__
#define __MEERKAT_LAYER_H__

#include "Common/Platform.h"
#include "Common/Tensor.h"

namespace DeepLearning
{
	class Layer
	{
	public:
		struct GradParam
		{
			dl_uint32 m_tensor_num;
			Tensor**  m_tensors;
		};

	public:
		Layer(ComputeType type)
			: m_type(type)
		{};

		virtual ~Layer() {};

		void Forward(const Tensor* input, Tensor* output)
		{
			(m_type == ComputeType_CPU) ? _ForwardCpu(input, output) : _ForwardGpu(input, output);
		}

		void Backward(const Tensor* intput, const Tensor* grad_input, Tensor* grad_output, GradParam* grad_param)
		{
			(m_type == ComputeType_CPU) ? _BackwardCpu(intput, grad_input, grad_output, grad_param) : _BackwardGpu(intput, grad_input, grad_output, grad_param);
		}

		virtual GradParam* CreateGradParam(dl_uint32 batch_size) = 0;
	protected:
		virtual void _ForwardGpu(const Tensor* input, Tensor* output) = 0;
		virtual void _ForwardCpu(const Tensor* input, Tensor* output) = 0;

		virtual void _BackwardGpu(const Tensor* intput, const Tensor* grad_input, Tensor* grad_output, GradParam* grad_param) = 0;
		virtual void _BackwardCpu(const Tensor* intput, const Tensor* grad_input, Tensor* grad_output, GradParam* grad_param) = 0;


	protected:
		ComputeType m_type;
	};

}
#endif