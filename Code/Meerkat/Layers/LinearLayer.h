#ifndef __MEERKAT_LINEAR_LAYER_H__
#define __MEERKAT_LINEAR_LAYER_H__

#include "Layer.h"

namespace DeepLearning
{
	class LinearLayer : public Layer
	{
	public:
		LinearLayer(ComputeType type, dl_uint32 input_num, dl_uint32 output_num);
		~LinearLayer();

		Tensor* GetWeight() { return m_weight; }
		Tensor* GetBias() { return m_bias; }

		Tensor* GetGradWeight() { return m_grad_weight; }
		Tensor* GetGradBias() { return m_grad_bias; }

		bool GetNextLearnablePair(Tensor*& param, Tensor*& grad_param) override;

	private:
		void _ForwardCpu(const Tensor* input, Tensor* output) override;
		void _ForwardGpu(const Tensor* input, Tensor* output) override;

		void _BackwardGpu(const Tensor* input, const Tensor* grad_input, Tensor* grad_output) override;
		void _BackwardCpu(const Tensor* input, const Tensor* grad_input, Tensor* grad_output) override;

		void _CreateGradParam(dl_uint32 batch_size);

	private:
		Tensor* m_weight{ nullptr };
		Tensor* m_bias{ nullptr };

		Tensor* m_grad_bias{ nullptr };
		Tensor* m_grad_weight{ nullptr };
		Tensor* m_bias_multi{ nullptr };

		dl_uint32 m_next_learnable_pair{ 0 };
	};
}

#endif