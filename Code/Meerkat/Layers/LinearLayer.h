#ifndef __MEERKAT_LINEAR_LAYER_H__
#define __MEERKAT_LINEAR_LAYER_H__

#include "Layer.h"

namespace DeepLearning
{
	class LinearLayer : public Layer
	{
	public:
		LinearLayer(ComputeType type, dl_uint32 input_num, dl_uint32 output_num);

		Tensor* GetWeight() { return m_weight; }
		Tensor* GetBias() { return m_bias; }

		Tensor* GetGradWeight() { return m_grad_weight; }
		Tensor* GetGradBias() { return m_grad_bias; }

		dl_tensor_shape GetOutputShape(const dl_tensor_shape& input_shape) override;

		void CreateTrainData(dl_uint32 batch_size) override;

	private:
		void _ForwardCpu(const Tensor* input, Tensor* output) override;
		void _ForwardGpu(const Tensor* input, Tensor* output) override;

		void _BackwardGpu(const Tensor* input, const Tensor* grad_input, Tensor* grad_output) override;
		void _BackwardCpu(const Tensor* input, const Tensor* grad_input, Tensor* grad_output) override;

		void _GetLearnableTensor(dl_vector<::std::pair<Tensor*, Tensor*>>& params) override;
	private:
		Tensor* m_weight{ nullptr };
		Tensor* m_bias{ nullptr };

		Tensor* m_grad_bias{ nullptr };
		Tensor* m_grad_weight{ nullptr };
		Tensor* m_bias_multi{ nullptr };
	};
}

#endif