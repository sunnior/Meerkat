#ifndef __MEERKAT_LINEARLAYER_H__
#define __MEERKAT_LINEARLAYER_H__

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

	private:
		void _ForwardCpu(const Tensor* input, Tensor* output) override;
		void _ForwardGpu(const Tensor* input, Tensor* output) override;
	private:
		Tensor* m_weight{ nullptr };
		Tensor* m_bias{ nullptr };
	};
}

#endif