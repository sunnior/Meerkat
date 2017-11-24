#ifndef __MEERKAT_LINEARLAYER_H__
#define __MEERKAT_LINEARLAYER_H__

#include "Layer.h"

namespace DeepLearning
{
	class LinearLayer : public Layer
	{
	public:
		LinearLayer(dl_uint32 input_num, dl_uint32 output_num);
		~LinearLayer();

		void Alloc(ComputeType type);
		Tensor* GetWeight() { return m_weight; }
		Tensor* GetBias() { return m_bias; }

	private:
		void _ForwardCpu(const Tensor* input, Tensor* output);
		void _ForwardGpu(const Tensor* input, Tensor* output);
	private:
		Tensor* m_weight{ nullptr };
		Tensor* m_bias{ nullptr };
	};
}

#endif