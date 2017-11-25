#include "LinearLayer.h"
#include "Common/Blas.h"

namespace DeepLearning
{
	LinearLayer::LinearLayer(ComputeType type, dl_uint32 input_num, dl_uint32 output_num)
		: Layer(type)
	{
		m_weight = DL_NEW(Tensor)(type, {input_num, output_num});
		m_bias = DL_NEW(Tensor)(type, {output_num});
	}

	LinearLayer::~LinearLayer()
	{
		DL_SAFE_DELETE(m_weight);
		DL_SAFE_DELETE(m_bias);
	}

	void LinearLayer::_ForwardCpu(const Tensor* input, Tensor* output)
	{
		TODO("check dimension");
		output->Zeros();

		dl_uint32 m_size = input->GetShape(0);//batch_size
		dl_uint32 k_size = m_weight->GetShape(0);
		dl_uint32 n_size = output->GetShape(1);


		const dl_tensor* input_data = input->GetCpuData();
		dl_tensor* output_data = output->GetCpuData();
		dl_tensor* weight_data = m_weight->GetCpuData();
		dl_tensor* bias_data = m_bias->GetCpuData();

		dl_memcpy_cpu<dl_tensor>(output_data, bias_data, n_size, m_size);

		dl_gemm_cpu<dl_tensor>(CblasRowMajor, CblasNoTrans, CblasTrans,
			m_size, n_size, k_size,
			1.0f, input_data, k_size,
			weight_data, k_size,
			1.0f, output_data, m_size);


	}

	void LinearLayer::_ForwardGpu(const Tensor* input, Tensor* output)
	{

	}

}
