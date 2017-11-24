#include "LinearLayer.h"
#include "Blas.h"

namespace DeepLearning
{
	LinearLayer::LinearLayer(dl_uint32 input_num, dl_uint32 output_num)
	{
		m_weight = DL_NEW(Tensor){input_num, output_num};
		m_bias = DL_NEW(Tensor){output_num};
	}

	LinearLayer::~LinearLayer()
	{
		DL_SAFE_DELETE(m_weight);
		DL_SAFE_DELETE(m_bias);
	}

	void LinearLayer::Alloc(ComputeType type)
	{
		m_weight->Alloc(type);
		m_bias->Alloc(type);
		TODO("way to init");
		m_weight->Zeros();
		m_bias->Zeros();
	}

	void LinearLayer::_ForwardCpu(Tensor* input, Tensor* output)
	{
		TODO("check dimension");
		output->Zeros();

		dl_uint32 m_size = input->GetShape(0);//batch_size
		dl_uint32 k_size = m_weight->GetShape(0);
		dl_uint32 n_size = output->GetShape(1);


		dl_tensor* input_data = input->GetCpuData();
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

	void LinearLayer::_ForwardGpu(Tensor* input, Tensor* output)
	{

	}

}
