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
			1.0f, input_data,
			weight_data,
			1.0f, output_data);

	}

	void LinearLayer::_ForwardGpu(const Tensor* input, Tensor* output)
	{

	}

	void LinearLayer::_CreateGradParam(dl_uint32 batch_size)
	{
		if (m_grad_bias != nullptr)
		{
			return;
		}

		dl_uint32 weight_size = m_weight->GetShape(0);
		dl_uint32 bias_size = m_bias->GetShape(0);

		m_grad_bias = DL_NEW(Tensor)(m_type, { bias_size });
		m_grad_weight = DL_NEW(Tensor)(m_type, { weight_size, bias_size });
		m_bias_multi = DL_NEW(Tensor)(m_type, { batch_size });

		m_grad_bias->Zeros();
		m_grad_weight->Zeros();
		m_bias_multi->FillWith((dl_tensor)1.0f);
	}


	void LinearLayer::_BackwardCpu(const Tensor* input, const Tensor* grad_input, Tensor* grad_output)
	{
		dl_uint32 batch_size = input->GetShape(0);
		dl_uint32 weight_size = m_weight->GetShape(0);
		dl_uint32 bias_size = m_bias->GetShape(0);

		_CreateGradParam(batch_size);

		const dl_tensor* input_data = input->GetCpuData();
		const dl_tensor* grad_input_data = grad_input->GetCpuData();
		dl_tensor* grad_output_data = grad_output->GetCpuData();
		const dl_tensor* weight_data = m_weight->GetCpuData();
		dl_tensor* grad_weight = m_grad_weight->GetCpuData();

		const dl_tensor* bias_multi_data = m_bias_multi->GetCpuData();
		dl_tensor* grad_bias_data = m_grad_bias->GetCpuData();

		dl_gemm_cpu<dl_tensor>(CblasRowMajor, CblasTrans, CblasNoTrans, batch_size, bias_size, weight_size, (dl_tensor)1.0f, input_data, grad_input_data, (dl_tensor)1.0f, grad_weight);
		dl_gemm_cpu<dl_tensor>(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, weight_size, bias_size, (dl_tensor)1.0f, grad_input_data, weight_data, (dl_tensor)0.0f, grad_output_data);
		dl_gemm_cpu<dl_tensor>(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, bias_size, batch_size, (dl_tensor)1.0f, bias_multi_data, grad_input_data, 1.0f, grad_bias_data);
	}

	void LinearLayer::_BackwardGpu(const Tensor* input, const Tensor* grad_input, Tensor* grad_output)
	{
		
	}


}
