#include "LinearLayer.h"
#include "Blas.h"

namespace Meerkat
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

		input_data[0] = 1.0f;
		input_data[1] = 2.0f;
		weight_data[0] = 2.0f;
		weight_data[1] = 3.0f;
		
		cblas_sgemm_ptr(CblasRowMajor, CblasNoTrans, CblasTrans,
			m_size, n_size, k_size,
			1.0f, input_data, k_size,
			weight_data, k_size,
			0.0f, output_data, m_size);

			//typedef void(*cblas_sgemm_type)(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
				//OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);

			/*
		dl_uint32 d1 = m_weight->GetShape(0);
		dl_uint32 d2 = m_weight->GetShape(1);


		dl_tensor* input_data = input->GetCpuData();
		dl_tensor* output_data = output->GetCpuData();
		dl_tensor* weight_data = m_weight->GetCpuData();
		dl_tensor* bias_data = m_bias->GetCpuData();

		for (dl_uint32 i = 0; i < d2; ++i)
		{
			for (dl_uint32 j = 0; j < d1; ++j)
			{
				output_data[i] += input_data[i*d1 + j] * weight_data[i*d1 + j];
			}
			output_data[i] += bias_data[i];
		}*/
	}

	void LinearLayer::_ForwardGpu(Tensor* input, Tensor* output)
	{

	}

}
