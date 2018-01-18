#include "LinearLayer.h"
#include "Common/Blas.h"

namespace DeepLearning
{
	DL_REFL_IMPLEMENT(LinearLayer, "layer_linear");

	void LinearLayer::FromJson(const rapidjson::Value& layer_json)
	{
		m_input_num = layer_json["input"].GetInt();
		m_output_num = layer_json["output"].GetInt();
	}

	void LinearLayer::ToJson(rapidjson::PrettyWriter<rapidjson::StringBuffer>& writer)
	{
		writer.Key("input");
		writer.Uint(m_input_num);
		writer.Key("output");
		writer.Uint(m_output_num);
	}

	dl_tensor_shape LinearLayer::GetOutputShape(const dl_tensor_shape& input_shape)
	{
		return dl_tensor_shape(m_bias->GetShape());
	}

	void LinearLayer::_CreateData()
	{
		m_weight = _CreateTensor({ m_input_num, m_output_num });
		m_bias = _CreateTensor({ m_output_num });
	}

	void LinearLayer::CreateTrainData(dl_uint32 batch_size)
	{
		m_grad_weight = _CreateTensor(m_weight->GetShape());
		m_grad_bias = _CreateTensor(m_bias->GetShape());

		m_bias_multi = _CreateTensor({ batch_size });
		m_bias_multi->FillWith((dl_tensor)1.0f);
	}

	void LinearLayer::_ForwardCpu(const Tensor* input, Tensor* output)
	{
		TODO("check dimension");
		output->Zeros();

		dl_uint32 m_size = input->GetShape(0);//batch_size
		dl_uint32 k_size = m_weight->GetShape(0);
		dl_uint32 n_size = output->GetShape(1);


		const dl_tensor* input_data = input->GetData();
		dl_tensor* output_data = output->GetData();
		dl_tensor* weight_data = m_weight->GetData();
		dl_tensor* bias_data = m_bias->GetData();

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

	/*
		void LinearLayer::_BackwardCpu(const Tensor* input, const Tensor* grad_input, Tensor* grad_output)
		{
			dl_uint32 batch_size = input->GetShape(0);
			dl_uint32 weight_size = m_weight->GetShape(0);
			dl_uint32 bias_size = m_bias->GetShape(0);

			_CreateGradParam(batch_size);

			const dl_tensor* input_data = input->GetData();
			const dl_tensor* grad_input_data = grad_input->GetData();
			const dl_tensor* weight_data = m_weight->GetData();
			dl_tensor* grad_weight = m_grad_weight->GetData();

			const dl_tensor* bias_multi_data = m_bias_multi->GetData();
			dl_tensor* grad_bias_data = m_grad_bias->GetData();

			dl_gemm_cpu<dl_tensor>(CblasRowMajor, CblasTrans, CblasNoTrans, weight_size, bias_size, batch_size, (dl_tensor)1.0f, input_data, grad_input_data, (dl_tensor)1.0f, grad_weight);
			dl_gemm_cpu<dl_tensor>(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, bias_size, batch_size, (dl_tensor)1.0f, bias_multi_data, grad_input_data, 1.0f, grad_bias_data);

			if (grad_output) {
				dl_tensor* grad_output_data = grad_output->GetData();
				dl_gemm_cpu<dl_tensor>(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, weight_size, bias_size, (dl_tensor)1.0f, grad_input_data, weight_data, (dl_tensor)0.0f, grad_output_data);
			}
		}
	*/

	void LinearLayer::_BackwardCpu(const Tensor* input, const Tensor* grad_input, Tensor* grad_output)
	{
		dl_uint32 batch_size = input->GetShape(0);
		dl_uint32 weight_size = m_weight->GetShape(0);
		dl_uint32 bias_size = m_bias->GetShape(0);

		const dl_tensor* input_data = input->GetData();
		const dl_tensor* grad_input_data = grad_input->GetData();
		const dl_tensor* weight_data = m_weight->GetData();
		dl_tensor* grad_weight_data = m_grad_weight->GetData();
		dl_tensor* grad_bias_data = m_grad_bias->GetData();

		for (dl_uint32 bias_idx = 0; bias_idx < bias_size; ++bias_idx)
		{
			for (dl_uint32 weight_idx = 0; weight_idx < weight_size; ++weight_idx)
			{
				for (dl_uint32 batch_idx = 0; batch_idx < batch_size; ++batch_idx)
				{
					*(grad_weight_data + bias_idx*weight_size + weight_idx) += *(input_data + batch_idx*weight_size + weight_idx) * *(grad_input_data + batch_idx*bias_size + bias_idx);
				}
			}

			for (dl_uint32 batch_idx = 0; batch_idx < batch_size; ++batch_idx)
			{
				*(grad_bias_data + bias_idx) += *(grad_input_data + batch_idx*bias_size + bias_idx);
			}
		}

		if (grad_output)
		{
			grad_output->Zeros();
			dl_tensor* grad_output_data = grad_output->GetData();
			for (dl_uint32 batch_idx = 0; batch_idx < batch_size; ++batch_idx)
			{
				for (dl_uint32 weight_idx = 0; weight_idx < weight_size; ++weight_idx)
				{
					for (dl_uint32 bias_idx = 0; bias_idx < bias_size; ++bias_idx)
					{
						*(grad_output_data + batch_idx*weight_size + weight_idx) += *(grad_input_data + batch_idx*bias_size + bias_idx) * *(weight_data + bias_idx*weight_size + weight_idx);
					}
				}
			}
		}

	}

	void LinearLayer::_GetLearnableTensor(dl_vector<::std::pair<Tensor*, Tensor*>>& params)
	{
		params.push_back(::std::pair<Tensor*, Tensor*>(m_weight, m_grad_weight));
		params.push_back(::std::pair<Tensor*, Tensor*>(m_bias, m_grad_bias));
	}

	void LinearLayer::_BackwardGpu(const Tensor* input, const Tensor* grad_input, Tensor* grad_output)
	{

	}


}
