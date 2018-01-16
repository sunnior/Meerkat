#include "Linker.h"
#include "Layers/Layer.h"

namespace DeepLearning
{
	Linker::Linker(ComputeType type, Layer* layer)
		 : m_layer(layer)
		 , m_type(type)
	{
	}

	void Linker::ForwardRecurrent()
	{
		if (m_input_linker->IsForwardReady() == false)
		{
			m_input_linker->ForwardRecurrent();
		}

		m_layer->Forward(m_input_linker->GetData(), m_data);
		SetForwardReady(true);
	}

	void Linker::BackwardRecurrent()
	{
		if (m_output_linker->IsBackwardReady() == false)
		{
			m_output_linker->BackwardRecurrent();
		}

		m_layer->Backward(m_input_linker->GetData(), m_output_linker->GetGradData(), m_grad_data);
		SetBackwardReady(true);
	}

	void Linker::CreateData(dl_uint32 batch_size, const dl_tensor_shape& data_shape)
	{
		dl_tensor_shape shape(data_shape);
		shape.insert(shape.begin(), batch_size);
		m_data = DL_NEW(Tensor)(m_type, shape);
	}

	void Linker::CreateDataRecurrent(dl_uint32 batch_size)
	{
		Tensor* input_data = m_input_linker->GetData();
		if (input_data == nullptr)
		{
			m_input_linker->CreateDataRecurrent(batch_size);
			input_data = m_input_linker->GetData();
		}

		const dl_tensor_shape& input_shape = input_data->GetShape();
		dl_tensor_shape shape(input_shape.begin() + 1, input_shape.end()); //the first is batch size
		m_grad_data = DL_NEW(Tensor)(m_type, input_shape);

		if (m_layer)
		{
			dl_tensor_shape output_shape = m_layer->GetOutputShape(shape);
			output_shape.insert(output_shape.begin(), batch_size);
			m_data = DL_NEW(Tensor)(m_type, output_shape);
		}

	}

	void Linker::GetLearnableParam(const dl_vector<Tensor*>*& params, const dl_vector<Tensor*>*& param_grads)
	{
		m_layer->GetLearnableParam(params, param_grads);
	}
}