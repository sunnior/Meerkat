#include "Linker.h"
#include "Layers/Layer.h"

namespace DeepLearning
{
	Linker::Linker(ComputeType type, Layer* layer)
		 : m_layer(layer)
		 , m_type(type)
	{
	}

	Linker::~Linker()
	{
		DL_SAFE_DELETE(m_data);
		DL_SAFE_DELETE(m_grad_data);
		DL_SAFE_DELETE(m_layer);
	}

	void Linker::ForwardRecurrent()
	{
		if (m_input_linker->IsForwardReady() == false)
		{
			m_input_linker->ForwardRecurrent();
		}
		
		if (m_layer)
		{
			m_layer->Forward(m_input_linker->GetData(), m_data);
			SetForwardReady(true);
		}
	}

	void Linker::BackwardRecurrent()
	{
		if (m_output_linker->IsBackwardReady() == false)
		{
			m_output_linker->BackwardRecurrent();
		}
		
		if (m_layer)
		{
			m_layer->Backward(m_input_linker->GetData(), m_output_linker->GetGradData(), m_grad_data);
			SetBackwardReady(true);
		}
	}

	void Linker::ClearState()
	{
		SetBackwardReady(false);
		SetForwardReady(false);
	}

	void Linker::CreateData(dl_uint32 batch_size, const dl_tensor_shape& data_shape)
	{
		dl_tensor_shape shape(data_shape);
		shape.insert(shape.begin(), batch_size);
		m_data = DL_NEW(Tensor)(m_type, shape);
	}

	void Linker::CreateDataRecurrent(dl_uint32 batch_size, bool if_train)
	{
		Tensor* input_data = m_input_linker->GetData();
		if (input_data == nullptr)
		{
			m_input_linker->CreateDataRecurrent(batch_size, if_train);
			input_data = m_input_linker->GetData();
		}

		const dl_tensor_shape& input_shape = input_data->GetShape();
		dl_tensor_shape shape(input_shape.begin() + 1, input_shape.end()); //the first is batch size
		m_grad_data = DL_NEW(Tensor)(m_type, input_shape);

		if (m_layer)
		{
			m_layer->CreateData(m_type);

			if (if_train)
			{
				m_layer->CreateTrainData(batch_size);
			}

			dl_tensor_shape output_shape = m_layer->GetOutputShape(shape);
			output_shape.insert(output_shape.begin(), batch_size);
			m_data = DL_NEW(Tensor)(m_type, output_shape);
		}

	}

	void Linker::Optimize(class Optimizer* opti)
	{
		if (m_layer)
		{
			m_layer->Optimize(opti);
		}
	}

}