#include "Model.h"
#include "Reflection/Reflection.h"
#include <cstdarg>

namespace DeepLearning
{
	Model::Model(ComputeType type, bool if_train)
		: m_type(type)
		, m_if_train(if_train)
	{
		m_begin_linker = DL_NEW(Linker)(m_type);
		m_end_linker = DL_NEW(Linker)(m_type);
		m_begin_linker->SetForwardReady(true);
		m_end_linker->SetBackwardReady(true);
	}

	Model::~Model()
	{
		for (auto it : m_linkers)
		{
			DL_SAFE_DELETE(it.second);
		}

		m_linkers.clear();
		DL_SAFE_DELETE(m_begin_linker);
		DL_SAFE_DELETE(m_end_linker);
	}

	void Model::CreateLayer(const char* class_name, const char* layer_name, ...)
	{
		va_list vl;
		va_start(vl, layer_name);
		Layer* layer = Reflection::CreateLayer(class_name, m_type, m_if_train, vl);
		Linker* linker = DL_NEW(Linker)(m_type, layer);
		m_linkers.insert(::std::pair<dl_string, Linker*>(dl_string(layer_name), linker));
	}

	void Model::Link(const char* input_name, const char* output_name)
	{
		Linker* input = m_linkers.find(input_name)->second;
		Linker* output = m_linkers.find(output_name)->second;

		input->SetOutput(output);
		output->SetInput(input);
	}

	void Model::LinkBegin(const char* name)
	{
		Linker* linker = m_linkers.find(name)->second;
		m_begin_linker->SetOutput(linker);
		linker->SetInput(m_begin_linker);
	}

	void Model::LinkEnd(const char* name)
	{
		Linker* linker = m_linkers.find(name)->second;
		m_end_linker->SetInput(linker);
		linker->SetOutput(m_end_linker);
	}

	void Model::GetLearnableParam(dl_vector<Tensor*>& params, dl_vector<Tensor*>& param_grads)
	{
		for (auto it : m_linkers)
		{
			const dl_vector<Tensor*>* p_params;
			const dl_vector<Tensor*>* p_param_grads;
			it.second->GetLearnableParam(p_params, p_param_grads);
			params.insert(params.end(), p_params->begin(), p_params->end());
			param_grads.insert(param_grads.end(), p_param_grads->begin(), p_param_grads->end());
		}
	}

	void Model::CreateData(dl_uint32 batch_size, const dl_tensor_shape& data_shape)
	{
		m_begin_linker->CreateData(batch_size, data_shape);
		m_end_linker->CreateDataRecurrent(batch_size);
	}

	void Model::Forward()
	{
		m_end_linker->ForwardRecurrent();
	}


	void Model::Backward()
	{
		m_begin_linker->BackwardRecurrent();
	}

	void Model::ClearState()
	{
		for (auto it : m_linkers)
		{
			it.second->ClearState();
		}
	}

}