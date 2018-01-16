#include "Model.h"
#include "Reflection/Reflection.h"
#include <cstdarg>

namespace DeepLearning
{
	Model::Model(ComputeType type)
		: m_type(type)
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
		Layer* layer = Reflection::CreateLayer(class_name, m_type, vl);
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

	void Model::CreateData(dl_uint32 batch_size, const dl_tensor_shape& data_shape, bool if_train)
	{
		m_begin_linker->CreateData(batch_size, data_shape);
		m_end_linker->CreateDataRecurrent(batch_size, if_train);
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

	void Model::Optimize(class Optimizer* opti)
	{
		for (auto it : m_linkers)
		{
			it.second->Optimize(opti);
		}
	}

}