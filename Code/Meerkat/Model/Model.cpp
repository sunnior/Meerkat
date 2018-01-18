#include "Model.h"
#include "Layers/Layer.h"

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

	void Model::Deserialize(const rapidjson::Document& doc)
	{
		const rapidjson::Value& layer_arr = doc["layers"];
		dl_uint32 layer_num = layer_arr.Size();
		for (dl_uint32 i = 0; i < layer_num; ++i)
		{
			const rapidjson::Value& layer_json = layer_arr[i];
			Layer* layer = ReflectionManager::GetInstance()->CreateLayer(m_type, layer_json);
			Linker* linker = DL_NEW(Linker)(m_type, layer);
			linker->SetName(dl_string(layer_json["name"].GetString()));
			m_linkers.insert(::std::pair<dl_string, Linker*>(dl_string(layer_json["name"].GetString()), linker));
		}

		const rapidjson::Value& links = doc["links"];
		LinkBegin(links["begin"].GetString());
		LinkEnd(links["end"].GetString());
		const rapidjson::Value& links_internal = links["internal"];

		dl_uint32 link_num = links_internal.Size();
		for (dl_uint32 i = 0; i < link_num; ++i)
		{
			Link(links_internal[i][0].GetString(), links_internal[i][1].GetString());
		}
	}

	void Model::Serialize(rapidjson::PrettyWriter<rapidjson::StringBuffer>& writer)
	{
		writer.StartObject(); {
			writer.Key("layers");
			writer.StartArray(); {
				for (auto& it : m_linkers)
				{
					writer.StartObject();
					Layer* layer = it.second->GetLayer();
					writer.Key("type");
					writer.String(layer->GetTypeName());
					writer.Key("name");
					writer.String(it.first.c_str());
					layer->ToJson(writer);
					writer.EndObject();
				}
			} writer.EndArray();

			writer.Key("links");
			writer.StartObject(); {
				writer.Key("begin");
				writer.String(m_begin_linker->GetOutput()->GetLayer()->GetTypeName());
				writer.Key("end");
				writer.String(m_end_linker->GetInput()->GetLayer()->GetTypeName());
				writer.Key("internal");
				writer.StartArray(); {
					for (auto& it : m_linkers)
					{
						Linker* linker_begin = it.second;
						Linker* linker_end = linker_begin->GetOutput();
						if (linker_end->GetLayer())
						{
							writer.StartArray(); {
								writer.String(linker_begin->GetName().c_str());
								writer.String(linker_end->GetName().c_str());
							} writer.EndArray();
						}
					}
				}; writer.EndArray();

			}; writer.EndObject();

		};  writer.EndObject();
	}

}