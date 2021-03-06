#ifndef __MEERKAT_MODEL_H__
#define __MEERKAT_MODEL_H__

#include "Common/Tensor.h"
#include "Reflection/Reflection.h"
#include "Reflection/TensorWriter.h"
#include "Reflection/TensorReader.h"
#include "Linker.h"


namespace DeepLearning
{
	class Model
	{
	public:
		Model(ComputeType type);
		~Model();

		void Link(const char* input_name, const char* output_name);
		void LinkBegin(const char* name);
		void LinkEnd(const char* name);

		void CreateData(dl_uint32 batch_size, const dl_tensor_shape& data_shape, bool if_train = false);

	public:
		void Forward();
		void Backward();

		void ClearState();

		Tensor* GetInputData() { return m_begin_linker->m_data; }
		Tensor* GetOutputData() { return m_end_linker->m_input_linker->m_data; }
		Tensor* GetOutputGradData() { return m_end_linker->m_grad_data; }


	public:
		void Optimize(class Optimizer* opti);

	public:
		void Deserialize(const rapidjson::Document& doc);
		void Serialize(rapidjson::PrettyWriter<rapidjson::StringBuffer>& writer);
			
		void SerializeData(TensorWriter& writer);
		void DeserializeData(TensorReader& reader);
	private:

		ComputeType m_type;

		//Currently I assume only one linker for input and output
		Linker* m_begin_linker{ nullptr };
		Linker* m_end_linker{ nullptr };

		dl_unordered_map<dl_string, Linker*> m_linkers;
		//for serialization, keep order.
		dl_vector<dl_string> m_linker_names;
	};

}

#endif