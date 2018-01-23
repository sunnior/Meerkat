#include "TensorReader.h"
#include "Common/Tensor.h"

namespace DeepLearning
{

	TensorReader::TensorReader(const char* file)
	{
		m_file = fopen(file, "rb");
	}

	TensorReader::~TensorReader()
	{
		fclose(m_file);
	}

	void TensorReader::Read(Tensor* tensor)
	{
		dl_tensor* data = tensor->GetData();
		dl_size size = tensor->GetSize();
		TODO("GPU data");
		_Read(data, size);
	}

	void TensorReader::_Read(dl_tensor* data, dl_size size)
	{
		fread(data, sizeof(dl_tensor), size, m_file);
	}

}