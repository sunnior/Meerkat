#include "TensorWriter.h"
#include "Common/Tensor.h"

namespace DeepLearning
{

	TensorWriter::TensorWriter(const char* file)
	{
		m_file = fopen(file, "wb");
	}

	TensorWriter::~TensorWriter()
	{
		fclose(m_file);
	}

	void TensorWriter::Write(Tensor* tensor)
	{
		dl_tensor* data = tensor->GetData();
		dl_size size = tensor->GetSize();
		TODO("GPU data");
		_Write(data, size);
	}

	void TensorWriter::_Write(dl_tensor* data, dl_size size)
	{
		fwrite(data, sizeof(dl_tensor), size, m_file);
	}

}
