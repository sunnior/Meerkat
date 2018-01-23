#ifndef __MEERKAT_TENSOR_WRITER_H__
 #define __MEERKAT_TENSOR_WRITER_H__

#include <cstdio>
#include "Common/Platform.h"

namespace DeepLearning
{
	class Tensor;

	class TensorWriter
	{
	public:
		TensorWriter(const char* file);
		~TensorWriter();

		void Write(Tensor* tensor);

	private:
		void _Write(dl_tensor* data, dl_size size);

	private:
		FILE* m_file;

	};
}

#endif