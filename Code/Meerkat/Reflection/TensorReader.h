#ifndef __MEERKAT_TENSOR_READER_H__
 #define __MEERKAT_TENSOR_READER_H__

#include <cstdio>
 #include "Common/Platform.h"

namespace DeepLearning
{
	class Tensor;

	class TensorReader
	{
	public:
		TensorReader(const char* file);
		~TensorReader();

		void Read(Tensor* tensor);

	private:
		void _Read(dl_tensor* data, dl_size size);

	private:
		FILE* m_file;

	};
}

#endif
