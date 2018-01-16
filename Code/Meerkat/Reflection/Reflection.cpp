#include "Common/Platform.h"
#include "Layers/LinearLayer.h"
#include "Layers/LogSoftMaxLayer.h"
#include "Reflection.h"
#include <cstring>
#include <cstdarg>

namespace DeepLearning
{
	Layer* Reflection::CreateLayer(const char* name, ComputeType type, bool if_train, va_list vl)
	{
		TODO("a real reflection system!");
		Layer* layer = nullptr;
		if (std::strcmp(name, "layer_logsoftmax") == 0)
		{
			layer = DL_NEW(LogSoftMaxLayer)(type);
		}

		else if (std::strcmp(name, "layer_linear") == 0)
		{
			dl_uint32 input_size = va_arg(vl, dl_uint32);
			dl_uint32 output_size = va_arg(vl, dl_uint32);

			layer = DL_NEW(LinearLayer)(type, if_train, input_size, output_size);
		}

		va_end(vl);
		return layer;
	}
}
