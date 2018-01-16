#ifndef __MEERKAT_REFLECTION_H__
#define __MEERKAT_REFLECTION_H__

#include <vector>
#include <cstdarg>
#include "Common/Platform.h"

namespace DeepLearning
{
	class Layer;

	class Reflection
	{
	public:
		static Layer* CreateLayer(const char* name, ComputeType type, bool if_train, va_list vl);
	};
}

#endif
