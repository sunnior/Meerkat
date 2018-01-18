#include "Meerkat.h"
#include "Blas.h"
#include "Reflection/Reflection.h"

namespace DeepLearning
{
	void Initialize()
	{
		InitBlas();
		ReflectionManager::Initialize();
	}

	void Finalize()
	{
		ReflectionManager::Finalize();
	}
}
