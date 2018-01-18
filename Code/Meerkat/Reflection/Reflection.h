#ifndef __MEERKAT_REFLECTION_H__
#define __MEERKAT_REFLECTION_H__

#include "Util/dl_stl.h"
#include "Util/rapidjson/document.h"
#include "Util/rapidjson/prettywriter.h"
#include "RuntimeType.h"


namespace DeepLearning
{
	class Layer;

	class ReflectionManager
	{
	public:
		static void Initialize();
		static void Finalize();
		static const ReflectionManager* GetInstance() { return s_instance; }

		static void AddRawType(RuntimeTypeBase* type);
	private:
		static ReflectionManager* s_instance;
		static RuntimeTypeBase s_raw_header;

	public:
		ReflectionManager();
		~ReflectionManager();

		Layer* CreateLayer(const char* type_name) const;

	private:
		const RuntimeTypeBase& _FindConstructor(const char* type_name) const;

	private:
		dl_unordered_map<dl_string, const RuntimeTypeBase*> m_constructors;
	};
}

#endif
