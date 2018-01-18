#include "Common/Platform.h"
#include "Layers/LinearLayer.h"
#include "Layers/LogSoftMaxLayer.h"
#include "Reflection.h"
#include <cstring>
#include <cstdarg>

namespace DeepLearning
{
	RuntimeTypeBase::RuntimeTypeBase(const char* type_name)
		: m_type_name(type_name)
	{
		ReflectionManager::AddRawType(this);
	}

	ReflectionManager* ReflectionManager::s_instance = nullptr;
	RuntimeTypeBase ReflectionManager::s_raw_header;

	static void SupidForceLink()
	{
		FORCE_LINK_THAT_LAYER(LinearLayer);
		FORCE_LINK_THAT_LAYER(LogSoftMaxLayer);
	}

	void ReflectionManager::Initialize()
	{
		s_instance = DL_NEW(ReflectionManager);
	}

	void ReflectionManager::Finalize()
	{
		DL_SAFE_DELETE(s_instance);
	}

	void ReflectionManager::AddRawType(RuntimeTypeBase* type)
	{
		type->m_next = s_raw_header.m_next;
		s_raw_header.m_next = type;
	}

	ReflectionManager::ReflectionManager()
	{
		SupidForceLink();

		RuntimeTypeBase* base = s_raw_header.m_next;
		while (base != &s_raw_header)
		{
			m_constructors.insert(::std::pair<dl_string, const RuntimeTypeBase*>(base->m_type_name, base));
			base = base->m_next;
		}
	}

	ReflectionManager::~ReflectionManager()
	{

	}

	Layer* ReflectionManager::CreateLayer(const char* type_name) const
	{
		return static_cast<Layer*>(_FindConstructor(type_name).CreateType());
	}

	const RuntimeTypeBase& ReflectionManager::_FindConstructor(const char* type_name) const
	{
		return *(m_constructors.find(type_name)->second);
	}


}
