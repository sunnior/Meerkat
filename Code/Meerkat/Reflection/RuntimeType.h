#ifndef __MEERKAT_RUNTIME_TYPE_H__
#define __MEERKAT_RUNTIME_TYPE_H__

#include "Common/Platform.h"

namespace DeepLearning
{
	struct RuntimeTypeBase
	{
		const char* m_type_name{ nullptr };
		RuntimeTypeBase* m_next{ nullptr };

		RuntimeTypeBase(const char* type_name = nullptr);

		virtual void* CreateType() const { return nullptr; }
	};

	template<typename T>
	struct RuntimeType
	{
	};

	template<typename T>
	RuntimeTypeBase*& GetRuntimeType(T* p)
	{
		static RuntimeTypeBase* s_type;
		return s_type;
	}

	//currently only for layers.
#define DL_REFL_DECLARE(class_type)                                                                     \
        public:                                                                                         \
        virtual void FromJson(const rapidjson::Value& layer_json);                               \
        virtual void ToJson(rapidjson::PrettyWriter<rapidjson::StringBuffer>& writer);                  \
        virtual const char* GetTypeName() { return GetRuntimeType(this)->m_type_name; }

#define DL_REFL_IMPLEMENT(class_type, name)                                                             \
        template<>                                                                                      \
        struct RuntimeType<class_type> : public RuntimeTypeBase {                                       \
            RuntimeType()                                                                               \
                : RuntimeTypeBase(name)                                                                 \
            {                                                                                           \
                RuntimeTypeBase*& type_base = GetRuntimeType<class_type>(nullptr);                      \
                type_base = this;                                                                       \
            };                                                                                          \
            void* CreateType() const override															\
            {                                                                                           \
                return DL_NEW(class_type);																\
            }                                                                                           \
        };                                                                                              \
        RuntimeType<class_type> RuntimeType_##class_type;                                               \
        FORCE_LINK_THIS_LAYER(class_type);

}
#endif