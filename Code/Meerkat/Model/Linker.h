#ifndef __MEERKAT_LINKER_H__
#define __MEERKAT_LINKER_H__

#include "Util/dl_stl.h"

namespace DeepLearning
{
	class Layer;
	class Tensor;

	class Linker
	{
		friend class Model;
	public:
		Linker(ComputeType type, Layer* layer = nullptr);
		~Linker();

		void ForwardRecurrent();
		void BackwardRecurrent();

		void ClearState();
		bool IsForwardReady() { return m_is_forward_ready; }
		void SetForwardReady(bool flag) { m_is_forward_ready = flag; }
		bool IsBackwardReady() { return m_is_backward_ready; }
		void SetBackwardReady(bool flag) { m_is_backward_ready = flag; }

		void CreateData(dl_uint32 batch_size, const dl_tensor_shape& data_shape);
		void CreateDataRecurrent(dl_uint32 batch_size, bool if_train);

		void Optimize(class Optimizer* opti);

	private:

		ComputeType m_type;
		dl_string m_name;
		Layer* m_layer{ nullptr };
		bool   m_is_forward_ready{ false };
		bool   m_is_backward_ready{ false };
		Linker* m_input_linker{ nullptr };
		Linker* m_output_linker{ nullptr };
		Tensor* m_data{ nullptr };
		Tensor* m_grad_data{ nullptr };
	};
}
#endif