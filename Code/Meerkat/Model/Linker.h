#ifndef __MEERKAT_LINKER_H__
#define __MEERKAT_LINKER_H__

#include "Util/dl_stl.h"

namespace DeepLearning
{
	class Layer;
	class Tensor;

	class Linker
	{
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

		void SetInput(Linker* linker) { m_input_linker = linker; }
		Linker* GetInput() { return m_input_linker; }

		void SetOutput(Linker* linker) { m_output_linker = linker; }
		Linker* GetOutput() { return m_output_linker; }

		void CreateData(dl_uint32 batch_size, const dl_tensor_shape& data_shape);
		void CreateDataRecurrent(dl_uint32 batch_size);

		Tensor* GetData() { return m_data; }
		Tensor* GetGradData() { return m_grad_data; }

		void GetLearnableParam(const dl_vector<Tensor*>*& params, const dl_vector<Tensor*>*& param_grads);
	private:

		ComputeType m_type;
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