#ifndef __MEERKAT_MODEL_H__
#define __MEERKAT_MODEL_H__

namespace DeepLearning
{
	class Layer;

	class Model
	{
	public:
		void Link(Layer* input_layer, Layer* output_layer);
		void LinkBegin(Layer* layer);
		void LinkEnd(Layer* layer);
		void Forward();
	};

}
#endif