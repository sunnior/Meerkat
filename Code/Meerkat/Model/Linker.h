#ifndef __MEERKAT_LINKER_H__
#define __MEERKAT_LINKER_H__

namespace DeepLearning
{
	class Layer;

	class Linker
	{
	public:
		Linker(Layer* input_layer, Layer* output_layer);

	};

}
#endif