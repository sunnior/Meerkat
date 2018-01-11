#include "MnistDB.h"
#include "Common/Meerkat.h"
#include "Layers/LinearLayer.h"
#include "Layers/LogSoftMaxLayer.h"
#include "Optimizer/SgdOptimizer.h"
#include "Criterions/ClassNllCriterion.h"
#include "Model/Model.h"

using namespace DeepLearning;

int main()
{
	const unsigned int batch_size = 64;
	const char* train_image_file_path = "train-images.idx3-ubyte";
	const char* train_label_file_path = "train-labels.idx1-ubyte";

	DeepLearning::Init();

	MnistDB train_db(train_image_file_path, train_label_file_path);

	unsigned int rows, columns;
	unsigned int labels;
	
	train_db.GetDimension(rows, columns, labels);

	Tensor* data = DL_NEW(Tensor)(ComputeType_CPU, { batch_size, rows*columns });
	Tensor* label = DL_NEW(Tensor)(ComputeType_CPU, { batch_size, 1 });
	Tensor* output1 = DL_NEW(Tensor)(ComputeType_CPU, { batch_size, labels });
	Tensor* output2 = DL_NEW(Tensor)(ComputeType_CPU, { batch_size, labels });

	Tensor* grad_output1 = DL_NEW(Tensor)(ComputeType_CPU, { batch_size, labels });
	Tensor* grad_output2 = DL_NEW(Tensor)(ComputeType_CPU, { batch_size, labels });

	LinearLayer* linear_layer = DL_NEW(LinearLayer)(ComputeType_CPU, true, rows*columns, labels);
	LogSoftMaxLayer* logsoftmax_layer = DL_NEW(LogSoftMaxLayer)(ComputeType_CPU);

	dl_uint32 count;
	Tensor** params;
	Tensor** grad_params;
	linear_layer->GetLearnableParam(params, grad_params, count);

	Optimizer* optimizer = DL_NEW(SgdOptimizer)(ComputeType_CPU, params, grad_params, count);
	ClassNllCriterion* criterion = DL_NEW(ClassNllCriterion)(ComputeType_CPU);

	Model* model = DL_NEW(Model);

	model->LinkBegin(linear_layer);
	model->Link(linear_layer, logsoftmax_layer);
	model->LinkEnd(logsoftmax_layer);

	train_db.LoadData(data, label, batch_size);

	while (true)
	{
		//linear_layer->Forward(data, output1);
		//logsoftmax_layer->Forward(output1, output2);
		model->Forward();

		dl_tensor loss;
		criterion->Forward(output2, label, &loss);

		criterion->Backward(output2, label, grad_output2);
		logsoftmax_layer->Backward(output1, grad_output2, grad_output1);
		linear_layer->Backward(data, grad_output1, nullptr);

		optimizer->Update();
	}

	return 0;
}