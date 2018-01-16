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

	Tensor* label = DL_NEW(Tensor)(ComputeType_CPU, { batch_size, 1 });

	Model* model = DL_NEW(Model)(ComputeType_CPU, true);
	model->CreateLayer("layer_linear", "linear", rows*columns, labels);
	model->CreateLayer("layer_logsoftmax", "logsoftmax");
	model->LinkBegin("linear");
	model->LinkEnd("logsoftmax");
	model->Link("linear", "logsoftmax");

	model->CreateData(batch_size, { rows*columns });
	Tensor* data = model->GetInputData();

	dl_vector<Tensor*> params;
	dl_vector<Tensor*> grad_params;
	model->GetLearnableParam(params, grad_params);

	Optimizer* optimizer = DL_NEW(SgdOptimizer)(ComputeType_CPU, params, grad_params);
	ClassNllCriterion* criterion = DL_NEW(ClassNllCriterion)(ComputeType_CPU);

	train_db.LoadData(data, label, batch_size);

	while (true)
	{
		model->ClearState();
		model->Forward();

		dl_tensor loss;
		criterion->Forward(model->GetOutputData(), label, &loss);

		criterion->Backward(model->GetOutputData(), label, model->GetOutputGradData());

		model->Backward();

		optimizer->Update();
	}

	DL_SAFE_DELETE(label);
	DL_SAFE_DELETE(model);
	DL_SAFE_DELETE(optimizer);
	DL_SAFE_DELETE(criterion);

	return 0;
}