#include "MnistDB.h"
#include "Common/Meerkat.h"
#include "Optimizer/SgdOptimizer.h"
#include "Criterions/ClassNllCriterion.h"
#include "Model/Model.h"

using namespace DeepLearning;

int main()
{
	const unsigned int batch_size = 64;
	const char* train_image_file_path = "train-images.idx3-ubyte";
	const char* train_label_file_path = "train-labels.idx1-ubyte";

	DeepLearning::Initialize();

	FILE* pJsonFile = fopen("model.json", "rb");
	fseek(pJsonFile, 0, SEEK_END);
	size_t file_size = ftell(pJsonFile);
	char* json_buffer = (char*)malloc(file_size + 1);
	fseek(pJsonFile, 0, SEEK_SET);
	fread(json_buffer, 1, file_size, pJsonFile);
	json_buffer[file_size] = 0;
	rapidjson::Document doc;
	doc.Parse(json_buffer);
	free(json_buffer);

	MnistDB train_db(train_image_file_path, train_label_file_path);

	unsigned int rows, columns;
	unsigned int labels;
	
	train_db.GetDimension(rows, columns, labels);

	Tensor* label = DL_NEW(Tensor)(ComputeType_CPU, { batch_size, 1 });

	Model* model = DL_NEW(Model)(ComputeType_CPU);
	model->Deserialize(doc);
	rapidjson::StringBuffer sb;
	rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(sb);
	
	model->Serialize(writer);
	const char* json = sb.GetString();

	model->CreateData(batch_size, { rows*columns }, true);
	Tensor* data = model->GetInputData();

	Optimizer* optimizer = DL_NEW(SgdOptimizer)(ComputeType_CPU);
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

		model->Optimize(optimizer);
	}

	DL_SAFE_DELETE(label);
	DL_SAFE_DELETE(model);
	DL_SAFE_DELETE(optimizer);
	DL_SAFE_DELETE(criterion);

	DeepLearning::Finalize();

	return 0;
}