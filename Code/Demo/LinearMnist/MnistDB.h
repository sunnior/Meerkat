#ifndef __DEMO_MNISTDB_H__
#define __DEMO_MNISTDB_H__

#include <stdio.h>
#include "Common/Tensor.h"

class MnistDB
{
public:
	MnistDB(const char* image_file_path, const char* label_file_path);

	void GetDimension(unsigned int& rows, unsigned int& columns, unsigned int& labels);
	void LoadData(DeepLearning::Tensor* data, DeepLearning::Tensor* label, const unsigned int batch_size);
	void Reset();

private:
	unsigned int m_rows;
	unsigned int m_columns;
	unsigned int m_size;
	unsigned int m_idx;

	FILE* m_pImageFile;
	FILE* m_pLabelFile;

	static const int s_image_header_size = 16;
	static const int s_label_header_size = 8;
};



#endif