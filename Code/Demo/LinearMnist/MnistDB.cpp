#include "MnistDB.h"
#include <stdlib.h>

using namespace DeepLearning;

static void swap_4byte(dl_uint8* buffer)
{
	dl_uint32 tmp = *(dl_uint32*)buffer;
	*(buffer + 0) = *((dl_uint8*)&tmp + 3);
	*(buffer + 1) = *((dl_uint8*)&tmp + 2);
	*(buffer + 2) = *((dl_uint8*)&tmp + 1);
	*(buffer + 3) = *((dl_uint8*)&tmp + 0);
}



MnistDB::MnistDB(const char* image_file_path, const char* label_file_path)
{
	m_pImageFile = fopen(image_file_path, "rb");
	m_pLabelFile = fopen(label_file_path, "rb");

	dl_uint8 image_header[s_image_header_size];

	fread(image_header, 1, s_image_header_size, m_pImageFile);
	swap_4byte(image_header + 4);
	swap_4byte(image_header + 8);
	swap_4byte(image_header + 12);

	m_size = *((dl_uint32*)image_header + 1);
	m_rows = *((dl_uint32*)image_header + 2);
	m_columns = *((dl_uint32*)image_header + 3);

	dl_uint8 label_header[s_label_header_size];

	fread(label_header, 1, s_label_header_size, m_pLabelFile);

	m_idx = 0;
}



void MnistDB::Reset()
{
	m_idx = 0;

	fseek(m_pImageFile, s_image_header_size, SEEK_SET);
	fseek(m_pLabelFile, s_label_header_size, SEEK_SET);
}



void MnistDB::GetDimension(unsigned int& rows, unsigned int& columns, unsigned int& labels)
{
	labels = 10;
	rows = m_rows;
	columns = m_columns;
}

void MnistDB::LoadData(Tensor* tensor_data, Tensor* tensor_label, const unsigned int batch_size)
{
	if (m_size - m_idx < batch_size)
	{
		Reset();
	}

	m_idx = batch_size;

	dl_uint32 size = batch_size*m_rows*m_columns;
	dl_uint8* buffer = (dl_uint8*)malloc(size);
	dl_tensor* data = (dl_tensor*)malloc(size * sizeof(dl_tensor));
	fread(buffer, 1, size, m_pImageFile);

	for (dl_uint32 i = 0; i < size; ++i)
	{
		*(data  + i) = (dl_tensor)*(buffer + i) / 255.0f;
	}

	/*
		 for (int k = 0; k < batch_size; k)
		 {
			 for (dl_uint32 i = 0; i < 28; i)
			 {
				 for (dl_uint32 j = 0; j < 28; j)
				 {
					 if (*(data  28*28*k  28 * i  j) > 0)
					 {
						 printf("X");
					 }
					 else
					 {
						 printf("_");
					 }
				 }
				 printf("\n");
			 }
		 }*/

	tensor_data->LoadData(data);
	free(buffer);
	free(data);

	dl_uint8* label_buffer = (dl_uint8*)malloc(batch_size);
	dl_tensor* label_data = (dl_tensor*)malloc(batch_size * sizeof(dl_tensor));
	fread(label_buffer, 1, batch_size, m_pLabelFile);

	for (dl_uint32 i = 0; i < batch_size; ++i)
	{
		*(label_data + i) = *(label_buffer + i);
	}

	tensor_label->LoadData(label_data);
	free(label_buffer);
	free(label_data);
}