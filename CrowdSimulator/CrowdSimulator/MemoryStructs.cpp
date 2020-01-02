#include "MemoryStructs.h"
#include <cstddef>
#include <cuda_runtime_api.h>

FloatArray::FloatArray()
{
	m_array = NULL;
	m_stride = 0;
}

void FloatArray::FreeArray()
{
	if (m_array != NULL)
		cudaFree(m_array);
	m_array = NULL;
}



UnsignedArray::UnsignedArray()
{
	m_array = NULL;
	m_stride = 0;
}

void UnsignedArray::FreeArray()
{
	if (m_array != NULL)
		cudaFree(m_array);
	m_array = NULL;
}
