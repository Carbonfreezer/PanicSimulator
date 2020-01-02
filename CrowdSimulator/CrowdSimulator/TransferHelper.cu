#include "TransferHelper.h"
#include "TgaReader.h"
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "CudaHelper.h"
#include <cstring>


int   TransferHelper::m_intArea[gGridSizeExternal * gGridSizeExternal];
float TransferHelper::m_floatArea[gGridSizeExternal * gGridSizeExternal];

UnsignedArray TransferHelper::UploadPicture(TgaReader* reader, unsigned char boundaryValue)
{
	assert(reader->GetWidth() == gGridSizeInternal);
	assert(reader->GetHeight() == gGridSizeInternal);

	
	
	unsigned char* internalPixelInformation = reader->GetPixels();

	for(int row = 0; row < gGridSizeExternal; ++row)
		for(int column = 0; column < gGridSizeExternal; ++column)
		{	
			unsigned char destinationValue = boundaryValue;
			if ((row != 0) && (row != gGridSizeExternal - 1) && (column != 0) && (column != gGridSizeExternal - 1))
				destinationValue = internalPixelInformation[3 * ((column - 1) + gGridSizeInternal * (row - 1))];
			m_intArea[column + row * gGridSizeExternal] = destinationValue;
		}

	// Allocate device memory.
	void* memory;
	size_t pitch;
	
	cudaMallocPitch(&memory, &pitch, gGridSizeExternal * 4, gGridSizeExternal);
	cudaMemcpy2D(memory, pitch, m_intArea, 4 * gGridSizeExternal, 4 * gGridSizeExternal, gGridSizeExternal, cudaMemcpyHostToDevice);

	pitch /= 4;

	UnsignedArray result;
	result.m_array = (unsigned int*)memory;
	result.m_stride = pitch;

	return result;
}


FloatArray TransferHelper::UploadPictureAsFloat(TgaReader* reader, float boundaryValue, float minValueMapped,
	float maxValueMapped)
{
	assert(reader->GetWidth() == gGridSizeInternal);
	assert(reader->GetHeight() == gGridSizeInternal);

	unsigned char* internalPixelInformation = reader->GetPixels();

	for (int row = 0; row < gGridSizeExternal; ++row)
		for (int column = 0; column < gGridSizeExternal; ++column)
		{
			float destinationValue = boundaryValue;
			if ((row != 0) && (row != gGridSizeExternal - 1) && (column != 0) && (column != gGridSizeExternal - 1))
			{
				destinationValue = internalPixelInformation[3 * ((column - 1) + gGridSizeInternal * (row - 1))];
				destinationValue = minValueMapped + (maxValueMapped - minValueMapped) * destinationValue / 255.0f;
			}
			m_floatArea[column + row * gGridSizeExternal] = destinationValue;
		}

	// Allocate device memory.
	void* memory;
	size_t pitch;

	cudaMallocPitch(&memory, &pitch, gGridSizeExternal * 4, gGridSizeExternal);
	cudaMemcpy2D(memory, pitch, m_floatArea, 4 * gGridSizeExternal, 4 * gGridSizeExternal, gGridSizeExternal, cudaMemcpyHostToDevice);

	pitch /= 4;
	FloatArray result;
	result.m_array = (float*)memory;
	result.m_stride = pitch;

	return result;
}

FloatArray TransferHelper::ReserveFloatMemory()
{
	// Allocate device memory.
	void* memory;
	size_t pitch;
	cudaMallocPitch(&memory, &pitch, gGridSizeExternal * 4, gGridSizeExternal);

	// We fill all with zero at the beginnig.
	memset(m_floatArea, 0, gGridSizeExternal * gGridSizeExternal * 4);
	cudaMemcpy2D(memory, pitch, m_floatArea, 4 * gGridSizeExternal, 4 * gGridSizeExternal, gGridSizeExternal, cudaMemcpyHostToDevice);
	
	pitch /= 4;

	FloatArray result;
	result.m_array = (float*)memory;
	result.m_stride = pitch;
	
	return result;
}

UnsignedArray TransferHelper::ReserveUnsignedMemory()
{
	// Allocate device memory.
	void* memory;
	size_t pitch;
	cudaMallocPitch(&memory, &pitch, gGridSizeExternal * 4, gGridSizeExternal);

	// We fill all with zero at the beginnig.
	memset(m_intArea, 0, gGridSizeExternal * gGridSizeExternal * 4);
	cudaMemcpy2D(memory, pitch, m_intArea, 4 * gGridSizeExternal, 4 * gGridSizeExternal, gGridSizeExternal, cudaMemcpyHostToDevice);

	pitch /= 4;

	UnsignedArray result;
	result.m_array = (unsigned int*)memory;
	result.m_stride = pitch;

	return result;
}

FloatArray TransferHelper::UpfronFilledValue(float value)
{
	for (int row = 0; row < gGridSizeExternal; ++row)
		for (int column = 0; column < gGridSizeExternal; ++column)
		{
			m_floatArea[column + row * gGridSizeExternal] = value;
		}

	// Allocate device memory.
	void* memory;
	size_t pitch;

	cudaMallocPitch(&memory, &pitch, gGridSizeExternal * 4, gGridSizeExternal);
	cudaMemcpy2D(memory, pitch, m_floatArea, 4 * gGridSizeExternal, 4 * gGridSizeExternal, gGridSizeExternal, cudaMemcpyHostToDevice);

	pitch /= 4;
	FloatArray result;
	result.m_array = (float*)memory;
	result.m_stride = pitch;

	return result;
}

FloatArray TransferHelper::BuildHorizontalGradient(float startMax, int direction)
{
	for (int row = 0; row < gGridSizeExternal; ++row)
		for (int column = 0; column < gGridSizeExternal; ++column)
		{
			float destinationValue;
			if (direction == 1)
				destinationValue = startMax * ((float)column) / (gGridSizeExternal - 1);
			else
				destinationValue = startMax - startMax * ((float)column) / (gGridSizeExternal - 1);
			m_floatArea[column + row * gGridSizeExternal] = destinationValue;
		}

	// Allocate device memory.
	void* memory;
	size_t pitch;

	cudaMallocPitch(&memory, &pitch, gGridSizeExternal * 4, gGridSizeExternal);
	cudaMemcpy2D(memory, pitch, m_floatArea, 4 * gGridSizeExternal, 4 * gGridSizeExternal, gGridSizeExternal, cudaMemcpyHostToDevice);

	pitch /= 4;
	FloatArray result;
	result.m_array = (float*)memory;
	result.m_stride = pitch;

	return result;
}

FloatArray TransferHelper::BuildVerticalGradient(float startMax, int direction)
{
	for (int row = 0; row < gGridSizeExternal; ++row)
		for (int column = 0; column < gGridSizeExternal; ++column)
		{
			float destinationValue;
			if (direction == 1)
				destinationValue = startMax * ((float)row) / (gGridSizeExternal - 1);
			else
				destinationValue = startMax - startMax * ((float)row) / (gGridSizeExternal - 1);
			m_floatArea[column + row * gGridSizeExternal] = destinationValue;
		}

	// Allocate device memory.
	void* memory;
	size_t pitch;

	cudaMallocPitch(&memory, &pitch, gGridSizeExternal * 4, gGridSizeExternal);
	cudaMemcpy2D(memory, pitch, m_floatArea, 4 * gGridSizeExternal, 4 * gGridSizeExternal, gGridSizeExternal, cudaMemcpyHostToDevice);

	pitch /= 4;
	FloatArray result;
	result.m_array = (float*)memory;
	result.m_stride = pitch;

	return result;
}

FloatArray TransferHelper::BuildRadialGradient(float startMax, int direction)
{

	float maxDistance = sqrtf(2) * gGridSizeExternal / 2.0f;
	for (int row = 0; row < gGridSizeExternal; ++row)
		for (int column = 0; column < gGridSizeExternal; ++column)
		{
			float distance = sqrtf((float)((row - gGridSizeExternal / 2) * (row - gGridSizeExternal / 2) + (column - gGridSizeExternal / 2) * (column - gGridSizeExternal / 2)));
			distance /= maxDistance;
			if (direction == 1)
				distance = 1.0f - distance;

			m_floatArea[column + row * gGridSizeExternal] = startMax * distance;
		}

	// Allocate device memory.
	void* memory;
	size_t pitch;

	cudaMallocPitch(&memory, &pitch, gGridSizeExternal * 4, gGridSizeExternal);
	cudaMemcpy2D(memory, pitch, m_floatArea, 4 * gGridSizeExternal, 4 * gGridSizeExternal, gGridSizeExternal, cudaMemcpyHostToDevice);

	pitch /= 4;
	FloatArray result;
	result.m_array = (float*)memory;
	result.m_stride = pitch;

	return result;
}











__global__ void CopyData(float* sourceArray, size_t sourceStride, float* destinationArray, size_t destinationStride)
{
	int baseX = (threadIdx.x + blockIdx.x * blockDim.x) + 1;
	int baseY = (threadIdx.y + blockIdx.y * blockDim.y) + 1;

	destinationArray[baseX + baseY * destinationStride] = sourceArray[baseX + baseY * sourceStride];

	// Here we have to deal with the boundaries.
	if (baseX == 1)
	{
		destinationArray[baseY * destinationStride] = sourceArray[ baseY * sourceStride];
	}
	if (baseY == 1)
	{
		destinationArray[baseX ] = sourceArray[baseX ];
	}
	if (baseX == gGridSizeExternal - 2)
	{
		destinationArray[(gGridSizeExternal - 1) + baseY * destinationStride] = sourceArray[(gGridSizeExternal - 1) + baseY * sourceStride];
	}
	if (baseY == gGridSizeExternal - 2)
	{
		destinationArray[baseX + (gGridSizeExternal - 1) * destinationStride] = sourceArray[baseX + (gGridSizeExternal - 1) * sourceStride];
	}

	// The 4 corner cases.
	if ((baseX == 1) && (baseY == 1))
	{
		destinationArray[0] = sourceArray[0];
		destinationArray[gGridSizeExternal - 1] = sourceArray[gGridSizeExternal - 1];
		destinationArray[(gGridSizeExternal - 1) * destinationStride] = sourceArray[(gGridSizeExternal - 1) * sourceStride];
		destinationArray[(gGridSizeExternal - 1) + (gGridSizeExternal - 1) * destinationStride] = sourceArray[(gGridSizeExternal - 1) + (gGridSizeExternal - 1) * sourceStride];
	}
}

void TransferHelper::CopyDataFromTo(FloatArray source, FloatArray destination)
{
	assert(source.m_array);
	assert(destination.m_array);
	CopyData CUDA_DECORATOR_LOGIC (source.m_array, source.m_stride, destination.m_array, destination.m_stride);
}
