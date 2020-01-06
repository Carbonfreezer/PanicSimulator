#include "VisualizationHelper.h"
#include <vector_functions.hpp>
#include <cassert>
#include "TransferHelper.h"
#include <math.h>
#include <device_launch_parameters.h>
#include "CudaHelper.h"



FloatArray VisualizationHelper::m_isoLineData;

__global__  void MarcIfFlagged(unsigned int* deviceMemory, size_t devicePitch, uchar4* pixelMemory, uchar4 color)
{
	int baseX = (threadIdx.x + blockIdx.x * blockDim.x) * gPixelsPerCell;
	int baseY = (threadIdx.y + blockIdx.y * blockDim.y) * gPixelsPerCell;
	for (int i = 0; i < gPixelsPerCell; ++i)
		for (int j = 0; j < gPixelsPerCell; ++j)
		{
			int srcX = i + baseX;
			int srcY = j + baseY;


			unsigned int candidate = deviceMemory[(srcX / gPixelsPerCell + 1) + devicePitch * (srcY / gPixelsPerCell + 1
				)];
			if (candidate)
				pixelMemory[srcX + gScreenResolution * srcY] = color;


		}
}

void VisualizationHelper::MarcColor(UnsignedArray data, uchar4 color, uchar4* pixelMemory)
{
	assert(data.m_array);
	MarcIfFlagged CUDA_DECORATOR_LOGIC(data.m_array, data.m_stride, pixelMemory, color);
}



__global__ void GenerateLineFlags(float* dataMemory, size_t dataStride, unsigned int* isoLineFlags,
                                  size_t isoLineStride, float isoLineStepSize)
{
	__shared__ float valueBuffer[gBlockSize + 2][gBlockSize + 2];


	// We keep tack of the pixel block we are responsible for.
	int xOrigin = threadIdx.x + gBlockSize * blockIdx.x + 1;
	int yOrigin = threadIdx.y + gBlockSize * blockIdx.y + 1;

	int xScan = threadIdx.x + 1;
	int yScan = threadIdx.y + 1;


	valueBuffer[xScan][yScan] = dataMemory[xOrigin + yOrigin * dataStride];

	if (threadIdx.x == 0)
		valueBuffer[xScan - 1][yScan] = dataMemory[(xOrigin - 1) + yOrigin * dataStride];
	if (threadIdx.x == 31)
		valueBuffer[xScan + 1][yScan] = dataMemory[(xOrigin + 1) + yOrigin * dataStride];
	if (threadIdx.y == 0)
		valueBuffer[xScan][yScan - 1] = dataMemory[xOrigin + (yOrigin - 1) * dataStride];
	if (threadIdx.y == 31)
		valueBuffer[xScan][yScan + 1] = dataMemory[xOrigin + (yOrigin + 1) * dataStride];

	__syncthreads();


	// Get nearest iso value.
	float localValue = valueBuffer[xScan][yScan];
	float nearestIsoValue = roundf(localValue / isoLineStepSize) * isoLineStepSize;

	bool linePlotting = false;

	linePlotting |= (valueBuffer[xScan - 1][yScan] > nearestIsoValue);
	linePlotting |= (valueBuffer[xScan + 1][yScan] > nearestIsoValue);
	linePlotting |= (valueBuffer[xScan][yScan + 1] > nearestIsoValue);
	linePlotting |= (valueBuffer[xScan][yScan - 1] > nearestIsoValue);

	linePlotting &= (localValue <= nearestIsoValue);

	isoLineFlags[xOrigin + yOrigin * isoLineStride] = linePlotting;


}



void VisualizationHelper::VisualizeIsoLines(FloatArray deviceData, float isoLineStepSize,
                                            uchar4* pixelMemory)
{

	assert(deviceData.m_array);

	if (m_isoLineData.m_array == NULL)
		m_isoLineData = TransferHelper::ReserveFloatMemory();

		
	GenerateLineFlags CUDA_DECORATOR_LOGIC(deviceData.m_array, deviceData.m_stride,
	                                       (unsigned int *)(m_isoLineData.m_array), m_isoLineData.m_stride,
	                                       isoLineStepSize);
	MarcIfFlagged CUDA_DECORATOR_LOGIC((unsigned int *)(m_isoLineData.m_array), m_isoLineData.m_stride, pixelMemory,
	                                   make_uchar4(128, 128, 128, 255));
}


__global__ void  VisualizeField(float* deviceMemory, size_t devicePitch, float maximumValue, uchar4* pixelMemory)
{
	int baseX = (threadIdx.x + blockIdx.x * blockDim.x);
	int baseY = (threadIdx.y + blockIdx.y * blockDim.y);


	float candidate = deviceMemory[(baseX + 1) + devicePitch * (baseY + 1)];

	unsigned char redColor = (unsigned char)(255.0f * fminf(maximumValue, candidate) / maximumValue);
	uchar4 finalColor = make_uchar4(redColor, 0, 255 - redColor, 255);

	baseX *= gPixelsPerCell;
	baseY *= gPixelsPerCell;

	for (int i = 0; i < gPixelsPerCell; ++i)
		for (int j = 0; j < gPixelsPerCell; ++j)
			pixelMemory[i + baseX + gScreenResolution * (j + baseY)] = finalColor;
}

void VisualizationHelper::VisualizeScalarField(FloatArray deviceData, float maximumValue,
	uchar4* pixelMemory)
{
	assert(deviceData.m_array);
	VisualizeField CUDA_DECORATOR_LOGIC(deviceData.m_array, deviceData.m_stride, maximumValue, pixelMemory);
}


__global__ void  VisualizeFieldFromTo(float* deviceMemory, size_t devicePitch, float minimumValue, float maximumValue, uchar4* pixelMemory)
{
	int baseX = (threadIdx.x + blockIdx.x * blockDim.x);
	int baseY = (threadIdx.y + blockIdx.y * blockDim.y);


	float candidate = deviceMemory[(baseX + 1) + devicePitch * (baseY + 1)];
	candidate = fmaxf(minimumValue, candidate);
	candidate = fminf(maximumValue, candidate);
	candidate -= minimumValue;
	

	unsigned char redColor = (unsigned char)(255.0f * candidate / (maximumValue - minimumValue ));
	uchar4 finalColor = make_uchar4(redColor, 0, 255 - redColor, 255);

	baseX *= gPixelsPerCell;
	baseY *= gPixelsPerCell;

	for (int i = 0; i < gPixelsPerCell; ++i)
		for (int j = 0; j < gPixelsPerCell; ++j)
			pixelMemory[i + baseX + gScreenResolution * (j + baseY)] = finalColor;
}


void VisualizationHelper::VisualizeScalarField(FloatArray deviceData, float minimumValue, float maximumValue,
	uchar4* pixelMemory)
{
	assert(deviceData.m_array);
	VisualizeFieldFromTo CUDA_DECORATOR_LOGIC(deviceData.m_array, deviceData.m_stride, minimumValue, maximumValue, pixelMemory);
}

__global__ void  VisualizeFieldWithNegative(float* deviceMemory, size_t devicePitch, float maximumValue,
                                            uchar4* pixelMemory)
{
	int baseX = (threadIdx.x + blockIdx.x * blockDim.x);
	int baseY = (threadIdx.y + blockIdx.y * blockDim.y);


	float candidate = deviceMemory[(baseX + 1) + devicePitch * (baseY + 1)];
	candidate = fminf(candidate, maximumValue);
	candidate = fmaxf(candidate, -maximumValue);
	candidate = (candidate + maximumValue) / (2.0f * maximumValue);

	unsigned char redColor = (unsigned char)(255.0f * candidate);
	uchar4 finalColor = make_uchar4(redColor, 0, 255 - redColor, 255);

	baseX *= gPixelsPerCell;
	baseY *= gPixelsPerCell;

	for (int i = 0; i < gPixelsPerCell; ++i)
		for (int j = 0; j < gPixelsPerCell; ++j)
			pixelMemory[i + baseX + gScreenResolution * (j + baseY)] = finalColor;
}

void VisualizationHelper::VisualizeScalarFieldWithNegative(FloatArray deviceData, float maximumValue,
                                                           uchar4* pixelMemory)
{
	assert(deviceData.m_array);
	VisualizeFieldWithNegative CUDA_DECORATOR_LOGIC(deviceData.m_array, deviceData.m_stride, maximumValue, pixelMemory);
}

