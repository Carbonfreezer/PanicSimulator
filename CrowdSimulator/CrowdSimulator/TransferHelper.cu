#include "TransferHelper.h"
#include "TgaReader.h"
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "CudaHelper.h"
#include <cstring>


unsigned int* TransferHelper::UploadPicture(TgaReader* reader, unsigned char boundaryValue, size_t& pitch)
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
	void* result;
	
	cudaMallocPitch(&result, &pitch, gGridSizeExternal * 4, gGridSizeExternal);
	cudaMemcpy2D(result, pitch, m_intArea, 4 * gGridSizeExternal, 4 * gGridSizeExternal, gGridSizeExternal, cudaMemcpyHostToDevice);

	pitch /= 4;
	

	return (unsigned int*)result;
}


float* TransferHelper::UploadPictureAsFloat(TgaReader* reader, float boundaryValue, float minValueMapped,
	float maxValueMapped, size_t& pitch)
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
				destinationValue = minValueMapped + (maxValueMapped - minValueMapped) * destinationValue / 255;
			}
			m_floatArea[column + row * gGridSizeExternal] = destinationValue;
		}

	// Allocate device memory.
	void* result;

	cudaMallocPitch(&result, &pitch, gGridSizeExternal * 4, gGridSizeExternal);
	cudaMemcpy2D(result, pitch, m_floatArea, 4 * gGridSizeExternal, 4 * gGridSizeExternal, gGridSizeExternal, cudaMemcpyHostToDevice);

	pitch /= 4;

	return (float*)result;
}

float* TransferHelper::ReserveFloatMemory(size_t& pitch)
{
	// Allocate device memory.
	void* result;
	cudaMallocPitch(&result, &pitch, gGridSizeExternal * 4, gGridSizeExternal);

	// We fill all with zero at the beginnig.
	void* temp = malloc(4 * gGridSizeExternal * gGridSizeExternal);
	memset(temp, 0, gGridSizeExternal * gGridSizeExternal * 4);
	cudaMemcpy2D(result, pitch, temp, 4 * gGridSizeExternal, 4 * gGridSizeExternal, gGridSizeExternal, cudaMemcpyHostToDevice);
	free(temp);
	

	pitch /= 4;
	return (float*)result;
}


__global__  void MarcIfFlagged(unsigned int* deviceMemory, size_t devicePitch, uchar4* pixelMemory, uchar4 color)
{
	int baseX = (threadIdx.x + blockIdx.x * blockDim.x) * gPixelsPerCell;
	int baseY = (threadIdx.y + blockIdx.y * blockDim.y) * gPixelsPerCell;
	for(int i = 0; i < gPixelsPerCell; ++i)
		for(int j = 0; j < gPixelsPerCell; ++j)
		{
			int srcX = i + baseX;
			int srcY = j + baseY;

			
			unsigned int candidate = deviceMemory[(srcX / gPixelsPerCell + 1) + devicePitch * (srcY / gPixelsPerCell + 1)];
			if (candidate)
				pixelMemory[srcX + gScreenResolution * srcY] = color;
		

		}
}

void TransferHelper::MarcColor(unsigned int* deviceMemory,size_t devicePitch, uchar4* pixelMemory, uchar4 color)
{
	MarcIfFlagged CUDA_DECORATOR_LOGIC (deviceMemory, devicePitch, pixelMemory, color);
}


__global__ void  VisualizeField(float* deviceMemory, size_t devicePitch, float maximumValue, uchar4* pixelMemory)
{
	int baseX = (threadIdx.x + blockIdx.x * blockDim.x) ;
	int baseY = (threadIdx.y + blockIdx.y * blockDim.y) ;


	float candidate = deviceMemory[(baseX + 1) + devicePitch * (baseY + 1)];

	unsigned char redColor  = (unsigned char) (255.0f * fminf(maximumValue, candidate) / maximumValue);
	uchar4 finalColor = make_uchar4(redColor, 0, 255 - redColor, 255);

	baseX *= gPixelsPerCell;
	baseY *= gPixelsPerCell;

	for (int i = 0; i < gPixelsPerCell; ++i)
		for (int j = 0; j < gPixelsPerCell; ++j)
			pixelMemory[i + baseX + gScreenResolution * (j + baseY)] = finalColor;
}

void TransferHelper::VisualizeScalarField(float* deviceMemory, float maximumValue, size_t devicePitch,
                                          uchar4* pixelMemory)
{
	
	VisualizeField CUDA_DECORATOR_LOGIC (deviceMemory, devicePitch, maximumValue, pixelMemory);
}




__global__ void GenerateLineFlags(float* dataMemory, size_t dataStride, unsigned int* isoLineFlags, size_t isoLineStride, float isoLineStepSize)
{
	__shared__ float valueBuffer[gBlockSize + 2][gBlockSize + 2];

	
	// We keep tack of the pixel block we are responsible for.
	int xOrigin = threadIdx.x + gBlockSize * blockIdx.x;   
	int yOrigin = threadIdx.y + gBlockSize * blockIdx.y;


	valueBuffer[threadIdx.x + 1][threadIdx.y + 1] = dataMemory[(xOrigin + 1) + (yOrigin + 1) * dataStride ];
	
	if (threadIdx.x == 0)
		valueBuffer[threadIdx.x][threadIdx.y + 1] = dataMemory[(xOrigin ) + (yOrigin + 1) * dataStride ];
	if (threadIdx.x == 31)
		valueBuffer[threadIdx.x + 2][threadIdx.y + 1] = dataMemory[(xOrigin + 2) + (yOrigin + 1) * dataStride ];
	if (threadIdx.y == 0)
		valueBuffer[threadIdx.x + 1][threadIdx.y ] = dataMemory[(xOrigin + 1) + (yOrigin ) * dataStride ];
	if (threadIdx.y == 31)
		valueBuffer[threadIdx.x + 1][threadIdx.y + 2] = dataMemory[(xOrigin + 1) + (yOrigin + 2) * dataStride ];
	
	__syncthreads();

	int xScan = threadIdx.x + 1;
	int yScan = threadIdx.y + 1;

	// Get nearest iso value.
	float localValue = valueBuffer[xScan][yScan];
	float nearestIsoValue = roundf(localValue / isoLineStepSize) * isoLineStepSize;

	bool linePlotting = false;

	linePlotting |= (valueBuffer[xScan - 1][yScan] > nearestIsoValue);
	linePlotting |= (valueBuffer[xScan + 1][yScan] > nearestIsoValue);
	linePlotting |= (valueBuffer[xScan][yScan + 1] > nearestIsoValue);
	linePlotting |= (valueBuffer[xScan][yScan - 1] > nearestIsoValue);

	linePlotting &= (localValue <= nearestIsoValue);

	isoLineFlags[xOrigin  + yOrigin  * isoLineStride] = linePlotting;
	
	
}

void TransferHelper::VisualizeIsoLines(float* dataMemory, float isoLineStepSize, size_t rawDataStride, uchar4* pixelMemory)
{
	if (m_isoLineData == NULL)
		m_isoLineData = (unsigned int*)ReserveFloatMemory(m_isoLineStride);

	dim3 block(32, 32);
	dim3 grid(3, 3);
	
	GenerateLineFlags CUDA_DECORATOR_LOGIC(dataMemory, rawDataStride, (unsigned int *)m_isoLineData, m_isoLineStride, isoLineStepSize);
	MarcIfFlagged CUDA_DECORATOR_LOGIC((unsigned int*)m_isoLineData, m_isoLineStride, pixelMemory, make_uchar4(128, 128, 128, 255));
}
