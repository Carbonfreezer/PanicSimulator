#include "CheckerTest.h"
#include <cuda.h>
#include  <vector_types.h>
#include <cassert>
#include <device_launch_parameters.h>
#include "CudaHelper.h"


__global__ void GenerateStructure(uchar4* deviceMemory, int pixelExtension, unsigned char updateCounter)
{
	
	int pixelPosX = threadIdx.x + blockIdx.x * blockDim.x;
	int pixelPosY = threadIdx.y + blockIdx.y * blockDim.y;

	int pixelPosition = pixelPosX + pixelPosY * pixelExtension;
	
	int offsetCounter = (blockIdx.x + blockIdx.y) % 2;
	unsigned char color = 0;
	
	if (offsetCounter == 1)
		color = updateCounter;
	
	uchar4 complete;
	complete.x = complete.y = complete.z = color;
	complete.w = 255;
	

	deviceMemory[pixelPosition] = complete;

	

	
}

void CheckerTest::UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase)
{
	m_updateCounter += 1;
	
	GenerateStructure << <dim3(gNumOfBlocks * gPixelsPerCell, gNumOfBlocks * gPixelsPerCell), dim3(gBlockSize, gBlockSize) >> > (deviceMemory, gScreenResolution, m_updateCounter);
	
}
