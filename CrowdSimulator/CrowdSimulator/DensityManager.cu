#include "DensityManager.h"
#include "GlobalConstants.h"
#include "CudaHelper.h"
#include <cassert>
#include <device_launch_parameters.h>
#include <math.h>

DensityManager::DensityManager()
{
	m_targetAreaData = NULL;
	m_spawnAreaData = NULL;
	m_densityBuffer[0] = m_densityBuffer[1] = NULL;
	m_doubleBufferDensity = 0;
}

void DensityManager::InitializeManager(const char* spawnAreaFile, const char* targetAreaFile)
{
	assert(m_targetAreaData == NULL);

	m_targetAreaReader.ReadFile(targetAreaFile);
	m_spawnAreaReader.ReadFile(spawnAreaFile);

	m_spawnAreaData = m_transferHelper.UploadPictureAsFloat(&m_spawnAreaReader, 0.0f, 0.0f, gMaximumDensity, m_spawnAreaStride);
	m_targetAreaData = m_transferHelper.UploadPicture(&m_targetAreaReader, 0, m_targetAreaStride);
	m_densityBuffer[0] = m_transferHelper.ReserveFloatMemory(m_densityStride);
	m_densityBuffer[1] = m_transferHelper.ReserveFloatMemory(m_densityStride);
}

__global__ void ApplyConditions(float* densityBuffer, size_t strideDensity, unsigned* wallInformaton, size_t strideWall, float* spawnArea, size_t strideSpawn,
                    unsigned int* targetAreaData, size_t strideTarget)
{
	int xRead = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yRead = threadIdx.y + blockIdx.y * blockDim.y + 1;


	// Deal with the border cases.
	if (xRead == 1)
		densityBuffer[yRead * strideDensity] = 0.0f;
	if (yRead == 1)
		densityBuffer[xRead] = 0.0f;
	if (xRead == gGridSizeExternal - 2)
		densityBuffer[(gGridSizeExternal - 1) + yRead * strideDensity] = 0.0f;
	if (yRead == gGridSizeExternal - 2)
		densityBuffer[xRead + (gGridSizeExternal - 1) * strideDensity] = 0.0f;


	// Special fix for the corner cases.
	if ((xRead == 1) && (yRead == 1))
	{
		densityBuffer[0] = 0.0f;
		densityBuffer[gGridSizeExternal - 1] = 0.0f;
		densityBuffer[gGridSizeExternal - 1 + strideDensity * (gGridSizeExternal - 1)] = 0.0f;
		densityBuffer[strideDensity * (gGridSizeExternal - 1)] = 0.0f;
	}

	if ((wallInformaton[xRead + strideWall * yRead]) || (targetAreaData[xRead + strideTarget * yRead]))
	{
		// Wliminate density on walls and targets.
		densityBuffer[xRead + strideDensity * yRead] = 0.0f;
	} else
	{
		// Build the maximum, there may already be other people walking through the spawn area.
		float spawnValue = spawnArea[xRead + yRead * strideSpawn];
		if (spawnValue > 0.001f)
			densityBuffer[xRead + strideDensity * yRead] = fmaxf(densityBuffer[xRead + strideDensity * yRead], spawnValue);
	}

	

}

void DensityManager::EnforceBoundaryConditions(unsigned* wallInformation, size_t wallStride)
{
	ApplyConditions  CUDA_DECORATOR_LOGIC (m_densityBuffer[m_doubleBufferDensity], m_densityStride, wallInformation, wallStride,
		m_spawnAreaData, m_spawnAreaStride, m_targetAreaData, m_targetAreaStride);
}

void DensityManager::GenerateDensityVisualization(uchar4* textureMemory)
{
	m_transferHelper.VisualizeScalarField(m_densityBuffer[m_doubleBufferDensity], gMaximumDensity, m_densityStride, textureMemory);
}
