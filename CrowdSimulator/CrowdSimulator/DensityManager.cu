#include "DensityManager.h"
#include "GlobalConstants.h"
#include "CudaHelper.h"
#include <cassert>
#include <device_launch_parameters.h>
#include <math.h>
#include "DataBase.h"
#include "VisualizationHelper.h"


void DensityManager::InitializeManager(DataBase* dataBase)
{
	m_density = TransferHelper::ReserveFloatMemory();
	TransferHelper::CopyDataFromTo(dataBase->GetInitialDensityData(), m_density);
	m_continuitySolver.PrepareSolver();
}

__global__ void ApplyConditions(float* densityBuffer, size_t strideDensity,  float* spawnArea, size_t strideSpawn,
                    unsigned int* despawnData, size_t despawnStride)
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

	if  (despawnData[xRead + despawnStride * yRead])
	{
		// Eliminate density on targets.
		densityBuffer[xRead + strideDensity * yRead] = 0.0f;
		
	} else
	{
		// Build the maximum, there may already be other people walking through the spawn area.
		float spawnValue = spawnArea[xRead + yRead * strideSpawn];
		if (spawnValue > 0.001f)
			densityBuffer[xRead + strideDensity * yRead] = fmaxf(densityBuffer[xRead + strideDensity * yRead], spawnValue);
	}

	

}

void DensityManager::EnforceBoundaryConditions(DataBase* dataBase)
{
	ApplyConditions  CUDA_DECORATOR_LOGIC (m_density.m_array, m_density.m_stride,
		dataBase->GetSpawnData().m_array, dataBase->GetSpawnData().m_stride,
		dataBase->GetDespawnData().m_array, dataBase->GetDespawnData().m_stride);
}



void DensityManager::GenerateDensityVisualization(uchar4* textureMemory)
{
	VisualizationHelper::VisualizeScalarField(m_density, gMaximumDensity,  textureMemory);
}

void DensityManager::UpdateDensityField(float timePassed, FloatArray timeField,  DataBase* dataBase)
{
	m_continuitySolver.IntegrateEquation(timePassed, m_density,  timeField, dataBase);
}

void DensityManager::ResetDensity(DataBase* dataBase)
{
	TransferHelper::CopyDataFromTo(dataBase->GetInitialDensityData(), m_density);
}
