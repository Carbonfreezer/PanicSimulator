#include "VelocityManager.h"
#include <cassert>
#include "CudaHelper.h"
#include <device_launch_parameters.h>
#include <math.h>
#include "DataBase.h"
#include "TransferHelper.h"
#include "VisualizationHelper.h"

// Contains the movmement velocity in m/s from 0 Persons / sqm to 5.5 persons / sqm in 0.5 steps.
__constant__ float m_velocityLookupTable[12] = {1.34f, 1.23f, 1.03f, 0.77f, 0.56f, 0.41f, 0.32f, 0.26f, 0.21f, 0.17f, 0.12f, 0.0f};

void VelocityManager::GenerateVelocityField()
{
	m_velocityField = TransferHelper::ReserveFloatMemory();
}


__global__ void UpdateVelocity(float* velocityField, size_t velocityStride, float* densityField, size_t density_stride, unsigned* wallArea, size_t wallStride)
{
	int xRead = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yRead = threadIdx.y + blockIdx.y * blockDim.y + 1;


	// Deal with the border cases.
	if (xRead == 1)
		velocityField[yRead * velocityStride] = 0.0f;
	if (yRead == 1)
		velocityField[xRead] = 0.0f;
	if (xRead == gGridSizeExternal - 2)
		velocityField[(gGridSizeExternal - 1) + yRead * velocityStride] = 0.0f;
	if (yRead == gGridSizeExternal - 2)
		velocityField[xRead + (gGridSizeExternal - 1) * velocityStride] = 0.0f;


	// Special fix for the corner cases.
	if ((xRead == 1) && (yRead == 1))
	{
		velocityField[0] = 0.0f;
		velocityField[gGridSizeExternal - 1] = 0.0f;
		velocityField[gGridSizeExternal - 1 + velocityStride * (gGridSizeExternal - 1)] = 0.0f;
		velocityField[velocityStride * (gGridSizeExternal - 1)] = 0.0f;
	}

	if (wallArea[xRead + yRead * wallStride])
	{
		velocityField[xRead + yRead * velocityStride] = 0.0f;
	}
	else
	{
		float density = densityField[xRead + yRead * density_stride];
		if (density >= gMaximumDensity)
		{
			velocityField[xRead + yRead * velocityStride] = 0.0f;
		}
		else
		{
			int baseCounter = (int)floorf(density * 2.0f);
			float alpha = (density - 0.5f * baseCounter) * 2.0f;
			velocityField[xRead + yRead * velocityStride] = alpha * m_velocityLookupTable[baseCounter + 1] + (1.0f - alpha) * m_velocityLookupTable[baseCounter];
		}

	}
}

void VelocityManager::UpdateVelocityField(FloatArray density, DataBase* dataBase)
{
	assert(m_velocityField.m_array);
	UpdateVelocity CUDA_DECORATOR_LOGIC(m_velocityField.m_array, m_velocityField.m_stride,
		density.m_array, density.m_stride, dataBase->GetWallData().m_array, dataBase->GetWallData().m_stride);
}


void VelocityManager::GenerateVelocityVisualization(uchar4* textureMemory, float isoLineDistance)
{
	VisualizationHelper::VisualizeScalarField(m_velocityField, gMaximumWalkingVelocity, textureMemory);
	VisualizationHelper::VisualizeIsoLines(m_velocityField, isoLineDistance, textureMemory);
}


