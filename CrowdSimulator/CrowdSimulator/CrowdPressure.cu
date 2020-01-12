#include "CrowdPressure.h"
#include "TransferHelper.h"
#include "DataBase.h"
#include <cassert>
#include "CudaHelper.h"
#include <device_launch_parameters.h>

void CrowdPressure::ToolSystem()
{
	m_gradienModule.PreprareModule();
	m_pressureArray = TransferHelper::ReserveFloatMemory();
}

__global__ void ComputeCrowdPressureCuda(size_t strides, float* densityField, float* gradientXVelocity, float* gradientYVelocity, float* result)
{
	int xOrigin = threadIdx.x + gBlockSize * blockIdx.x + 1;
	int yOrigin = threadIdx.y + gBlockSize * blockIdx.y + 1;

	int totalIndex = xOrigin + strides * yOrigin;

	float xGrad = gradientXVelocity[totalIndex];
	float yGrad = gradientYVelocity[totalIndex];
	
	result[totalIndex] = densityField[totalIndex] * (xGrad * xGrad + yGrad * yGrad);
}

void CrowdPressure::ComputeCrowdPressure(FloatArray density, FloatArray velocity, DataBase* dataBase)
{
	// Get the gradient for the velocity.
	m_gradienModule.ComputeGradient(velocity, dataBase->GetWallData());
	FloatArray gradX = m_gradienModule.GetXComponent();
	FloatArray gradY = m_gradienModule.GetYComponent();

	assert(density.m_stride == gradX.m_stride);
	assert(density.m_stride == gradY.m_stride);
	assert(density.m_stride == m_pressureArray.m_stride);

	ComputeCrowdPressureCuda CUDA_DECORATOR_LOGIC (density.m_stride, density.m_array, gradX.m_array, gradY.m_array, m_pressureArray.m_array);
	
}
