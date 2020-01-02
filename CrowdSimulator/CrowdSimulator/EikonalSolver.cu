#include "EikonalSolver.h"
#include "CudaHelper.h"
#include <device_launch_parameters.h>
#include <cmath>
#include <cassert>
#include "DataBase.h"


void EikonalSolver::PrepareSolving()
{
	assert(m_timeArray->m_array == NULL);

	m_timeArray[0] = TransferHelper::ReserveFloatMemory();
	m_timeArray[1] = TransferHelper::ReserveFloatMemory();
	assert(m_timeArray[0].m_stride == m_timeArray[1].m_stride);
	m_usedDoubleBuffer = 0;
}


__global__ void PrepareTimeField(float* timeBuffer, size_t timeStide, unsigned* targetBuffer, size_t targetStride)
{
	int xRead = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yRead = threadIdx.y + blockIdx.y * blockDim.y + 1;

	// Deal with the border cases.
	if (xRead == 1)
		timeBuffer[yRead * timeStide] = INFINITY;
	if (yRead == 1)
		timeBuffer[xRead] = INFINITY;
	if (xRead == gGridSizeExternal - 2)
		timeBuffer[(gGridSizeExternal - 1) + yRead * timeStide] = INFINITY;
	if (yRead == gGridSizeExternal - 2)
		timeBuffer[xRead + (gGridSizeExternal - 1) * timeStide] = INFINITY;

	// Special fix for the corner cases.
	if ((xRead == 1) && (yRead == 1))
	{
		timeBuffer[0] = INFINITY;
		timeBuffer[gGridSizeExternal - 1] = INFINITY;
		timeBuffer[gGridSizeExternal - 1 + timeStide * (gGridSizeExternal - 1)] = INFINITY;
		timeBuffer[timeStide * (gGridSizeExternal - 1)] = INFINITY;
	}

	
	if (targetBuffer[xRead + yRead * targetStride])
		timeBuffer[xRead + yRead * timeStide] = 0.0f;
	else
		timeBuffer[xRead + yRead * timeStide] = INFINITY;

}



__global__ void RunIconalGodunov(float* timeFieldSource, float* timeFieldDestination, size_t timeStride, float* velocityField, size_t velocityStride)
{
	__shared__ float timeBuffer[2][gBlockSize + 2][gBlockSize + 2];


	// We keep tack of the pixel  we are responsible for.
	int xOrigin = threadIdx.x + gBlockSize * blockIdx.x + 1;
	int yOrigin = threadIdx.y + gBlockSize * blockIdx.y + 1;

	int xScan = threadIdx.x + 1;
	int yScan = threadIdx.y + 1;

	timeBuffer[0][xScan][yScan] = timeBuffer[1][xScan][yScan] = timeFieldSource[xOrigin  + yOrigin  * timeStride];

	if (threadIdx.x == 0)
		timeBuffer[0][xScan - 1][yScan] = timeBuffer[1][xScan - 1][yScan] = timeFieldSource[(xOrigin - 1)+ yOrigin  * timeStride];
	if (threadIdx.x == 31)
		timeBuffer[0][xScan + 1][yScan] = timeBuffer[1][xScan + 1][yScan] = timeFieldSource[(xOrigin + 1) + yOrigin * timeStride];
	if (threadIdx.y == 0)
		timeBuffer[0][xScan][yScan - 1] = timeBuffer[1][xScan][yScan - 1] = timeFieldSource[xOrigin + (yOrigin - 1) * timeStride];
	if (threadIdx.y == 31)
		timeBuffer[0][xScan][yScan + 1] = timeBuffer[1][xScan][yScan + 1] = timeFieldSource[xOrigin + (yOrigin + 1) * timeStride];
	// Diagonal case is not used in the godunov part.
	__syncthreads();

	int sourceBuffer = 0;
	
	float currentInverseVelocity = gCellSize / velocityField[xOrigin  + yOrigin  * velocityStride];
	float timeEstimate = timeBuffer[sourceBuffer][xScan][yScan];
	// Initial guess for iterations.
	for (int count = 0; count < gBlockSize; ++count)
	{

		float u = fminf(timeBuffer[sourceBuffer][xScan + 1][yScan], timeBuffer[sourceBuffer][xScan - 1][yScan]);
		float v = fminf(timeBuffer[sourceBuffer][xScan ][yScan + 1], timeBuffer[sourceBuffer][xScan][yScan - 1]);


		int conda = (timeEstimate <= u);
		int condb = (timeEstimate <= v);

		if (conda + condb == 0)
		{
			float beta = -(u + v);
			float c = u * u + v * v - currentInverseVelocity * currentInverseVelocity;
			float sqrtContainer = (beta * beta - 2.0f * c);
			if (sqrtContainer >= 0.0f)
				timeEstimate = (-beta + sqrtf(sqrtContainer)) * 0.5f;
			else
				timeEstimate = currentInverseVelocity + fminf(u, v);

		}
		else if (conda == 0)
			timeEstimate = currentInverseVelocity + u;
		else if (condb == 0)
			timeEstimate = currentInverseVelocity + v;
		
		
		sourceBuffer = 1 - sourceBuffer;
		timeBuffer[sourceBuffer][xScan][yScan] = timeEstimate;
		__syncthreads();
	}

	timeFieldDestination[xOrigin  + yOrigin  * timeStride] = timeEstimate;
}





void EikonalSolver::PerformIterations(int outerIterations , FloatArray velocity, DataBase* dataBase)
{

	assert(m_timeArray[0].m_array);
	PrepareTimeField CUDA_DECORATOR_LOGIC (m_timeArray[0].m_array, m_timeArray[0].m_stride, dataBase->GetTargetData().m_array, dataBase->GetTargetData().m_stride);
	TransferHelper::CopyDataFromTo(m_timeArray[0], m_timeArray[1]);


	// We have tu run a prestep to get the border filled.
	m_usedDoubleBuffer = 0;
	for (int i = 0; i < outerIterations; ++i)
	{
		RunIconalGodunov CUDA_DECORATOR_LOGIC(m_timeArray[m_usedDoubleBuffer].m_array, m_timeArray[1 - m_usedDoubleBuffer].m_array, m_timeArray[0].m_stride,
			velocity.m_array, velocity.m_stride);
		m_usedDoubleBuffer = 1 - m_usedDoubleBuffer;
	}

}


