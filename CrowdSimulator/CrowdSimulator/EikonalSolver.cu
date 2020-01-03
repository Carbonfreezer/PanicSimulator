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


__device__ int gHasConverged;

__device__ int gBlocksCountedForConvergence;
__device__ int gBlocksHaveConverged;
// __device__ int gNumChecks;


__global__ void ResetConvergence()
{
	gHasConverged = 0;
	gBlocksCountedForConvergence = 0;
	gBlocksHaveConverged = 0;
	// gNumChecks = 0;
}


__global__ void RunIconalGodunov(float* timeFieldSource, float* timeFieldDestination, size_t timeStride, float* velocityField, size_t velocityStride)
{
	// Once we have found a convergence we can quit.
	if (gHasConverged)
		return;
	
	__shared__ float timeBuffer[2][gBlockSize + 2][gBlockSize + 2];

	__shared__ int verifiedThreads;
	__shared__ int convergedThreads;


	// Needed for the convergence check.
	if ((threadIdx.x == 0) && (threadIdx.y == 0))
	{
		verifiedThreads = 0;
		convergedThreads = 0;
	}

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
	float valueOriginal = timeEstimate;
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

	// We do the convergence check herer.
	int converged = 0;

	if (isinf(valueOriginal) && isinf(timeEstimate))
		converged = 1;
	else if (isfinite(valueOriginal) && isfinite(timeEstimate))
		converged = fabsf(valueOriginal - timeEstimate) < gMaximalGodunovError;

	atomicAdd(&convergedThreads, converged);
	int threadsProcessed = atomicAdd(&verifiedThreads, 1) + 1;

	// When we are not the last thread we are done here.
	if (threadsProcessed != gBlockSize * gBlockSize)
		return;

	int blockHasConverged = (convergedThreads == gBlockSize * gBlockSize);
	atomicAdd(&gBlocksHaveConverged, blockHasConverged);

	int blocksProcessed = atomicAdd(&gBlocksCountedForConvergence, 1) + 1;
	if (blocksProcessed != gNumOfBlocks * gNumOfBlocks)
		return;

	// Now we are the last thread of the last block;
	gHasConverged = (gBlocksHaveConverged == gNumOfBlocks * gNumOfBlocks);


	/*
	gNumChecks += 1;
	if (gHasConverged)
		gHasConverged = gHasConverged;
	*/
	
	gBlocksCountedForConvergence = 0;
	gBlocksHaveConverged = 0;

	
}




void EikonalSolver::SolveEquation( FloatArray velocity, DataBase* dataBase)
{

	assert(m_timeArray[0].m_array);
	PrepareTimeField CUDA_DECORATOR_LOGIC (m_timeArray[0].m_array, m_timeArray[0].m_stride, dataBase->GetTargetData().m_array, dataBase->GetTargetData().m_stride);
	TransferHelper::CopyDataFromTo(m_timeArray[0], m_timeArray[1]);

	ResetConvergence << <1, 1 >> > ();
	
	
	// We have tu run a prestep to get the border filled.
	for (int i = 0; i < gMaximumIterationsGodunov; ++i)
	{
		RunIconalGodunov CUDA_DECORATOR_LOGIC(m_timeArray[i % 2].m_array, m_timeArray[1 - i % 2 ].m_array, m_timeArray[0].m_stride,
			velocity.m_array, velocity.m_stride);
	}

}


