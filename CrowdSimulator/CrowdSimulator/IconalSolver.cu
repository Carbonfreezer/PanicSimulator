#include "IconalSolver.h"
#include "CudaHelper.h"
#include <device_launch_parameters.h>
#include <cmath>

void IconalSolver::SetWallFile(const char* wallInformation)
{
	m_wallPicture.ReadFile(wallInformation);
	m_wallInformation = m_transferHelper.UploadPicture(&m_wallPicture, 255, m_wallStride);
}

void IconalSolver::SetTargetArea(const char* targetInformation)
{
	m_targetPicture.ReadFile(targetInformation);
	m_targetAreaInformation = m_transferHelper.UploadPicture(&m_targetPicture, 0, m_targetStride);
}

void IconalSolver::PrepareSolving()
{
	m_velocityField = m_transferHelper.ReserveFloatMemory(m_velocityStride);
	for(int i = 0; i < 2; ++i)
		m_bufferTime[i] = m_transferHelper.ReserveFloatMemory(m_timeStride);
}

__global__ void PrepareVelocityField(unsigned* wallInformation, size_t wallStride, float* velocityField, size_t velocityStride)
{
	
	
	int xRead = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yRead = threadIdx.y + blockIdx.y * blockDim.y + 1;


	// Deal with the border cases.
	if (xRead == 1)
		velocityField[ yRead * velocityStride] = 0.0f;
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
	
	if (wallInformation[xRead + yRead * wallStride ])
		velocityField[xRead + yRead * velocityStride ] = 0.0f;
	else
		velocityField[xRead + yRead * velocityStride ] = gMaximumWalkingVelocity;
		
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



__global__ void RunIconalDijkstra(float* timeFieldSource,  float* timeFieldDestination, size_t timeStride, float* velocityField, size_t velocityStride)
{
	__shared__ float timeBuffer[2][gBlockSize + 2][gBlockSize + 2];


	// We keep tack of the pixel  we are responsible for.
	int xOrigin = threadIdx.x + gBlockSize * blockIdx.x;
	int yOrigin = threadIdx.y + gBlockSize * blockIdx.y;


	timeBuffer[0][threadIdx.x + 1][threadIdx.y + 1] = timeBuffer[1][threadIdx.x + 1][threadIdx.y + 1] = timeFieldSource[(xOrigin + 1) + (yOrigin + 1) * timeStride];

	// We do not need to copy over the boundary elements of the velocity because we will not need them.
	
	if (threadIdx.x == 0)
		timeBuffer[0][threadIdx.x ][threadIdx.y + 1] = timeBuffer[1][threadIdx.x][threadIdx.y + 1] = timeFieldSource[(xOrigin) + (yOrigin + 1) * timeStride];
	if (threadIdx.x == 31)
		timeBuffer[0][threadIdx.x + 2][threadIdx.y + 1] = timeBuffer[1][threadIdx.x + 2][threadIdx.y + 1] = timeFieldSource[(xOrigin + 2) + (yOrigin + 1) * timeStride];
	if (threadIdx.y == 0)
		timeBuffer[0][threadIdx.x + 1][threadIdx.y] = timeBuffer[1][threadIdx.x + 1][threadIdx.y] = timeFieldSource[(xOrigin + 1) + (yOrigin) * timeStride];
	if (threadIdx.y == 31)
		timeBuffer[0][threadIdx.x + 1][threadIdx.y + 2] = timeBuffer[1][threadIdx.x + 1][threadIdx.y + 2] = timeFieldSource[(xOrigin + 1) + (yOrigin + 2) * timeStride];

	// Special fix for the corner case. (only used in dikjstra solver because of diagonal element).
	if ((threadIdx.x == 0) && (threadIdx.y == 0))
	{
		timeBuffer[0][0][0] = timeBuffer[1][0][0] = timeFieldSource[xOrigin + yOrigin  * timeStride];
		timeBuffer[0][33][0] = timeBuffer[1][33][0] = timeFieldSource[(xOrigin + gBlockSize) + yOrigin * timeStride];
		timeBuffer[0][0][33] = timeBuffer[1][0][33] = timeFieldSource[(xOrigin ) + (yOrigin + gBlockSize) * timeStride];
		timeBuffer[0][33][33] = timeBuffer[1][33][33] = timeFieldSource[(xOrigin + gBlockSize) + (yOrigin + gBlockSize) * timeStride];
	}


	__syncthreads();



	int sourceBuffer = 0;
	int xScan = threadIdx.x + 1;
	int yScan = threadIdx.y + 1;

	float bestTime = timeBuffer[sourceBuffer][xScan][yScan];

	float reciprocalVelocity = 1.0f / velocityField[(xOrigin + 1) + (yOrigin + 1) * velocityStride];
	float horrizontalTime = reciprocalVelocity * gCellSize;
	float diagonalTime = reciprocalVelocity * gCellSizeDiagonal;



	// Initial guess for iterations.
	for (int count = 0; count < gBlockSize ; ++count)
	{
		
		bestTime = fminf(bestTime, timeBuffer[sourceBuffer][xScan - 1][yScan] + horrizontalTime);
		bestTime = fminf(bestTime, timeBuffer[sourceBuffer][xScan + 1][yScan] + horrizontalTime);
		bestTime = fminf(bestTime, timeBuffer[sourceBuffer][xScan][yScan - 1] + horrizontalTime);
		bestTime = fminf(bestTime, timeBuffer[sourceBuffer][xScan][yScan + 1] + horrizontalTime);

		
		bestTime = fminf(bestTime, timeBuffer[sourceBuffer][xScan - 1][yScan - 1] + diagonalTime);
		bestTime = fminf(bestTime, timeBuffer[sourceBuffer][xScan + 1][yScan + 1] + diagonalTime);
		bestTime = fminf(bestTime, timeBuffer[sourceBuffer][xScan + 1][yScan - 1] + diagonalTime);
		bestTime = fminf(bestTime, timeBuffer[sourceBuffer][xScan - 1][yScan + 1] + diagonalTime);
		
		sourceBuffer = 1 - sourceBuffer;
		timeBuffer[sourceBuffer][xScan][yScan] = bestTime;

		__syncthreads();
	}

	timeFieldDestination[(xOrigin + 1) + (yOrigin + 1) * timeStride] = bestTime;
		
}

__global__ void RunIconalGodunov(float* timeFieldSource, float* timeFieldDestination, size_t timeStride, float* velocityField, size_t velocityStride)
{
	__shared__ float timeBuffer[2][gBlockSize + 2][gBlockSize + 2];


	// We keep tack of the pixel  we are responsible for.
	int xOrigin = threadIdx.x + gBlockSize * blockIdx.x ;
	int yOrigin = threadIdx.y + gBlockSize * blockIdx.y ;

	timeBuffer[0][threadIdx.x + 1][threadIdx.y + 1] = timeBuffer[1][threadIdx.x + 1][threadIdx.y + 1] = timeFieldSource[(xOrigin + 1) + (yOrigin + 1) * timeStride];

	if (threadIdx.x == 0)
		timeBuffer[0][threadIdx.x][threadIdx.y + 1] = timeBuffer[1][threadIdx.x][threadIdx.y + 1] = timeFieldSource[(xOrigin)+(yOrigin + 1) * timeStride];
	if (threadIdx.x == 31)
		timeBuffer[0][threadIdx.x + 2][threadIdx.y + 1] = timeBuffer[1][threadIdx.x + 2][threadIdx.y + 1] = timeFieldSource[(xOrigin + 2) + (yOrigin + 1) * timeStride];
	if (threadIdx.y == 0)
		timeBuffer[0][threadIdx.x + 1][threadIdx.y] = timeBuffer[1][threadIdx.x + 1][threadIdx.y] = timeFieldSource[(xOrigin + 1) + (yOrigin)* timeStride];
	if (threadIdx.y == 31)
		timeBuffer[0][threadIdx.x + 1][threadIdx.y + 2] = timeBuffer[1][threadIdx.x + 1][threadIdx.y + 2] = timeFieldSource[(xOrigin + 1) + (yOrigin + 2) * timeStride];
	// Diagonal case is not used in the godunov part.
	__syncthreads();

	int sourceBuffer = 0;
	int xScan = threadIdx.x + 1;
	int yScan = threadIdx.y + 1;
	float currentInverseVelocity = gCellSize / velocityField[(xOrigin + 1) + (yOrigin + 1) * velocityStride];
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
		}
		else if (conda == 0)
			timeEstimate = currentInverseVelocity + u;
		else if (condb == 0)
			timeEstimate = currentInverseVelocity + v;
		
		sourceBuffer = 1 - sourceBuffer;
		timeBuffer[sourceBuffer][xScan][yScan] = timeEstimate;
		__syncthreads();
	}

	timeFieldDestination[(xOrigin + 1) + (yOrigin + 1) * timeStride] = timeEstimate;
}





void IconalSolver::PerformIterations(int outerIterations)
{
	PrepareVelocityField CUDA_DECORATOR_LOGIC (m_wallInformation, m_wallStride, m_velocityField, m_velocityStride);
	PrepareTimeField CUDA_DECORATOR_LOGIC (m_bufferTime[0], m_timeStride, m_targetAreaInformation, m_targetStride);
	PrepareTimeField CUDA_DECORATOR_LOGIC(m_bufferTime[1], m_timeStride, m_targetAreaInformation, m_targetStride);


	// We have tu run a prestep to get the border filled.
	int m_usedDoubleBuffer = 0;
	for (int i = 0; i < outerIterations; ++i)
	{
		// RunIconalDijkstra CUDA_DECORATOR_LOGIC(m_bufferTime[m_usedDoubleBuffer], m_bufferTime[ 1 - m_usedDoubleBuffer], m_timeStride, m_velocityField, m_velocityStride);
		RunIconalGodunov CUDA_DECORATOR_LOGIC(m_bufferTime[m_usedDoubleBuffer], m_bufferTime[1 - m_usedDoubleBuffer], m_timeStride, m_velocityField, m_velocityStride);
		m_usedDoubleBuffer = 1 - m_usedDoubleBuffer;
	}
	

	

}

float* IconalSolver::GetTimeField(size_t& timeStride)
{
	
	timeStride = m_timeStride;
	return m_bufferTime[m_usedDoubleBuffer];
}


