#include "ContinuityEquationSolver.h"
#include <cassert>
#include "CudaHelper.h"
#include <device_launch_parameters.h>
#include <math.h>
#include "TransferHelper.h"
#include "DataBase.h"




void ContinuityEquationSolver::PrepareSolver()
{
	m_gradientIconal.PreprareModule();
	m_derivativeBuffer = TransferHelper::ReserveFloatMemory();
}


__global__ void GetTimeDerivative( size_t strides, float* density,  
								float* derivativeX, float* derivativeY,
							float* result)
{
	__shared__ float xDiv[gBlockSize + 2][gBlockSize + 2];
	__shared__ float yDiv[gBlockSize + 2][gBlockSize + 2];

	

	// Prefill the data.

	// We keep tack of the pixel  we are responsible for.
	int xOrigin = threadIdx.x + gBlockSize * blockIdx.x + 1;
	int yOrigin = threadIdx.y + gBlockSize * blockIdx.y + 1;

	int xScan = threadIdx.x + 1;
	int yScan = threadIdx.y + 1;

	float factor;
	
	float xDivPure;
	float yDivPure;
	float localDensity;

	float xGrad, yGrad;

	xGrad = derivativeX[xOrigin + yOrigin * strides];
	yGrad = derivativeY[xOrigin + yOrigin * strides];


	// If both are zero we get an error here.
	factor = 1.0f / (xGrad * xGrad + yGrad * yGrad + FLT_EPSILON);
	localDensity = density[xOrigin + yOrigin * strides];



	// Deal with the inf * zero situation.
	if (factor == 0.0f)
	{
		xDivPure = 0.0f;
		yDivPure = 0.0f;
	}
	else
	{
		xDivPure = xGrad * factor;
		yDivPure = yGrad * factor;
	}
	
	xDiv[xScan][yScan] = xDivPure * localDensity;
	yDiv[xScan][yScan] = yDivPure * localDensity;

	// Copy over the border lines, we do not need the corner elements.
	// We also do not need the density on the border lines.
	if (threadIdx.x == 0)
	{
		xGrad = derivativeX[(xOrigin - 1) + yOrigin * strides];
		yGrad = derivativeY[(xOrigin - 1) + yOrigin * strides];
		
		factor  = density[(xOrigin - 1) + yOrigin * strides] /(xGrad * xGrad + yGrad * yGrad + FLT_EPSILON);

		if (factor == 0.0f)
		{
			xDiv[xScan - 1][yScan] = 0.0f;
			yDiv[xScan - 1][yScan] = 0.0f;
		}
		else
		{
			xDiv[xScan - 1][yScan] = xGrad * factor;
			yDiv[xScan - 1][yScan] = yGrad * factor;
		}
		
	}
		
	if (threadIdx.x == 31)
	{
		xGrad = derivativeX[(xOrigin + 1) + yOrigin * strides];
		yGrad = derivativeY[(xOrigin + 1) + yOrigin * strides];
		
		factor = density[(xOrigin + 1) + yOrigin * strides] / (xGrad * xGrad + yGrad * yGrad  + FLT_EPSILON);

		if (factor == 0.0f)
		{
			xDiv[xScan + 1][yScan] = 0.0f;
			yDiv[xScan + 1][yScan] = 0.0f;

		}
		else
		{
			xDiv[xScan + 1][yScan] = xGrad * factor;
			yDiv[xScan + 1][yScan] = yGrad * factor;

		}
	}
		
	if (threadIdx.y == 0)
	{
		xGrad = derivativeX[(xOrigin)+(yOrigin - 1) * strides];
		yGrad = derivativeY[(xOrigin)+(yOrigin - 1) * strides];

		factor = density[(xOrigin)+(yOrigin - 1) * strides] / (xGrad * xGrad + yGrad * yGrad + FLT_EPSILON);

		if (factor == 0.0f)
		{
			xDiv[xScan][yScan - 1] = 0.0f;
			yDiv[xScan][yScan - 1] = 0.0f;

		}
		else
		{
			xDiv[xScan][yScan - 1] = xGrad * factor;
			yDiv[xScan][yScan - 1] = yGrad * factor;

		}
	}
		
	if (threadIdx.y == 31)
	{
		xGrad = derivativeX[(xOrigin)+(yOrigin + 1) * strides];
		yGrad = derivativeY[(xOrigin)+(yOrigin + 1) * strides];

		factor = density[(xOrigin)+(yOrigin + 1) * strides] / (xGrad * xGrad + yGrad * yGrad + FLT_EPSILON);

		if (factor == 0.0f)
		{
			xDiv[xScan][yScan + 1] = 0.0f;
			yDiv[xScan][yScan + 1] = 0.0f;
		}
		else
		{
			xDiv[xScan][yScan + 1] = xGrad * factor;
			yDiv[xScan][yScan + 1] = yGrad * factor;
		}
	
	}
		

	__syncthreads();

	
	float xDerivative;
	if (xDivPure >= 0.0f)
		xDerivative = (xDiv[xScan + 1][yScan] - xDiv[xScan][yScan]) / (gCellSize);
	else
		xDerivative = (xDiv[xScan][yScan] - xDiv[xScan - 1][yScan]) / (gCellSize);

	
	float yDerivative;
	if (yDivPure >= 0.0f)
		yDerivative = (yDiv[xScan][yScan + 1] - yDiv[xScan][yScan]) / (gCellSize);
	else
		yDerivative = (yDiv[xScan][yScan] - yDiv[xScan ][yScan - 1]) / (gCellSize);


	result[xOrigin + yOrigin * strides] = xDerivative + yDerivative;
	
}


__global__ void PerformCorrectedIntegrationStep(float* originalData,  float* derivativeData, size_t stride, float timePassed)
{
	__shared__ float positiveSumPhaseA[gBlockSize][gBlockSize];
	__shared__ float negativeSumPhaseA[gBlockSize][gBlockSize];

	__shared__ float positiveSumPhaseB[4][4];
	__shared__ float negativeSumPhaseB[4][4];

	__shared__ float correctionFactor;

	// First we have to build the sum over all negative and positive elements over the derivative data.
	float localPosSum = 0.0f;
	float localNegSum = 0.0f;
	for(int j = threadIdx.y; j < gGridSizeInternal; j += gBlockSize)
		for(int i = threadIdx.x; i < gGridSizeInternal; i+= gBlockSize)
		{
			float candidate = derivativeData[(i + 1) + (j + 1) * stride];
			if (candidate > 0.0f)
				localPosSum += candidate;
			else
				localNegSum -= candidate;
		}
	positiveSumPhaseA[threadIdx.x][threadIdx.y] = localPosSum;
	negativeSumPhaseA[threadIdx.x][threadIdx.y] = localNegSum;

	__syncthreads();
	if ((threadIdx.x < 4) && (threadIdx.y < 4))
	{
		localPosSum = 0.0f;
		localNegSum = 0.0f;
		for (int j = threadIdx.y; j < gBlockSize; j += 4)
			for (int i = threadIdx.x; i < gBlockSize; i += 4)
			{
				localPosSum += positiveSumPhaseA[i][j];
				localNegSum += negativeSumPhaseA[i][j];
			}
		positiveSumPhaseB[threadIdx.x][threadIdx.y] = localPosSum;
		negativeSumPhaseB[threadIdx.x][threadIdx.y] = localNegSum;

	}
	__syncthreads();
	if ((threadIdx.x == 0) && (threadIdx.y == 0))
	{
		localPosSum = 0.0f;
		localNegSum = 0.0f;
		for (int j = 0; j < 4; ++j)
			for (int i = 0; i < 4; ++i)
			{
				localPosSum += positiveSumPhaseB[i][j];
				localNegSum += negativeSumPhaseB[i][j];
			}

		correctionFactor = (localNegSum - localPosSum) / (localPosSum + localNegSum);
		if (isnan(correctionFactor))
			correctionFactor = 0.0f;
		
	}

	__syncthreads();


	
	// Now the read integration part.
	for (int j = threadIdx.y; j < gGridSizeInternal; j += gBlockSize)
		for (int i = threadIdx.x; i < gGridSizeInternal; i += gBlockSize)
		{
			float candidate = derivativeData[(i + 1) + (j + 1) * stride];
			float baseValue = originalData[(i + 1) + (j + 1) * stride];
			
			if (candidate > 0.0f)
				baseValue += timePassed * (1.0f + correctionFactor) * candidate;
			else
				baseValue += timePassed * (1.0f - correctionFactor) * candidate;
			
			baseValue = fmaxf(0.0f, baseValue);
			originalData[(i + 1) + (j + 1) * stride] = baseValue;

		}	
}

void ContinuityEquationSolver::IntegrateEquation(float timePassed, FloatArray density,  FloatArray timeToDestination, DataBase* dataBase)
{

	UnsignedArray wallData = dataBase->GetWallData();
	// First we need the gradient of the iconal equation.
	m_gradientIconal.ComputeGradient(timeToDestination, wallData);

	FloatArray gradX = m_gradientIconal.GetXComponent();
	FloatArray gradY = m_gradientIconal.GetYComponent();
	assert(gradX.m_stride == gradY.m_stride);
	assert(gradX.m_stride == density.m_stride);
	assert(gradX.m_stride == m_derivativeBuffer.m_stride);
	assert(gradX.m_stride == wallData.m_stride);

	bool isTerminated = false;
	float localTimeStep;
	while(!isTerminated)
	{
		if (timePassed > gMaximumStepsizeContinuitySolver)
		{
			localTimeStep = gMaximumStepsizeContinuitySolver;
			timePassed -= gMaximumStepsizeContinuitySolver;
		}
		else
		{
			isTerminated = true;
			localTimeStep = timePassed;
		}

		GetTimeDerivative CUDA_DECORATOR_LOGIC (gradX.m_stride, density.m_array, gradX.m_array, gradY.m_array, m_derivativeBuffer.m_array);
		PerformCorrectedIntegrationStep <<<1,dim3(gBlockSize, gBlockSize)>>>(density.m_array, m_derivativeBuffer.m_array, m_derivativeBuffer.m_stride, localTimeStep);
		
		// IntegrateCuda CUDA_DECORATOR_LOGIC(localTimeStep, gradX.m_stride, density.m_array, gradX.m_array, gradY.m_array, m_resultBuffer.m_array);
		// IntegrateCuda CUDA_DECORATOR_LOGIC(localTimeStep, gradX.m_stride, m_resultBuffer.m_array, gradX.m_array, gradY.m_array, density.m_array);

		
	}

		
}


