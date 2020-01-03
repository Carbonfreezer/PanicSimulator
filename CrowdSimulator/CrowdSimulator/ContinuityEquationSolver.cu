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
	m_specialXDerivative.PreprareModule();
	m_specialYDerivative.PreprareModule();

	m_premultipliedGradientX = TransferHelper::ReserveFloatMemory();
	m_premultipliedGradientY = TransferHelper::ReserveFloatMemory();

	m_blockedElements = TransferHelper::ReserveUnsignedMemory();
}

__global__ void Multiply(float* gradientXSource, float* gradientYSource, float* gradientXDestination, float* gradientYDestination, 
		size_t gradientStride, float* densityArray, size_t densityStride, 
	float* velocityArray, size_t velocityStride)
{
	// We keep tack of the pixel  we are responsible for.
	int xOrigin = threadIdx.x + gBlockSize * blockIdx.x + 1;
	int yOrigin = threadIdx.y + gBlockSize * blockIdx.y + 1;

	float density = densityArray[xOrigin + yOrigin * densityStride];
	float velocity = velocityArray[xOrigin + yOrigin * velocityStride];
	float factor = density * velocity * velocity;

	gradientXDestination[xOrigin + yOrigin * gradientStride] = gradientXSource[xOrigin + yOrigin * gradientStride] * factor;
	gradientYDestination[xOrigin + yOrigin * gradientStride] = gradientYSource[xOrigin + yOrigin * gradientStride] * factor;
}

__global__ void IntegrateEuler(float timePassed, float* density, size_t densityStride, float* xComponent, float* yComponent, size_t componentStride)
{
	// We keep tack of the pixel  we are responsible for.
	int xOrigin = threadIdx.x + gBlockSize * blockIdx.x + 1;
	int yOrigin = threadIdx.y + gBlockSize * blockIdx.y + 1;


	float sum = xComponent[xOrigin + yOrigin * componentStride] + yComponent[xOrigin + yOrigin * componentStride];
	
	float accumulator = density[xOrigin + yOrigin * densityStride];
	accumulator += timePassed * sum;
	accumulator = fmaxf(accumulator, 0.0f);
	accumulator = fminf(accumulator, gMaximumDensity);

	density[xOrigin + yOrigin * densityStride] = accumulator;
	
}

__global__ void LogicalOr(unsigned int* dataA, size_t strideA, unsigned int* dataB, size_t strideB, unsigned int* destination, size_t destinationStride)
{
	int xOrigin = threadIdx.x + gBlockSize * blockIdx.x + 1;
	int yOrigin = threadIdx.y + gBlockSize * blockIdx.y + 1;

	destination[xOrigin + yOrigin * destinationStride] = dataA[xOrigin + yOrigin * strideA] || dataB[xOrigin + yOrigin * strideB];
}

void ContinuityEquationSolver::IntegrateEquation(float timePassed, FloatArray density, FloatArray velocity,
	FloatArray timeToDestination, DataBase* dataBase)
{

	// First we need the gradient of the iconal equation.
	m_gradientIconal.ComputeGradient(timeToDestination, dataBase->GetWallData());

	// We need a logical or for the wall and the despawn data for the divergence computation later on.
	// We do this in every update, because we do not want to rely on the fact that the data in the base remains constant.
	LogicalOr CUDA_DECORATOR_LOGIC (dataBase->GetWallData().m_array, dataBase->GetWallData().m_stride, dataBase->GetDespawnData().m_array, dataBase->GetDespawnData().m_stride,
		m_blockedElements.m_array, m_blockedElements.m_stride);
	
	FloatArray gradX = m_gradientIconal.GetXComponent();
	FloatArray gradY = m_gradientIconal.GetYComponent();

	assert(gradX.m_stride == gradY.m_stride);
	assert(gradX.m_stride == m_premultipliedGradientX.m_stride);
	assert(m_premultipliedGradientY.m_stride == m_premultipliedGradientX.m_stride);
	// Hard limit iterations.
	if (timePassed > 10.0f * gMaximumStepsizeContinuitySolver)
		timePassed = 10.0f * gMaximumStepsizeContinuitySolver;
	
	bool endOfIterationReached = false;
	do
	{
		float timeStep;

		if (timePassed <= gMaximumStepsizeContinuitySolver)
		{
			timeStep = timePassed;
			endOfIterationReached = true;
		}
		else
		{
			timeStep = gMaximumStepsizeContinuitySolver;
			timePassed -= timeStep;
		}

		// Pre multiply the gradient with the density and velocity fields.
		Multiply CUDA_DECORATOR_LOGIC(gradX.m_array, gradY.m_array, m_premultipliedGradientX.m_array, m_premultipliedGradientY.m_array,
			gradX.m_stride, density.m_array, density.m_stride, velocity.m_array, velocity.m_stride);


		// Now we can compute the gradients of both fields needed for the final integration step.
		m_specialXDerivative.ComputeGradientXForDivergence(m_premultipliedGradientX, m_blockedElements);
		m_specialYDerivative.ComputeGradientYForDivergence(m_premultipliedGradientY, m_blockedElements);

		// We need the x component of the x derivative and the y component of the y derivative.
		FloatArray xComponent = m_specialXDerivative.GetXComponent();
		FloatArray yComponent = m_specialYDerivative.GetYComponent();

		// Now we can integrate the equation of motion.
		assert(xComponent.m_stride == yComponent.m_stride);

		
		IntegrateEuler CUDA_DECORATOR_LOGIC(timeStep, density.m_array, density.m_stride, xComponent.m_array, yComponent.m_array, xComponent.m_stride);

		
	} while (!endOfIterationReached);

	

	
}

