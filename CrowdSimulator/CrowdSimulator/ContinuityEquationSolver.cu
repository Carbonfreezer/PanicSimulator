#include "ContinuityEquationSolver.h"
#include <cassert>
#include "CudaHelper.h"
#include <device_launch_parameters.h>
#include <math.h>

void ContinuityEquationSolver::PrepareSolver()
{
	m_gradientIconal.PreprareModule();
	m_specialXDerivative.PreprareModule();
	m_specialYDerivative.PreprareModule();
}

__global__ void InplaceMultiply(float* gradientX, float* gradientY, size_t gradientStride, float* densityArray, size_t densityStride, float* velocityArray,
                    size_t velocityStride)
{
	// We keep tack of the pixel  we are responsible for.
	int xOrigin = threadIdx.x + gBlockSize * blockIdx.x + 1;
	int yOrigin = threadIdx.y + gBlockSize * blockIdx.y + 1;

	float density = densityArray[xOrigin + yOrigin * densityStride];
	float velocity = velocityArray[xOrigin + yOrigin * velocityStride];
	float factor = density * velocity * velocity;

	gradientX[xOrigin + yOrigin * gradientStride] *= factor;
	gradientY[xOrigin + yOrigin * gradientStride] *= factor;
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

void ContinuityEquationSolver::IntegrateEquation(float timePassed, FloatArray density, FloatArray velocity,
	FloatArray timeToDestination, UnsignedArray wallArray)
{
	// First we need the gradient of the iconal equation.
	m_gradientIconal.ComputeGradient(timeToDestination, wallArray);

	// Now we have to multiply both components by density and velocity squared.
	FloatArray gradX = m_gradientIconal.GetXComponent();
	FloatArray gradY = m_gradientIconal.GetYComponent();
	assert(gradX.m_stride == gradY.m_stride);

	InplaceMultiply CUDA_DECORATOR_LOGIC (gradX.m_array, gradY.m_array, gradX.m_stride, density.m_array, density.m_stride, velocity.m_array, velocity.m_stride);

	// Now we can compute the gradients of both fields needed for the final integration step.
	m_specialXDerivative.ComputeGradientXForDivergence(gradX, wallArray);
	m_specialYDerivative.ComputeGradientYForDivergence(gradY, wallArray);

	// We need the x component of the x derivative and the y component of the y derivative.
	FloatArray xComponent = m_specialXDerivative.GetXComponent();
	FloatArray yComponent = m_specialYDerivative.GetYComponent();

	// Now we can integrate the equation of motion.
	assert(xComponent.m_stride == yComponent.m_stride);
	IntegrateEuler CUDA_DECORATOR_LOGIC (timePassed, density.m_array, density.m_stride, xComponent.m_array, yComponent.m_array, xComponent.m_stride);
	
}

