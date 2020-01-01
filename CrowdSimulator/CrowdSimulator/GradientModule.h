#pragma once
#include <surface_types.h>
#include "TransferHelper.h"

class GradientModule
{
public:
	GradientModule() { m_gradientResultX = m_gradientResultY = NULL; }
	
	void PreprareModule();

	// Gets the gradient.
	void ComputeGradient(float* inputField, size_t inputStride, unsigned int* wallField, size_t wallStride);


	// Selectively computes the x derivative (needed for divergence).
	void ComputeXDerivative(float* inputField, size_t inputStride, unsigned int* wallField, size_t wallStride);
	// Same for the y derivative.
	void ComputeYDerivative(float* inputField, size_t inputStride, unsigned int* wallField, size_t wallStride);


	
	float* GetXComponent(size_t& stride)
	{
		stride = m_gradientStride;
		return m_gradientResultX;
	}
	
	float* GetYComponent(size_t& stride)
	{
		stride = m_gradientStride;
		return m_gradientResultY;
	}

	void VisualizeField(int component, float maxExepctedValue, uchar4* textureMemory);
	
private:
	float* m_gradientResultX;
	float* m_gradientResultY;
	size_t m_gradientStride;
	TransferHelper m_transferHelper;
};
