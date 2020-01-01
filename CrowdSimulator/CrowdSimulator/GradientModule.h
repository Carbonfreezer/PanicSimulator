#pragma once
#include <surface_types.h>
#include "TransferHelper.h"
#include "MemoryStructs.h"

class GradientModule
{
public:

	// Has to be called upfront once.
	void PreprareModule();
	// Gets the gradient.
	void ComputeGradient(FloatArray inputField, UnsignedArray wallField);

	
	FloatArray GetXComponent() { return m_gradientResultX; }	
	FloatArray GetYComponent() { return m_gradientResultY; }
	

	void VisualizeField(int component, float maxExepctedValue, uchar4* textureMemory);
	
private:
	FloatArray m_gradientResultX;
	FloatArray m_gradientResultY;

	TransferHelper m_transferHelper;
};
