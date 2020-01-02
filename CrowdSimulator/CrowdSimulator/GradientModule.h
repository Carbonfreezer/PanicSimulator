#pragma once
#include <surface_types.h>
#include "TransferHelper.h"
#include "MemoryStructs.h"

class DataBase;

/**
 * \brief Computes the gradient. For the full gradient it assumes right sided / left sided or
 * up and down difference quotient in the border cases. In the divergence cases it uses the minimum of this
 * and assuming zero for the border in central difference quotient.
 *
 * Wall information is used.
 */
class GradientModule
{
public:

	// Has to be called upfront once.
	void PreprareModule();
	// Gets the gradient.
	void ComputeGradient(FloatArray inputField, DataBase* dataBase);
	// Assumes zeroes at the undefined areas. Needed for the continuum equation and the divergence computation.
	void ComputeGradientXForDivergence(FloatArray inputField, DataBase* dataBase);
	// Assumes zeroes at the undefined areas. Needed for the continuum equation and the divergence computation.
	void ComputeGradientYForDivergence(FloatArray inputField, DataBase* dataBase);


	
	FloatArray GetXComponent() { return m_gradientResultX; }	
	FloatArray GetYComponent() { return m_gradientResultY; }
	

	void VisualizeField(int component, float maxExepctedValue, uchar4* textureMemory);
	
private:
	FloatArray m_gradientResultX;
	FloatArray m_gradientResultY;

	TransferHelper m_transferHelper;
};
