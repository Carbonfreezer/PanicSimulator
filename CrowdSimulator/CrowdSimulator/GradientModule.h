#pragma once
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

	
	/**
	 * \brief Has to be called upfront once. 
	 */
	void PreprareModule();

	
	
	/**
	 * \brief Gets the gradient. 
	 * \param inputField Scalar field to compute gradient from.
	 * \param blockedElements Indicates which elements not to use for the gradient computation.
	 */
	void ComputeGradient(FloatArray inputField, UnsignedArray blockedElements);



	/**
	 * \brief Only computes the x component of the gradient. Need for divergence computation. Pays special attention to the boundary
	 * conditions that always the smaller result from right/left sided difference quotients and from central difference quotient
	 * assuming 0 on boundary values is chosen.
	 * \param inputField Field to compute x derivative from. 
	 *  \param blockedElements Indicates which elements not to use for the gradient computation.
	 */
	void ComputeGradientXForDivergence(FloatArray inputField, UnsignedArray blockedElements);

	
	/**
	 * \brief Only computes the y component of the gradient. Need for divergence computation. Pays special attention to the boundary
	 * conditions that always the smaller result from bottom/top sided difference quotients and from central difference quotient
	 * assuming 0 on boundary values is chosen.
	 * \param inputField Field to compute y derivative from.
	 * \param blockedElements Indicates which elements not to use for the gradient computation.
	 */
	void ComputeGradientYForDivergence(FloatArray inputField, UnsignedArray blockedElements);


	
	/**
	 * \brief Gets the x component of the gradient
	 * \return x component of gradient
	 */
	FloatArray GetXComponent() { return m_gradientResultX; }	


	/**
	 * \brief Gets the y component of the gradient
	 * \return y component of gradient
	 */
	FloatArray GetYComponent() { return m_gradientResultY; }
	

	/**
	 * \brief Visualizes the gradient of the field. For debug purposes only.
	 * \param component Which component to show x : 0 y : 1
	 * \param maxExepctedValue The maximum value we expect.
	 * \param textureMemory The texture memory to paint into.
	 */
	void VisualizeField(int component, float maxExepctedValue, uchar4* textureMemory);
	
private:
	FloatArray m_gradientResultX;
	FloatArray m_gradientResultY;

};
