#pragma once
#include "MemoryStructs.h"
#include <channel_descriptor.h>


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
	 * \brief Gives information on saddle point, meaning that the indicated pixel represents a local maximum
	 * or minimum. The x component is encoded in the first bit, the y component in the second bit.
	 * \return 
	 */
	UnsignedArray GerExtremPointInformation() { return m_extremPoint; }

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

	UnsignedArray m_extremPoint;

};
