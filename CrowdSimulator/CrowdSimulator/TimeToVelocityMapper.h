#pragma once


#include "MemoryStructs.h"
#include <channel_descriptor.h>


class DataBase;

/**
 * \brief Computes the velocity from the time gradient For the full gradient it assumes right sided / left sided or
 * up and down difference quotient in the border cases.It also informs if the indicated sample point represents a local extremum.
 *
 * Wall information is used.
 */
class TimeToVelocityMapper
{
public:


	/**
	 * \brief Has to be called upfront once.
	 */
	void PreprareModule();

	/**
	 * \brief Frees all the reources from the graphics card.
	 */
	void FreeResources();


	/**
	 * \brief Gets the velocity.
	 * \param inputField Scalar field to compute gradient from.
	 * \param blockedElements Indicates which elements not to use for the gradient computation.
	 * \param targetElements The target elements where the velocity vector is forced to be zero. 
	 */
	void ComputeVelocity(FloatArray inputField, UnsignedArray blockedElements, UnsignedArray targetElements);



	/**
	 * \brief Gets the x component of the gradient
	 * \return x component of gradient
	 */
	FloatArray GetXComponent() { return m_velocityXResult; }


	/**
	 * \brief Gets the y component of the gradient
	 * \return y component of gradient
	 */
	FloatArray GetYComponent() { return m_velocityYResult; }



	/**
	 * \brief Gives information on saddle point, meaning that the indicated pixel represents a local maximum
	 * or minimum. The x component is encoded in the first bit, the y component in the second bit.
	 * \return
	 */
	UnsignedArray GerExtremPointInformation() { return m_extremPoint; }


private:
	FloatArray m_velocityXResult;
	FloatArray m_velocityYResult;

	UnsignedArray m_extremPoint;

};
