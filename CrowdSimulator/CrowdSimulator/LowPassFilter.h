#pragma once
#include "MemoryStructs.h"

class LowPassFilter
{
public:
	/**
	 * \brief Generates the internal structures to run the module.
	 */
	void PrepareModule();

	/**
	 * \brief Frees resources from the graphics card.
	 */
	void FreeResources();
	
	/**
	 * \brief Executes the filter on the area handed over.
	 * \param inputField The field that should get low pass filtered.
	 * \param blockedElements The elements that should not be taken into account (walls in our simuation)
	 */
	void Filter(FloatArray inputField, UnsignedArray blockedElements);


	/**
	 * \brief Gets the result of the filter.
	 * \return The result.
	 */
	FloatArray GetResult() { return m_result; }
	
private:
	FloatArray m_result;
};
