#pragma once
#include "MemoryStructs.h"
#include <cstddef>


/**
 * \brief The result structure of the accumulator.
 */
struct AccuResult
{

	/**
	 * \brief The sum of all elememnts.
	 */
	float m_sum;

	/**
	 * \brief The maximum of all elememnts.
	 */
	float m_maximum;

};


/**
 * \brief Helper class to build the maximum and the sum over the float array.
 * Asking for the result forces a snychronization between GPU and CPU.
 */
class Accumulator
{
public:

	/**
	 * \brief Defaults pointer to zero.
	 */
	Accumulator() : m_accumulationResult(NULL) {}
	
	/**
	 * \brief Allocates the required memory.
	 */
	void ToolSystem();

	/**
	 * \brief Frees the resources that the system acquired.
	 */
	void FreeResources();

	/**
	 * \brief Incoves the processing on the GPU.
	 * \param fieldToProcess The array we want to get the results from.
	 */
	void ProcessField(FloatArray fieldToProcess);
	
	/**
	 * \brief Asks for the results of the previous invocation of ProcessField.
	 * Forces a synchronization with the GPU.
	 * \param sum The sum of elements we want to get.
	 * \param maximum The maximal value of the elements we got. 
	 */
	void GetResult(float& sum, float& maximum);

private:
	AccuResult* m_accumulationResult;

};
