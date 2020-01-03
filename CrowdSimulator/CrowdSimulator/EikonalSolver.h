#pragma once
#include <surface_types.h>
#include "MemoryStructs.h"

class DataBase;

/**
 * \brief Solves the eikonal equation.
 * Uses the target information
 */
class EikonalSolver
{
public:
	
	// Visualizes the outcome of the simulation.
	void VisualizeOutcome(uchar4* destinationMemory, float maxTimeAssumed, float distanceBetweenIsoLines);

	// Prepare iterating.
	void PrepareSolving();

	// Solves the equation.
	void SolveEquation(FloatArray velocityField, DataBase* dataBase );

	// Asks for the time field.
	FloatArray GetTimeField() { return m_timeArray[1]; }

private:
	// Which double buffer do we use?
	FloatArray m_timeArray[2];

};
