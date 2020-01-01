#pragma once
#include <surface_types.h>
#include "TransferHelper.h"
#include "MemoryStructs.h"

class IconalSolver
{
public:
	
	// Visualizes the outcome of the simulation.
	void VisualizeOutcome(uchar4* destinationMemory, float maxTimeAssumed, float distanceBetweenIsoLines);

	// Prepare iterating.
	void PrepareSolving();

	// Performs n iteration steps.
	void PerformIterations(int outerIterations, FloatArray velocityField, UnsignedArray targetInformation);

	// Asks for the time field.
	FloatArray GetTimeField() { return m_timeArray[m_usedDoubleBuffer]; }

private:
	// Which double buffer do we use?
	int m_usedDoubleBuffer;
	FloatArray m_timeArray[2];

	TransferHelper m_transferHelper;
};
