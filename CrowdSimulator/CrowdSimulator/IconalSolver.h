#pragma once
#include <surface_types.h>
#include "TgaReader.h"
#include "TransferHelper.h"
#include "LogicClass.h"


class IconalSolver
{
public:

	// Visualizes the outcome of the simulation.
	void VisualizeOutcome(uchar4* destinationMemory, float maxTimeAssumed, float distanceBetweenIsoLines);

	// Prepare iterating.
	void PrepareSolving();

	// Performs n iteration steps.
	void PerformIterations(int outerIterations, float* velocityField, size_t velocityStride, unsigned int* targetAreaInformation, size_t targetAreaStride);

	// Asks for the time field.
	float* GetTimeField(size_t& timeStride);

private:
	// Which double buffer do we use?
	int m_usedDoubleBuffer;
	// Contains the double buffered time, where the iterator is working on (Device memory).
	float* m_bufferTime[2];
	// The stride for the time buffer.
	size_t m_timeStride;
	


	TransferHelper m_transferHelper;
};
