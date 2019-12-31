#pragma once
#include <surface_types.h>
#include "TgaReader.h"
#include "TransferHelper.h"
#include "LogicClass.h"


class IconalSolver
{
public:
	// Sets the information where the walls are.
	void SetWallFile(const char* wallInformation);
	// Sets the information where the target areas are.
	void SetTargetArea(const char* targetInformation);

	// Visualizes the outcome of the simulation.
	void VisualizeOutcome(uchar4* destinationMemory, float maxTimeAssumed, float distanceBetweenIsoLines);

	// Prepare iterating.
	void PrepareSolving();

	// Performs n iteration steps.
	void PerformIterations(int outerIterations);

	// Asks for the time field.
	float* GetTimeField(size_t& timeStride);

private:
	// Which double buffer do we use?
	int m_usedDoubleBuffer;
	// Contains the double buffered time, where the iterator is working on (Device memory).
	float* m_bufferTime[2];
	// The stride for the time buffer.
	size_t m_timeStride;
	// The original wall field.
	unsigned int* m_wallInformation;
	size_t m_wallStride;
	// The original informaton with the target area.
	unsigned int* m_targetAreaInformation;
	size_t m_targetStride;
	// The area with the velocity field we have.
	float* m_velocityField;
	size_t m_velocityStride;

	TgaReader m_wallPicture;
	TgaReader m_targetPicture;

	TransferHelper m_transferHelper;
};
