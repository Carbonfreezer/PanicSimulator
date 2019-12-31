#pragma once

#include "TgaReader.h"
#include "TransferHelper.h"

// This class contains the density of the people. It manages all the
// spawn and target information things. Later on it will also contain the
// integrator of the continuity equation.
class DensityManager
{

public:
	DensityManager();

	// Sets up all the data.
	void InitializeManager(const char* spawnAreaFile, const char* targetAreaFile);

	// Asks for the target Area (Needed also for the iconal solver.)
	unsigned int* GetTargetArea(size_t& targetStride)
	{
		targetStride = m_targetAreaStride;
		return m_targetAreaData;
	}

	float* GetDensityField(size_t& densityStride)
	{
		densityStride = m_densityStride;
		return m_densityBuffer[m_doubleBufferDensity];
	}

	// Enforces the current information on spawn, target and wall information (handed over)
	void EnforceBoundaryConditions(unsigned int* wallInformation, size_t wallStride);

	// Visualizes the current density information.
	void GenerateDensityVisualization(uchar4* textureMemory);

	// To be implemented later on.
	void UpdateDensityField(float timePassed, float* gradientEukonal, size_t strideGradienEukonal, float* velocityField, size_t strideVelocity){};

private:

	
	unsigned int* m_targetAreaData;
	size_t m_targetAreaStride;


	float* m_spawnAreaData;
	size_t m_spawnAreaStride;
	
	TransferHelper m_transferHelper;

	TgaReader m_spawnAreaReader;
	TgaReader m_targetAreaReader;

	int m_doubleBufferDensity;
	size_t m_densityStride;
	float* m_densityBuffer[2];
	
};
