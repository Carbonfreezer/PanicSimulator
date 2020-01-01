#pragma once

#include "TgaReader.h"
#include "TransferHelper.h"
#include "MemoryStructs.h"

// This class contains the density of the people. It manages all the
// spawn and target information things. Later on it will also contain the
// integrator of the continuity equation.
class DensityManager
{

public:
	

	// Sets up all the data.
	void InitializeManager(const char* spawnAreaFile, const char* targetAreaFile);

	// Asks for the target Area (Needed also for the iconal solver.)
	UnsignedArray GetTargetArea() { return m_targetArea; }

	FloatArray GetDensityField() { return m_density; }

	// Enforces the current information on spawn, target and wall information (handed over)
	void EnforceBoundaryConditions(UnsignedArray wallInformation);

	// Visualizes the current density information.
	void GenerateDensityVisualization(uchar4* textureMemory);

	// To be implemented later on.
	void UpdateDensityField(float timePassed, FloatArray eikonalSolution, FloatArray velocity){};

private:

	UnsignedArray m_targetArea;
	FloatArray m_spawnArea;

	FloatArray m_density;
	
	TransferHelper m_transferHelper;

	TgaReader m_spawnAreaReader;
	TgaReader m_targetAreaReader;
	
};
