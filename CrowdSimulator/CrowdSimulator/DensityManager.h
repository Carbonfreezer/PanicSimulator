#pragma once

#include "TransferHelper.h"
#include "MemoryStructs.h"

class DataBase;

/**
 * \brief  This class contains the density of the people. It manages all the
	 spawn and target information things. Later on it will also contain the
	 integrator of the continuity equation.

	It uses spawn area, despawn area, wall and initial density.
 */
class DensityManager
{

public:
	
	// Sets up all the data. Uses the initial density to fill the density field.
	void InitializeManager(DataBase* dataBase);

	
	FloatArray GetDensityField() { return m_density; }

	// Enforces the current information on spawn, target and wall information (handed over)
	void EnforceBoundaryConditions(DataBase* dataBase);

	// Visualizes the current density information.
	void GenerateDensityVisualization(uchar4* textureMemory);

	// To be implemented later on.
	void UpdateDensityField(float timePassed, FloatArray eikonalSolution, DataBase* dataBase){};

private:

	FloatArray m_density;
	
};
