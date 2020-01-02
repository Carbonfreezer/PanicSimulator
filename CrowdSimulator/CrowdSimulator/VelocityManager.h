#pragma once
#include <surface_types.h>
#include "TransferHelper.h"
#include "MemoryStructs.h"
#include "VisualizationHelper.h"

class DataBase;

// 

/**
 * \brief This class is responsible to map the density to the velocity and provide a velocity scalaer field.
 * Uses the wall information from the data base
 */
class VelocityManager
{
public:

	
	// Generate velocity field.
	void GenerateVelocityField();

	// Updates the velocity field with the density field handed over.
	void UpdateVelocityField(FloatArray density, DataBase* dataBase);

	// Obtains the velocity information.
	FloatArray GetVelocityField() { return m_velocityField; }
	

	// Generates a visualization of the velocity field.
	void GenerateVelocityVisualization(uchar4* textureMemory, float isoLineDistance);
	
private:
	FloatArray m_velocityField;
	VisualizationHelper m_visualizer;

};
