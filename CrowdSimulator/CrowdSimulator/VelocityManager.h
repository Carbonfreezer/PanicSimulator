#pragma once
#include "MemoryStructs.h"
#include <channel_descriptor.h>

class DataBase;

// 

/**
 * \brief This class is responsible to map the density to the velocity and provide a velocity scalaer field.
 * Uses the wall information from the data base
 */
class VelocityManager
{
public:

	/**
	 * \brief Generates the internal representation of the velocity field.
	 */
	void GenerateVelocityField();


	/**
	 * \brief Frees the required resources.
	 */
	void FreeResources();

	
	/**
	 * \brief  Updates the velocity field with the density field handed over.
	 * This is not an incremental process.
	 * \param density The density of the persons we generate the velocity from.
	 * \param dataBase The database with the boundary conditions.
	 */
	void UpdateVelocityField(FloatArray density, DataBase* dataBase);

	
	/**
	 * \brief Obtains the velocity information. 
	 * \return The contained velocity field.
	 */
	FloatArray GetVelocityField() { return m_velocityField; }
	

	
	/**
	 * \brief Generates a visualization of the velocity field. For debug and test purposes only.
	 * \param textureMemory The texture memory we paint into.
	 * \param isoLineDistance The distances we apply between iso lines.
	 */
	void GenerateVelocityVisualization(uchar4* textureMemory, float isoLineDistance);

private:
	FloatArray m_velocityField;

};
