#pragma once

#include "MemoryStructs.h"
#include "ContinuityEquationSolver.h"
#include <channel_descriptor.h>

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
	
	/**
	 * \brief Sets up all the data. Uses the initial density to fill the density field. 
	 * \param dataBase all the static data.
	 */
	void InitializeManager(DataBase* dataBase);

	
	/**
	 * \brief Gets the density field.
	 * \return Current density field.
	 */
	FloatArray GetDensityField() { return m_continuitySolver.GetCurrentDensityField(); }

	/**
	 * \brief Enforces the current information on spawn, target and wall information (handed over)
	 * \param dataBase Contains all the boundary conditions.
	 * \param timePassed The time passed needed for the spawning.
	 */
	void EnforceBoundaryConditions(DataBase* dataBase, float timePassed);

	// 
	/**
	 * \brief Visualizes the current density information. For test purposes only. 
	 * \param textureMemory Memory to draw into.
	 */
	void GenerateDensityVisualization(uchar4* textureMemory);

	
	/**
	 * \brief Performs the integration of the continuity equation.
	 * \param timePassed The time integration step size.
	 * \param timeField The time field that comes from the Eikonal Solver.
	 * \param dataBase The database for the boundary conditions.
	 */
	void UpdateDensityField(float timePassed, FloatArray timeField,  DataBase* dataBase);


	/**
	 * \brief Gets called when the density should be put back to the initial condition.
	 * \param data_base The databdase with all the information.
	 */
	void ResetDensity(DataBase* dataBase);

private:

	ContinuityEquationSolver m_continuitySolver;
	
};
