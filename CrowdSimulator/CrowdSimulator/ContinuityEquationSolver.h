#pragma once
#include "TimeToVelocityMapper.h"
class DataBase;

/**
 * \brief Solves the continuity equation over time.
 * It uses the wall information.
 */
class ContinuityEquationSolver
{
public:
	/**
	 * \brief Does an initial prepare of the solver. 
	 */
	void PrepareSolver();


	/**
	 * \brief Frees all the resources from the GPU.
	 */
	void FreeResources();

	// 
	/**
	 * \brief Integrate the continuity equation and modifies the density field. 
	 * \param timePassed The time step to integrate into the future.
	 * \param velocity The velocity with which the persons walk.
	 * \param timeToDestination The time field to destination that comes from the eikonal solver.
	 * \param dataBase The database with the boundary conditions.
	 */
	void IntegrateEquation(float  timePassed,  FloatArray timeToDestination, DataBase* dataBase);


	/**
	 * \brief Asks for the currently used density field.
	 * \return Up to date density field.
	 */
	FloatArray GetCurrentDensityField() { return m_currentDensityField[m_currentFieldValid]; }
	
private:
	// This contains the gradient of the eikonal equation.
	TimeToVelocityMapper m_timeToVelocity;
	// The current buffer flag for the velocity field.
	int m_currentFieldValid;
	// The velocity field we process.
	FloatArray m_currentDensityField[2];

	};