#pragma once
#include "GradientModule.h"
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

	// 
	/**
	 * \brief Integrate the continuity equation and modifies the density field. 
	 * \param timePassed The time step to integrate into the future.
	 * \param density The density of the persons that gets updated.
	 * \param velocity The velocity with which the persons walk.
	 * \param timeToDestination The time field to destination that comes from the eikonal solver.
	 * \param dataBase The database with the boundary conditions.
	 */
	void IntegrateEquation(float  timePassed, FloatArray density,  FloatArray timeToDestination, DataBase* dataBase);

private:
	// This contains the gradient of the eikonal equation.
	GradientModule m_gradienEikonal;
	// The buffer with the computed derivative Data..
	FloatArray m_derivativeBuffer;

	};