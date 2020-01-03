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
	void IntegrateEquation(float  timePassed, FloatArray density, FloatArray velocity, FloatArray timeToDestination, DataBase* dataBase);

private:
	// This contains the gradient of the iconal equation.
	GradientModule m_gradientIconal;
	// This is the module for the x derivative of the special field.
	GradientModule m_specialXDerivative;
	// The same for the y derivative.
	GradientModule m_specialYDerivative;

	// The array with the premultiplied gradient.
	FloatArray m_premultipliedGradientX;
	FloatArray m_premultipliedGradientY;

	// The array with the blocked elements (logical or of wall and despawn).
	UnsignedArray m_blockedElements;
	
};