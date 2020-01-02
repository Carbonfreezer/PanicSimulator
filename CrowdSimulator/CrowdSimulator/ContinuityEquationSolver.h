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
	// Does an initial prepare of the solver.
	void PrepareSolver();

	// Integrate the continuity equation and modifies the density field.
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
	
};