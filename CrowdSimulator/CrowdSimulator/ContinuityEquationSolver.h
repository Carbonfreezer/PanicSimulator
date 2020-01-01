#pragma once
#include "GradientModule.h"

class ContinuityEquationSolver
{
public:
	// Does an initial prepare of the solver.
	void PrepareSolver();

	// Integrate the continuity equation and modifies the density field.
	void IntegrateEquation(float  timePassed, FloatArray density, FloatArray velocity, FloatArray timeToDestination, UnsignedArray wallArray);

	void DebugHack(float  timePassed, FloatArray density, FloatArray velocity, FloatArray timeToDestination, UnsignedArray wallArray, uchar4* pixel);

private:
	// This contains the gradient of the iconal equation.
	GradientModule m_gradientIconal;
	// This is the module for the x derivative of the special field.
	GradientModule m_specialXDerivative;
	// The same for the y derivative.
	GradientModule m_specialYDerivative;


	// HACK HACK HACK
	TransferHelper	m_helper;
	
};