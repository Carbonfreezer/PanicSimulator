#pragma once
#include "GradientModule.h"

class ContinuityEquationSolver
{
public:
	// Does an initial prepare of the solver.
	void PrepareSolver();

	// Integrate the continuity equation and modifies the density field.
	void IntegrateEquation(float  timePassed, float* densityField, size_t densityStride, float* velocityField, size_t velocityStride, float* iconalField, size_t iconalSolution);

private:
	// This contains the gradient of the iconal equation.
	GradientModule m_gradientIconal;
	// This is the module for the x derivative of the special field.
	GradientModule m_specialXDerivative;
	// The same for the y derivative.
	GradientModule m_specialYDerivative;
	
};