#pragma once
#include <surface_types.h>
#include "MemoryStructs.h"

class DataBase;


/**
 * \brief Helper struct to administrate the convergence internally.
 */
struct ConvergenceMeasure
{
	int m_hasConverged;
	int m_blocksCountedForConvergence;
	int m_blocksHaveConverged;
};


/**
 * \brief Solves the eikonal equation.
 * Uses the target information
 */
class EikonalSolver
{
public:
	
	/**
	 * \brief Prepare iterating. 
	 */
	void PrepareSolving();

	/**
	 * \brief Frees the allocated resources.
	 */
	void FreeResources();

	
	/**
	 * \brief Solves the equation. 
	 * \param velocityField Scalar velocity
	 * \param dataBase Data base used for targets and walls.
	 */
	void SolveEquation(FloatArray velocityField, DataBase* dataBase );

	/**
	 * \brief Asks for the time field.
	 * \return Time field returned.
	 */
	FloatArray GetTimeField() { return m_timeArray[1]; }
	
private:
	// Which double buffer do we use?
	FloatArray m_timeArray[2];
	// Struct kept in device memory for convergence measure.
	ConvergenceMeasure* m_convergence;

};
