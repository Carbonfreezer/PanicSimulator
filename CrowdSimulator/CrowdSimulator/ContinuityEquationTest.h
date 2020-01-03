#pragma once
#include "LogicClass.h"
#include "ContinuityEquationSolver.h"



/**
 * \brief Uses the Density data and Wall data. and runs a continuity test.
 */
class ContinuityEquationTest : public LogicClass
{
public:
	void PrepareTest(int eikonalMode);

	/**
	 * \brief Is called for update and effectively rendering.
	 * \param deviceMemory The texture memory we write our data to.
	 * \param timePassedInSeconds The time in seconds passed since last update.
	 * \param dataBase The database with all the relevant information in.
	 */
	virtual void UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase);

	/**
	 * \brief This is called after all the graphics stuff has been initialized before the first update system but only once.
	 * \param dataBase The data base with all the formal stuff in there.
	 */
	virtual void ToolSystem(DataBase* dataBase);

	
private:
	
	FloatArray m_pseudoIconalData;
	FloatArray m_velocityData;
	FloatArray m_densityData;
	
	ContinuityEquationSolver m_solver;

	
};
