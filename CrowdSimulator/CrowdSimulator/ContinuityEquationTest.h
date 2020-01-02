#pragma once
#include "LogicClass.h"
#include "TgaReader.h"
#include "TransferHelper.h"
#include "ContinuityEquationSolver.h"



/**
 * \brief Uses the Density data and Wall data.
 */
class ContinuityEquationTest : public LogicClass
{
public:
	void PrepareTest(int eikonalMode);

	virtual void UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase);

	virtual void ToolSystem(DataBase* dataBase);

	
private:
	
	FloatArray m_pseudoIconalData;
	FloatArray m_velocityData;
	FloatArray m_densityData;
	
	ContinuityEquationSolver m_solver;

	
};
