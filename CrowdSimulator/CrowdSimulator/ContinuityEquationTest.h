#pragma once
#include "LogicClass.h"
#include "TgaReader.h"
#include "TransferHelper.h"
#include "ContinuityEquationSolver.h"


class ContinuityEquationTest : public LogicClass
{
public:
	void PrepareTest(int eikonalMode, const char* densityFile, const char* wallFile);

	virtual void UpdateSystem(uchar4* deviceMemory, double timePassedInSeconds);

	
private:
	TgaReader m_densityFile;
	TgaReader m_wallFile;
	

	FloatArray m_pseudoIconalData;
	FloatArray m_densityData;
	FloatArray m_velocityData;
	UnsignedArray m_wallData;
	
	TransferHelper m_helper;
	ContinuityEquationSolver m_solver;

	
};
