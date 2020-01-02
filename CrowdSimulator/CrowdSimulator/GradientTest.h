#pragma once
#include "LogicClass.h"
#include "TransferHelper.h"
#include "GradientModule.h"
#include "MemoryStructs.h"


/**
 * \brief Uses wall data.
 */
class GradientTest : public LogicClass
{
public:

	// 0: x derivative, 1 : y derivative
	void PrepareTest(int visualizationDecision);

	virtual void UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase);

private:
	TransferHelper m_helper;

	FloatArray m_densityInformation;
	int m_visualizationDecision;
	
	GradientModule m_gradientModule;
	
};

