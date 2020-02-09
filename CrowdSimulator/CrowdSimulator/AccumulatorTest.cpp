#include "AccumulatorTest.h"
#include "TransferHelper.h"
#include <cstdio>

void AccumulatorTest::ToolSystem(DataBase* dataBase)
{
	// m_testCandidate = TransferHelper::UpfrontFilledValue(0.001f);
	m_testCandidate = TransferHelper::BuildRadialGradient(0.75f, 1);
	m_accuToTest.ToolSystem();
	m_accuToTest.ProcessField(m_testCandidate);

	float sum, max;
	m_accuToTest.GetResult(sum, max);

	printf("Sum: %f,Max: %f  \n", sum, max);
}

void AccumulatorTest::FreeResources()
{
	m_accuToTest.FreeResources();
	m_testCandidate.FreeArray();
}


