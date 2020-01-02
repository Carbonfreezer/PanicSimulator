#include "GradientTest.h"
#include "DataBase.h"
#include <cassert>
#include <cmath>
#include <vector_functions.hpp>


void GradientTest::PrepareTest(int visualizationDecision)
{
	assert(m_densityInformation.m_array == NULL);
	m_visualizationDecision = visualizationDecision;

	m_densityInformation = TransferHelper::BuildRadialGradient(100.0f, 0);

	m_gradientModule.PreprareModule();
}

void GradientTest::UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase)
{
	m_gradientModule.ComputeGradient(m_densityInformation, dataBase);
	m_gradientModule.VisualizeField(m_visualizationDecision, 3.0f, deviceMemory);
}
