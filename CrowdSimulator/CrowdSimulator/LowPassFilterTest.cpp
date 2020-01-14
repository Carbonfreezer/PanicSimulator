#include "LowPassFilterTest.h"
#include "TransferHelper.h"
#include "DataBase.h"
#include "VisualizationHelper.h"

void LowPassFilterTest::ToolSystem(DataBase* dataBase)
{
	m_filterA.PrepareModule();
	m_filterB.PrepareModule();
}

void LowPassFilterTest::FreeResources()
{
	m_filterA.FreeResources();
	m_filterB.FreeResources();
}

void LowPassFilterTest::UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase)
{
	m_filterA.Filter(dataBase->GetInitialDensityData(), dataBase->GetWallData());
	if (m_numOfIterations == 1)
	{
		VisualizationHelper::VisualizeScalarField(m_filterA.GetResult(), gMaximumDensity, deviceMemory);
	}
	else
	{
		m_filterB.Filter(m_filterA.GetResult(), dataBase->GetWallData());
		VisualizationHelper::VisualizeScalarField(m_filterB.GetResult(), gMaximumDensity, deviceMemory);
	}
}
	


