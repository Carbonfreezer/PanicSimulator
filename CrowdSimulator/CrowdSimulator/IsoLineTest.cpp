#include "IsoLineTest.h"
#include "DataBase.h"
#include "VisualizationHelper.h"


void IsoLineTest::UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase)
{
	VisualizationHelper::VisualizeScalarField(dataBase->GetInitialDensityData(), gMaximumDensity, deviceMemory);
	VisualizationHelper::VisualizeIsoLines(dataBase->GetInitialDensityData(), gMaximumDensity / 15.0f,  deviceMemory);
}
