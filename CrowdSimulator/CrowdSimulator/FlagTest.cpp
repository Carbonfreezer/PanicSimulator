#include "FlagTest.h"
#include  "DataBase.h"
#include <vector_functions.hpp>
#include "VisualizationHelper.h"


void FlagTest::UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase)
{
	VisualizationHelper::VisualizeScalarField(dataBase->GetInitialDensityData(), gMaximumDensity, deviceMemory);
	VisualizationHelper::MarcColor(dataBase->GetWallData(), deviceMemory, make_uchar4(255, 255, 255, 255));
}
