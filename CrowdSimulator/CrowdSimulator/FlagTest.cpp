#include "FlagTest.h"
#include  "DataBase.h"
#include <vector_functions.hpp>


void FlagTest::UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase)
{
	m_transferHelper.VisualizeScalarField(dataBase->GetInitialDensityData(), gMaximumDensity, deviceMemory);
	m_transferHelper.MarcColor(dataBase->GetWallData(), deviceMemory, make_uchar4(255, 255, 255, 255));
}
