#include "IsoLineTest.h"
#include "DataBase.h"



void IsoLineTest::UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase)
{
	m_transferHelper.VisualizeScalarField(dataBase->GetInitialDensityData(), gMaximumDensity, deviceMemory);
	m_transferHelper.VisualizeIsoLines(dataBase->GetInitialDensityData(), gMaximumDensity / 15.0f,  deviceMemory);
}
