#include "VelocityTest.h"
#include "DataBase.h"


void VelocityTest::ToolSystem(DataBase* dataBase)
{
	m_velocityManager.GenerateVelocityField();
}

void VelocityTest::UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase)
{
	// We can take the initial density here, because it is not updated.
	m_velocityManager.UpdateVelocityField(dataBase->GetInitialDensityData(), dataBase);
	m_velocityManager.GenerateVelocityVisualization(deviceMemory, gMaximumWalkingVelocity / m_strideOnGradient);
}
