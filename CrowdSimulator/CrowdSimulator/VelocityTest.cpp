#include "VelocityTest.h"

void VelocityTest::PrepareTest(const char* densityFile, const char* wallFile, float isoLineDistance)
{
	m_densityFile.ReadFile(densityFile);
	m_densityData = m_helper.UploadPictureAsFloat(&m_densityFile, 0.0f, 0.0f, gMaximumDensity);
	m_velocityManager.SetWallFile(wallFile);
	m_velocityManager.GenerateVelocityField();
	m_isoLineDistance = isoLineDistance;
}

void VelocityTest::UpdateSystem(uchar4* deviceMemory, double timePassedInSeconds)
{
	m_velocityManager.UpdateVelocityField(m_densityData);
	m_velocityManager.GenerateVelocityVisualization(deviceMemory, m_isoLineDistance);
}
