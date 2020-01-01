#include "ContinuityEquationTest.h"
#include <cmath>

void ContinuityEquationTest::PrepareTest(int eikonalMode, const char* densityFile, const char* wallFile)
{
	m_solver.PrepareSolver();



	switch(eikonalMode)
	{
	case 0:
		m_pseudoIconalData = m_helper.BuildHorizontalGradient(100.0f, 0);
		break;
	case 1:
		m_pseudoIconalData = m_helper.BuildHorizontalGradient(100.0f, 1);
		break;
	case 2:
		m_pseudoIconalData = m_helper.BuildVerticalGradient(100.0f, 0);
		break;
	case 3:
		m_pseudoIconalData = m_helper.BuildVerticalGradient(100.0f, 1);
		break;
	case 4:
		m_pseudoIconalData = m_helper.BuildRadialGradient(50.0f, 0);
		break;
	case 5:
		m_pseudoIconalData = m_helper.BuildRadialGradient(50.0f, 1);
		break;
	}
	
	
	m_densityFile.ReadFile(densityFile);
	m_densityData = m_helper.UploadPictureAsFloat(&m_densityFile, 0.0f, 0.0f, gMaximumDensity);


	m_velocityData = m_helper.UpfronFilledValue(gMaximumWalkingVelocity);

	m_wallFile.ReadFile(wallFile);

	m_wallData = m_helper.UploadPicture(&m_wallFile,0);
}

void ContinuityEquationTest::UpdateSystem(uchar4* deviceMemory, double timePassedInSeconds)
{
	m_solver.IntegrateEquation((float)timePassedInSeconds, m_densityData, m_velocityData, m_pseudoIconalData, m_wallData);
	m_helper.VisualizeScalarField(m_densityData, gMaximumDensity, deviceMemory);
}

