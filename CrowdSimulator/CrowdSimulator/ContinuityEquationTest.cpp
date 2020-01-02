#include "ContinuityEquationTest.h"
#include "DataBase.h"
#include <cmath>
#include "VisualizationHelper.h"

void ContinuityEquationTest::PrepareTest(int eikonalMode)
{
	m_solver.PrepareSolver();



	switch(eikonalMode)
	{
	case 0:
		m_pseudoIconalData =  TransferHelper::BuildHorizontalGradient(100.0f, 0);
		break;
	case 1:
		m_pseudoIconalData = TransferHelper::BuildHorizontalGradient(100.0f, 1);
		break;
	case 2:
		m_pseudoIconalData = TransferHelper::BuildVerticalGradient(100.0f, 0);
		break;
	case 3:
		m_pseudoIconalData = TransferHelper::BuildVerticalGradient(100.0f, 1);
		break;
	case 4:
		m_pseudoIconalData = TransferHelper::BuildRadialGradient(50.0f, 0);
		break;
	case 5:
		m_pseudoIconalData = TransferHelper::BuildRadialGradient(50.0f, 1);
		break;
	}
	
	m_velocityData = TransferHelper::UpfronFilledValue(gMaximumWalkingVelocity);
	m_densityData = TransferHelper::ReserveFloatMemory();

}

void ContinuityEquationTest::UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase)
{
	m_solver.IntegrateEquation(timePassedInSeconds, m_densityData, m_velocityData, m_pseudoIconalData, dataBase);
	VisualizationHelper::VisualizeScalarField(m_densityData, gMaximumDensity, deviceMemory);
}

void ContinuityEquationTest::ToolSystem(DataBase* dataBase)
{
	// Copy over the density data.
	TransferHelper::CopyDataFromTo(dataBase->GetInitialDensityData(), m_densityData);
}

