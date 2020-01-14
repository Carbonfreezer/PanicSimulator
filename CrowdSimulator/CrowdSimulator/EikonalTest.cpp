#include "EikonalTest.h"
#include "DataBase.h"
#include <vector_functions.hpp>
#include "VisualizationHelper.h"


void EikonalTest::ToolSystem(DataBase* dataBase)
{
	m_velocityManager.GenerateVelocityField();
	m_densityManager.InitializeManager(dataBase);
	m_solver.PrepareSolving();
}

void EikonalTest::FreeResources()
{
	m_velocityManager.FreeResources();
	m_densityManager.FreeResources();
	m_solver.FreeResources();
}

void EikonalTest::UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase)
{
	m_densityManager.EnforceBoundaryConditions(dataBase, timePassedInSeconds);
	FloatArray densityField = m_densityManager.GetDensityField();
	m_velocityManager.UpdateVelocityField(densityField,dataBase);

	FloatArray velocityField = m_velocityManager.GetVelocityField();
	m_solver.SolveEquation( velocityField,  dataBase);
	
	
	FloatArray timeField = m_solver.GetTimeField();

	VisualizationHelper::VisualizeScalarField(timeField, m_maximumTime, deviceMemory);
	VisualizationHelper::VisualizeIsoLines(timeField, m_isoLineDistance,  deviceMemory);
}
