#include "IconalTest.h"
#include "DataBase.h"
#include <vector_functions.hpp>


void IconalTest::ToolSystem(DataBase* dataBase)
{
	m_velocityManager.GenerateVelocityField();
	m_densityManager.InitializeManager(dataBase);
	m_solver.PrepareSolving();
}

void IconalTest::UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase)
{
	m_densityManager.EnforceBoundaryConditions(dataBase);
	FloatArray densityField = m_densityManager.GetDensityField();
	m_velocityManager.UpdateVelocityField(densityField,dataBase);

	FloatArray velocityField = m_velocityManager.GetVelocityField();
	m_solver.PerformIterations(m_numOfGridIterations, velocityField,  dataBase);
	
	
	FloatArray timeField = m_solver.GetTimeField();

	m_visualizer.VisualizeScalarField(timeField, m_maximumTime, deviceMemory);
	m_visualizer.VisualizeIsoLines(timeField, m_isoLineDistance,  deviceMemory);
}
