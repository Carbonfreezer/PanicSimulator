#include "IconalTest.h"
#include <vector_functions.hpp>


void IconalTest::PrepareTest(const char* targetFile, const char* wallFile)
{
	m_numOfGridIterations = 0;
	m_velocityManager.SetWallFile(wallFile);
	m_velocityManager.GenerateVelocityField();

	m_densityManager.InitializeManager("Empty.tga", targetFile);

	m_solver.PrepareSolving();
}

void IconalTest::PrepareTest(const char* targetFile, const char* wallFile, const char* spawnFile)
{
	m_numOfGridIterations = 0;
	m_velocityManager.SetWallFile(wallFile);
	m_velocityManager.GenerateVelocityField();

	m_densityManager.InitializeManager(spawnFile, targetFile);

	m_solver.PrepareSolving();
}

void IconalTest::UpdateSystem(uchar4* deviceMemory, double timePassedInSeconds)
{
	UnsignedArray wallInformation = m_velocityManager.GetWallInformation();
	m_densityManager.EnforceBoundaryConditions(wallInformation);
	FloatArray densityField = m_densityManager.GetDensityField();
	

	m_velocityManager.UpdateVelocityField(densityField);

	FloatArray velocityField = m_velocityManager.GetVelocityField();
	UnsignedArray targetArea = m_densityManager.GetTargetArea();
	m_solver.PerformIterations(m_numOfGridIterations, velocityField,  targetArea);
	
	
	FloatArray timeField = m_solver.GetTimeField();

	m_visualizer.VisualizeScalarField(timeField, m_maximumTime, deviceMemory);
	m_visualizer.VisualizeIsoLines(timeField, m_isoLineDistance,  deviceMemory);
}
