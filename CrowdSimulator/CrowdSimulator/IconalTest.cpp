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
	size_t wallStride;
	unsigned int* wallInformation = m_velocityManager.GetWallInformation(wallStride);
	m_densityManager.EnforceBoundaryConditions(wallInformation, wallStride);
	float* densityField;
	size_t densityStride;
	densityField = m_densityManager.GetDensityField(densityStride);

	unsigned int* targetArea;
	size_t targetAreaStride;
	targetArea = m_densityManager.GetTargetArea(targetAreaStride);


	m_velocityManager.UpdateVelocityField(densityField, densityStride);

	float* velocityField;
	size_t velocityStride;
	velocityField = m_velocityManager.GetVelocityField(velocityStride);

	
	m_solver.PerformIterations(m_numOfGridIterations, velocityField, velocityStride, targetArea, targetAreaStride);
	
	size_t timeStride;
	float* timeField = m_solver.GetTimeField(timeStride);

	m_visualizer.VisualizeScalarField(timeField, m_maximumTime, timeStride, deviceMemory);
	m_visualizer.VisualizeIsoLines(timeField, m_isoLineDistance, timeStride, deviceMemory);
}
