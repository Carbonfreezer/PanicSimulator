#pragma once
#include "LogicClass.h"
#include "IconalSolver.h"
#include "DensityManager.h"
#include "VelocityManager.h"

class IconalTest :
	public LogicClass
{
public:
	IconalTest() { m_maximumTime = 100.0f; m_isoLineDistance = 2.5f; }

	void PrepareTest(const char* targetFile, const char* wallFile);

	void PrepareTest(const char* targetFile, const char* wallFile, const char* spawnFile);

	void SetIterations(int gridIterations)
	{
		m_numOfGridIterations = gridIterations;
	}

	void SetMaximumTime(float time) { m_maximumTime = time; }

	void SetIsoLineDistance(float time) { m_isoLineDistance = time; }
	
	virtual void UpdateSystem(uchar4* deviceMemory, double timePassedInSeconds);

private:
	IconalSolver m_solver;
	TransferHelper m_visualizer;
	int m_numOfGridIterations;
	float m_maximumTime;
	float m_isoLineDistance;

	DensityManager m_densityManager;
	VelocityManager m_velocityManager;
};

