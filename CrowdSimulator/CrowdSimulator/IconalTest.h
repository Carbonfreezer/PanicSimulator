#pragma once
#include "LogicClass.h"
#include "IconalSolver.h"

class IconalTest :
	public LogicClass
{
public:
	IconalTest() { m_maximumTime = 100.0f; }

	void PrepareTest(const char* spawnFile, const char* wallFile)
	{
		m_solver.SetTargetArea(spawnFile);
		m_solver.SetWallFile(wallFile);
		m_solver.PrepareSolving();
	}

	void SetIterations(int gridIterations)
	{
		m_numOfGridIterations = gridIterations;
	}

	void SetMaximumTime(float time) { m_maximumTime = time; }
	
	virtual void UpdateSystem(uchar4* deviceMemory);

private:
	IconalSolver m_solver;
	TransferHelper m_visualizer;
	int m_numOfGridIterations;
	float m_maximumTime;
};

