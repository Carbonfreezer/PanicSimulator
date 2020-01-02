#pragma once
#include "LogicClass.h"
#include "EikonalSolver.h"
#include "DensityManager.h"
#include "VelocityManager.h"

/**
 * \brief Uses target data, wall data and eventually initial spawn data.
 */
class IconalTest : public LogicClass
{
public:
	IconalTest() { m_maximumTime = 100.0f; m_isoLineDistance = 2.5f; m_numOfGridIterations = 25;}

	void PrepareTest(const char* targetFile, const char* wallFile);
	void PrepareTest(const char* targetFile, const char* wallFile, const char* spawnFile);


	void SetIterations(int gridIterations) { m_numOfGridIterations = gridIterations; }
	void SetMaximumTime(float time) { m_maximumTime = time; }
	void SetIsoLineDistance(float time) { m_isoLineDistance = time; }

	virtual void ToolSystem(DataBase* dataBase);
	
	virtual void UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase);

private:
	EikonalSolver m_solver;
	TransferHelper m_visualizer;
	int m_numOfGridIterations;
	float m_maximumTime;
	float m_isoLineDistance;

	DensityManager m_densityManager;
	VelocityManager m_velocityManager;
};

