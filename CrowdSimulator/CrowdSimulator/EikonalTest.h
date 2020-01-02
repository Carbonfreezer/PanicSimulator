#pragma once
#include "LogicClass.h"
#include "EikonalSolver.h"
#include "DensityManager.h"
#include "VelocityManager.h"

/**
 * \brief Uses target data, wall data and eventually initial spawn data.
 */
class EikonalTest : public LogicClass
{
public:
	EikonalTest() { m_maximumTime = 100.0f; m_isoLineDistance = 2.5f;}

	void PrepareTest(const char* targetFile, const char* wallFile);
	void PrepareTest(const char* targetFile, const char* wallFile, const char* spawnFile);


	void SetMaximumTime(float time) { m_maximumTime = time; }
	void SetIsoLineDistance(float time) { m_isoLineDistance = time; }

	virtual void ToolSystem(DataBase* dataBase);
	
	virtual void UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase);

private:
	EikonalSolver m_solver;
	float m_maximumTime;
	float m_isoLineDistance;

	DensityManager m_densityManager;
	VelocityManager m_velocityManager;

	VisualizationHelper m_visualizer;
};

