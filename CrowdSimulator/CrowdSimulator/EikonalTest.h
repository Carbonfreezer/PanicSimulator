#pragma once
#include "LogicClass.h"
#include "EikonalSolver.h"
#include "DensityManager.h"
#include "VelocityManager.h"

/**
 * \brief Uses target data, wall data and eventually initial spawn data.Runs a test for the eikonal equation.
 */
class EikonalTest : public LogicClass
{
public:
	EikonalTest() { m_maximumTime = 100.0f; m_isoLineDistance = 2.5f;}


	/**
	 * \brief The maximum time assumed for visualization.
	 * \param time Maximum Time.
	 */
	void SetMaximumTime(float time) { m_maximumTime = time; }


	/**
	 * \brief Sets the iso line distance for visualization.
	 * \param time Time distance between visualizations.
	 */
	void SetIsoLineDistance(float time) { m_isoLineDistance = time; }

	/**
	 * \brief This is called after all the graphics stuff has been initialized before the first update system but only once.
	 * \param dataBase The data base with all the formal stuff in there.
	 */
	virtual void ToolSystem(DataBase* dataBase);


	/**
	 * \brief Is called for update and effectively rendering.
	 * \param deviceMemory The texture memory we write our data to.
	 * \param timePassedInSeconds The time in seconds passed since last update.
	 * \param dataBase The database with all the relevant information in.
	 */
	virtual void UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase);

private:
	EikonalSolver m_solver;
	float m_maximumTime;
	float m_isoLineDistance;

	DensityManager m_densityManager;
	VelocityManager m_velocityManager;
};

