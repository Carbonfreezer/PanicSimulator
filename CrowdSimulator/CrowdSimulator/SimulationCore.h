#pragma once
#include "LogicClass.h"
#include "DensityManager.h"
#include "VelocityManager.h"
#include "EikonalSolver.h"
#include "LowPassFilter.h"
#include "CrowdPressure.h"
#include "Accumulator.h"

/**
 * \brief This is the core class that runs the complete simulation.
 */
class SimulationCore : public LogicClass
{
public:
	SimulationCore() : m_visualizationMode(0){}
	
	/**
	 * \brief This is called after all the graphics stuff has been initialized before the first update system but only once.
	 * \param dataBase The data base with all the formal stuff in there.
	 */
	virtual void ToolSystem(DataBase* dataBase);

	/**
	* \brief Frees all the resources currently allocated on the graphics card.
	*/
	virtual void FreeResources();


	/**
	 * \brief Is called to handle the input (if desired). Handle input is called just before update system.
	 * \param input Reference to the input system to be asked for the state.
	 * \param dataBase The database with all the relevant information.
	 */
	virtual void HandleInput(InputSystem* input, DataBase* dataBase);
	
	/**
	 * \brief Is called for update and effectively rendering.
	 * \param deviceMemory The texture memory we write our data to.
	 * \param timePassedInSeconds The time in seconds passed since last update.
	 * \param dataBase The database with all the relevant information in.
	 */
	virtual void UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase);

	
private:
	// Contains the visualization mode.
	int m_visualizationMode;
	// Contains the decorator mode. 
	bool  m_showsAnnotation;
	// Annotates if we want to have the solution of the eikonal equation.
	bool m_showEikonalSolution;
	
	// Factor used for simulation speed.
	float m_simulationFactor;

	// The distance we have between iso lines for the eikonal solution.
	float m_isoLineDistance;
	
	DensityManager m_density;
	VelocityManager m_velocity;
	EikonalSolver m_eikonalSolver;

	LowPassFilter m_velocityFilter[2];
	LowPassFilter m_timeFilter[2];

	CrowdPressure m_crowdPressure;

	Accumulator m_pressureAccumulator;
	Accumulator m_densityAccumulator;

	FloatArray PerformLowPassIterations(FloatArray input, LowPassFilter filterPair[2], UnsignedArray wallData,  int iterations);

};
