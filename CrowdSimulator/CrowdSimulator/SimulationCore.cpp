#include "SimulationCore.h"
#include "VisualizationHelper.h"
#include "GlobalConstants.h"
#include "InputSystem.h"
#include "DataBase.h"
#include <vector_functions.hpp>

void SimulationCore::ToolSystem(DataBase* dataBase)
{
	m_velocity.GenerateVelocityField();
	m_density.InitializeManager(dataBase);
	m_eikonalSolver.PrepareSolving();
	m_simulationFactor = 1.0f;
	m_showsAnnotation = false;
	m_showEikonalSolution = false;

}

void SimulationCore::HandleInput(InputSystem* input, DataBase* dataBase)
{
	if (input->WasResetToggled())
		m_density.ResetDensity(dataBase);

	m_visualizationMode = input->GetVisualizationMode();
	if (input->IsSimulationPaused())
	{
		m_simulationFactor = 0.0f;
	}
	else
	{
		switch(input->GetVelocityCounter())
		{
		case 0:
			m_simulationFactor = 1.0f;
			break;
		case 1:
			m_simulationFactor = 2.0f;
			break;
		case 2:
			m_simulationFactor = 4.0f;
			break;
		case 3:
			m_simulationFactor = 8.0f;
			break;
		}
	
	}

	m_showsAnnotation = input->GetAnnotationMode();
	m_showEikonalSolution = input->ShowsEikonalSolution();
	m_isoLineDistance = input->GetIsoLineDistance();
}

void SimulationCore::UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase)
{
	// Make sure people get spawned and despawned at the correct locations.
	m_density.EnforceBoundaryConditions(dataBase);
	FloatArray densityField = m_density.GetDensityField();

	// Now we feed the density into the velocity system.
	m_velocity.UpdateVelocityField(densityField, dataBase);
	FloatArray velocityField = m_velocity.GetVelocityField();
	
	
	// TODO: Later on comes the crowd pressure here.


	// Feed the stuff into the Eukanal solver.
	m_eikonalSolver.SolveEquation(velocityField, dataBase);
	FloatArray timeField = m_eikonalSolver.GetTimeField();

	// Now we integrate the continuity equation.
	m_density.UpdateDensityField(timePassedInSeconds * m_simulationFactor, timeField, velocityField, dataBase);

	densityField = m_density.GetDensityField();

	switch(m_visualizationMode)
	{
	case 0:
		VisualizationHelper::VisualizeScalarField(densityField, gMaximumDensity, deviceMemory);
		break;
	case 1:
		VisualizationHelper::VisualizeScalarField(velocityField, gMaximumWalkingVelocity, deviceMemory);
		break;
	}

	if (m_showEikonalSolution)
		VisualizationHelper::VisualizeIsoLines(timeField, m_isoLineDistance, deviceMemory);


	if (m_showsAnnotation)
	{
		VisualizationHelper::MarcColor(dataBase->GetWallData(), make_uchar4(255, 255, 255, 255), deviceMemory);
		VisualizationHelper::MarcColor(dataBase->GetDespawnData(), make_uchar4(255, 255, 0, 255), deviceMemory);
		VisualizationHelper::MarcColor(dataBase->GetTargetData(), make_uchar4(0, 255, 0, 255), deviceMemory);

	}

	
	
}

