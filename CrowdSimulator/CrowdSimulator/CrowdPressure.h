#pragma once
#include "MemoryStructs.h"


class DataBase;

class CrowdPressure
{
public:
	/**
	 * \brief Generates all the required components to run the module.
	 */
	void ToolSystem();

	/**
	 * \brief Creates the crowd pressure from the density and velocity magnitude information.
	 * \param density The density we calculate the pressure from.
	 * \param velocity The velocity information we calculate pressure from.
	 * \param dataBase The data base for the wall information.
	 */
	void ComputeCrowdPressure(FloatArray density, FloatArray velocity, DataBase* dataBase);

	/**
	 * \brief Gets pressure array.
	 * \return Last computed crowd pressure.
	 */
	FloatArray GetCrowdPressure() { return m_pressureArray; }

private:
	FloatArray m_pressureArray;
};
