#pragma once
#include "LogicClass.h"

/**
 * \brief Uses density data and wall data. Simple test for scalar field visualization and flagging.
 */
class FlagTest :public LogicClass
{
public:

	/**
	 * \brief Is called for update and effectively rendering.
	 * \param deviceMemory The texture memory we write our data to.
	 * \param timePassedInSeconds The time in seconds passed since last update.
	 * \param dataBase The database with all the relevant information in.
	 */
	virtual void UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase);

};

