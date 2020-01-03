#pragma once
#include "LogicClass.h"



/**
 * \brief Simple test to draw a checker pattern.f
 */
class CheckerTest : public LogicClass
{
public:

	CheckerTest() : m_updateCounter(0) {}

	/**
	 * \brief Is called for update and effectively rendering.
	* \param deviceMemory The texture memory we write our data to.
	 * \param timePassedInSeconds The time in seconds passed since last update.
	* \param dataBase The database with all the relevant information in.
	*/
	virtual void UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase);

private:

	unsigned char m_updateCounter;

};

