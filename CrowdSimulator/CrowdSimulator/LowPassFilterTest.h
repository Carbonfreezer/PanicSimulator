#pragma once
#include "LogicClass.h"
#include "LowPassFilter.h"

/**
 * \brief Tests the low pass filter system Uses density and wall data.
 */
class LowPassFilterTest : public LogicClass
{
public:

	/**
	 * \brief Define the number of consecutive runs we want to apply on the filter.
	 * \param iterations Can be 1 or 2
	 */
	void SetIterations(int iterations) { m_numOfIterations = iterations; }
	
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
	LowPassFilter m_filterA;
	LowPassFilter m_filterB;
	int m_numOfIterations;

	

};
