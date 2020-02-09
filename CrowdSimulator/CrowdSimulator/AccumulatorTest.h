#pragma once
#include "LogicClass.h"
#include "Accumulator.h"

class AccumulatorTest:public LogicClass
{
	/**
	 * \brief This is called after all the graphics stuff has been initialized before the first update system but only once.
	 * \param dataBase The data base with all the formal stuff in there.
	 */
	virtual void ToolSystem(DataBase* dataBase);

	/**
	 * \brief Frees all the resources currently allocated on the graphics card.
	 */
	virtual void FreeResources();

private:
	FloatArray m_testCandidate;
	Accumulator m_accuToTest;

};
