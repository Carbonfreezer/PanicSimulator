#pragma once
#include "LogicClass.h"
#include "VelocityManager.h"

/**
 * \brief Test that checks, if the density is correctly mapped to the velocity.
 * Uses initial density and wall data.
 */
class VelocityTest : public LogicClass
{
public:
	VelocityTest() : m_strideOnGradient(15) {};

	/**
	 * \brief Used for iso line visualization. How many lines should be shown from 0 to max velocity.
	 * \param stride Amouint of lines to show.
	 */
	void SetStridesOnGradient(int stride) { m_strideOnGradient = stride; }

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
	 * \brief Is called for update and effectively rendering.
	 * \param deviceMemory The texture memory we write our data to.
	 * \param timePassedInSeconds The time in seconds passed since last update.
	 * \param dataBase The database with all the relevant information in.
	 */
	virtual void UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase);

private:
	VelocityManager m_velocityManager;
	int m_strideOnGradient;
};