#pragma once
#include <surface_types.h>


class DataBase;
/**
 * \brief Class used as an interface to bind in the logic for the kernel.
 */
class LogicClass
{
public:
	virtual ~LogicClass() {};

	/**
	 * \brief This is called after all the graphics stuff has been initialized before the first update system but only once.
	 * \param dataBase 
	 */
	virtual void ToolSystem(DataBase* dataBase){};

	
	/**
	 * \brief Is called for update and effectively rendering.
	 * \param deviceMemory The texture memory we write our data to.
	 * \param timePassedInSeconds The time in seconds passed since last update.
	 * \param dataBase The database with all the relevant information in.
	 */
	virtual void UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase){};


};
