#pragma once
#include <surface_types.h>


class DataBase;
class InputSystem;

/**
 * \brief Class used as an interface to bind in the logic for the kernel.
 */
class LogicClass
{
public:
	virtual ~LogicClass() {};

	/**
	 * \brief This is called after all the graphics stuff has been initialized before the first update system but only once.
	 * \param dataBase The data base with all the formal stuff in there.
	 */
	virtual void ToolSystem(DataBase* dataBase){};

	/**
	 * \brief Frees all the resources currently allocated on the graphics card.
	 */
	virtual void FreeResources() {};


	/**
	 * \brief Is called to handle the input (if desired). Handle input is called just before update system.
	 * \param input Reference to the input system to be asked for the state.
	 * \param dataBase The database with all the relevant information.
	 */
	virtual void HandleInput(InputSystem* input, DataBase* dataBase){}
	
	/**
	 * \brief Is called for update and effectively rendering.
	 * \param deviceMemory The texture memory we write our data to.
	 * \param timePassedInSeconds The time in seconds passed since last update.
	 * \param dataBase The database with all the relevant information in.
	 */
	virtual void UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase){};


};
