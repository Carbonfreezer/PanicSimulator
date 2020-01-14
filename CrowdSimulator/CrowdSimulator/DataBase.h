#pragma once
#include "MemoryStructs.h"
#include "TgaReader.h"
#include "TransferHelper.h"
#include "BaseFileNames.h"

/**
 * \brief  This is a central accumulation point for all the data we have available.
 * The program is designed that way, that no constance is assumed between updates (the system is stateless),
 * so manipulations of the data can be performed during updates.
 */
class DataBase
{
public:
	DataBase() { m_alreadyLoaded = false; }

	/**
	 * \brief Loads all the files for the specific cases. NULL pointers can be inserted to assume default configurations.
	 * \param fileNames Struct with the file names.
	 */
	void LoadFiles(BaseFileNames fileNames);

	
	/**
	 * \brief Frees all the resources from the graphics card.
	 */
	void FreeResources();

	/**
	 * \brief Gets initial density data. This is the density at the beginning of the simulation.
	 * \return Density data
	 */
	FloatArray GetInitialDensityData() { return m_initialDensityData; }


	/**
	 * \brief Gets the spawn data, this field guarantees for a certain minimum of
	 * people desnity in an indicated area. This is done on a per update basis-
	 * \return Spawn data.
	 */
	FloatArray GetSpawnData() { return m_spawnData; }

	/**
	 * \brief Contains the wall information (digital)
	 * \return Wall information.
	 */
	UnsignedArray GetWallData() { return m_wallData; }


	/**
	 * \brief Contains target data where people try to walk to. (digital)
	 * People also despawn here.
	 * \return The target data.
	 */
	UnsignedArray GetTargetData() { return  m_targetData; }

	
	
private:
	bool m_alreadyLoaded;
	
	TgaReader m_reader;
	
	FloatArray m_initialDensityData;
	FloatArray m_spawnData;

	UnsignedArray m_wallData;
	UnsignedArray m_targetData;
	UnsignedArray m_wallDespawn;


	UnsignedArray DefaultLoadUnsigned(const char* fileName);
	FloatArray DefaultLoadFloat(const char* fileName);
	
	
	
};