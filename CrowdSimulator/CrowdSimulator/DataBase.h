#pragma once
#include "MemoryStructs.h"
#include "TgaReader.h"
#include "TransferHelper.h"
#include "BaseFileNames.h"

// This is a central accumulation point for all the data we have available.
class DataBase
{
public:
	DataBase() { m_alreadyLoaded = false; }
	// Loads all the files for the specific cases. NULL pointers can be inserted to assume default configurations.
	void LoadFiles(BaseFileNames fileNames);
	// Frees all the resources from the graphics card.
	void FreeResources();

	FloatArray GetInitialDensityData() { return m_initialDensityData; }
	FloatArray GetSpawnData() { return m_spawnData; }

	UnsignedArray GetWallData() { return m_wallData; }
	UnsignedArray GetTargetData() { return  m_targetData; }
	UnsignedArray GetDespawnData() { return m_despawnData; }
	
private:
	bool m_alreadyLoaded;

	TgaReader m_reader;
	
	FloatArray m_initialDensityData;
	FloatArray m_spawnData;

	UnsignedArray m_wallData;
	UnsignedArray m_targetData;
	UnsignedArray m_despawnData;


	UnsignedArray DefaultLoadUnsigned(const char* fileName);
	FloatArray DefaultLoadFloat(const char* fileName);
	
	
	
};