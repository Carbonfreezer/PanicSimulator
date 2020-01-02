#pragma once
#include <cstddef>


// Helper struct to accumulate all the names of the files that may be 
// relevant for the simulation.


struct BaseFileNames
{
	BaseFileNames() : m_wallFile(NULL), m_targetFile(NULL), m_spawnFile(NULL), m_despawnFile(NULL), m_initialDensityFile(NULL) {}
	const char* m_wallFile;
	const char* m_targetFile;
	const char* m_spawnFile;
	const char* m_despawnFile;
	const char* m_initialDensityFile;
};