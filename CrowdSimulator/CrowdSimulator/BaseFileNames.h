#pragma once
#include <cstddef>
#include <xstring>


/**
 * \brief Helper struct to accumulate all the names of the files that may be
 * relevant for the simulation.
 */
struct BaseFileNames
{
	BaseFileNames() : m_wallFile(NULL), m_targetFile(NULL), m_spawnFile(NULL), m_despawnFile(NULL), m_initialDensityFile(NULL) {}

	
	/**
	 * \brief Loads an ini type file where all the other filenames maybe stored.
	 * \param initializationFile 
	 */
	void LoadFilenames(const char* initializationFile);


	/**
	 * \brief Filename for tga with wall information (binary)
	 */
	const char* m_wallFile;

	
	/**
	 * \brief Filename with information for the target info (binary)
	 */
	const char* m_targetFile;

	
	/**
	 * \brief Filename with the information for the spawn file (non binary).
	 */
	const char* m_spawnFile;
	
	/**
	 * \brief Information for the area with despawn information (binary)
	 */
	const char* m_despawnFile;
	
	/**
	 * \brief Filename for the initial density information (non binary)
	 */
	const char* m_initialDensityFile;

private:
	void ParseLine(const std::string& line, std::string& key, std::string& file);

	std::string m_internallWallFile;
	std::string m_internalTargetFile;
	std::string m_internalSpawnFile;
	std::string m_internalDespawnFile;
	std::string m_internalDensityFile;
};