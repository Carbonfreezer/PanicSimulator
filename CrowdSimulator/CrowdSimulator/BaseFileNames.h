#pragma once
#include <cstddef>
#include <xstring>


/**
 * \brief Helper struct to accumulate all the names of the files that may be
 * relevant for the simulation.
 */
struct BaseFileNames
{
	BaseFileNames() : m_wallFile(NULL), m_targetFile(NULL), m_spawnFile(NULL),  m_initialDensityFile(NULL), m_velocityLimiter(NULL) {}

	
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
	 * \brief Filename with information for the target info (binary). People also despawn here.
	 */
	const char* m_targetFile;

	
	/**
	 * \brief Filename with the information for the spawn file (non binary).
	 */
	const char* m_spawnFile;
	
	
	/**
	 * \brief Filename for the initial density information (non binary)
	 */
	const char* m_initialDensityFile;

	/**
	 * \brief The filename of the velocity limiter if we have one.
	 */
	const char* m_velocityLimiter;

	

private:
	void ParseLine(const std::string& line, std::string& key, std::string& file);

	std::string m_internallWallFile;
	std::string m_internalTargetFile;
	std::string m_internalSpawnFile;
	std::string m_internalDensityFile;
	std::string m_velocityLimiterFile;
};