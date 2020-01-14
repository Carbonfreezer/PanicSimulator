#include "BaseFileNames.h"
#include <fstream>
#include <string>
#include <cassert>

using namespace std;

void BaseFileNames::LoadFilenames(const char* initializationFile)
{
	ifstream inputFile;

	inputFile.open(initializationFile);
	string line;

	while(getline(inputFile, line))
	{
		string key;
		string file;

		ParseLine(line, key, file);

		// Check if we have found a : as a separater
		if (strlen(file.c_str()) == 0)
			continue;

		if (key == "wall")
		{
			m_internallWallFile = file;
			m_wallFile = m_internallWallFile.c_str();
		}

		if (key == "target")
		{
			m_internalTargetFile = file;
			m_targetFile = m_internalTargetFile.c_str();
		}

		if(key == "spawn")
		{
			m_internalSpawnFile = file;
			m_spawnFile = m_internalSpawnFile.c_str();
		}


		if (key == "density")
		{
			m_internalDensityFile = file;
			m_initialDensityFile = m_internalDensityFile.c_str();
		}		
	}
}

void BaseFileNames::ParseLine(const std::string& line, std::string& key, std::string& file)
{
	key = "";
	file = "";

	bool filepart = false;
	for (char letter : line)
	{
		// Eatup white spaces and tabs.
		if ((letter == ' ') || (letter == 9))
			continue;

		// If we have a # as a comment we skip the rest of the line.
		if (letter == '#')
			return;

		if (letter == ':')
		{
			assert(!filepart);
			filepart = true;
			continue;
		}

		char lowerLetter = tolower(letter);

		if (filepart)
			file.push_back(letter);
		else
			key.push_back(lowerLetter);
	}
}
