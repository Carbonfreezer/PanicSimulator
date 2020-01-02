#include "DataBase.h"
#include <cstddef>
#include "TgaReader.h"
#include "TransferHelper.h"
#include <cassert>

void DataBase::LoadFiles(BaseFileNames filenames)
{
	// If we are double loaded something has gone wrong.
	assert(!m_alreadyLoaded);
	m_alreadyLoaded = true;


	m_wallData = DefaultLoadUnsigned(filenames.m_wallFile);
	m_targetData = DefaultLoadUnsigned(filenames.m_targetFile);
	m_despawnData = DefaultLoadUnsigned(filenames.m_despawnFile);

	m_spawnData = DefaultLoadFloat(filenames.m_spawnFile);
	m_initialDensityData = DefaultLoadFloat(filenames.m_initialDensityFile);
}

void DataBase::FreeResources()
{
	m_initialDensityData.FreeArray();
	m_spawnData.FreeArray();

	m_wallData.FreeArray();
	m_targetData.FreeArray();
	m_despawnData.FreeArray();
}

UnsignedArray DataBase::DefaultLoadUnsigned(const char* fileName)
{
	if (fileName != NULL)
	{
		m_reader.ReadFile(fileName);
		return m_helper.UploadPicture(&m_reader, 0);
	}

	return m_helper.ReserveUnsignedMemory();
}

FloatArray DataBase::DefaultLoadFloat(const char* fileName)
{
	if (fileName != NULL)
	{
		m_reader.ReadFile(fileName);
		return m_helper.UploadPictureAsFloat(&m_reader, 0.0f, 0.0f, gMaximumDensity);
	}

	return m_helper.ReserveFloatMemory();
}
