#include "FlagTest.h"
#include <vector_functions.hpp>


void FlagTest::LoadFlagPicture(const char* fileName)
{
	m_maskPicture.ReadFile(fileName);
	m_maskData = m_transferHelper.UploadPicture(&m_maskPicture, 0);
}

void FlagTest::LoadScalarPicture(const char* fileName)
{
	m_scalarPicture.ReadFile(fileName);
	m_scalarData = m_transferHelper.UploadPictureAsFloat(&m_scalarPicture, 0.0f, 0.0f, 1.0f);
}

void FlagTest::UpdateSystem(uchar4* deviceMemory, double timePassedInSeconds)
{
	m_transferHelper.VisualizeScalarField(m_scalarData, 1.0f, deviceMemory);
	m_transferHelper.MarcColor(m_maskData, deviceMemory, make_uchar4(255, 255, 255, 255));
}
