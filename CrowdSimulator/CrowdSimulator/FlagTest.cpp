#include "FlagTest.h"
#include <vector_functions.hpp>


void FlagTest::LoadFlagPicture(const char* fileName)
{
	m_maskPicture.ReadFile(fileName);
	m_maskMemory = m_transferHelper.UploadPicture(&m_maskPicture, 0, m_maskStride);
}

void FlagTest::LoadScalarPicture(const char* fileName)
{
	m_scalarPicture.ReadFile(fileName);
	m_scalarMemory = m_transferHelper.UploadPictureAsFloat(&m_scalarPicture, 0.0f, 0.0f, 1.0f, m_scalarStride);
}

void FlagTest::UpdateSystem(uchar4* deviceMemory, double timePassedInSeconds)
{
	m_transferHelper.VisualizeScalarField(m_scalarMemory, 1.0f, m_scalarStride, deviceMemory);
	m_transferHelper.MarcColor(m_maskMemory, m_maskStride, deviceMemory, make_uchar4(255, 255, 255, 255));
}
