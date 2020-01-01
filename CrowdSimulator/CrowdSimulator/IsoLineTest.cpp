#include "IsoLineTest.h"



IsoLineTest::IsoLineTest()
{
}


IsoLineTest::~IsoLineTest()
{
}

void IsoLineTest::LoadScalarPicture(const char* fileName)
{
	m_scalarPicture.ReadFile(fileName);
	m_scalarMemory = m_transferHelper.UploadPictureAsFloat(&m_scalarPicture, 0.0f, 0.0f, 100.0f);
}

void IsoLineTest::UpdateSystem(uchar4* deviceMemory, double timePassedInSeconds)
{
	m_transferHelper.VisualizeScalarField(m_scalarMemory, 100.0f, deviceMemory);
	m_transferHelper.VisualizeIsoLines(m_scalarMemory, 10.0f,  deviceMemory);
}