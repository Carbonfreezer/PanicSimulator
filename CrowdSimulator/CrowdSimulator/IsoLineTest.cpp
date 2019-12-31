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
	m_scalarMemory = m_transferHelper.UploadPictureAsFloat(&m_scalarPicture, 0.0f, 0.0f, 100.0f, m_scalarStride);
}

void IsoLineTest::UpdateSystem(uchar4* deviceMemory)
{
	m_transferHelper.VisualizeScalarField(m_scalarMemory, 100.0f, m_scalarStride, deviceMemory);
	m_transferHelper.VisualizeIsoLines(m_scalarMemory, 10.0f, m_scalarStride, deviceMemory);
}
