#include "GradientTest.h"
#include <cassert>
#include <cmath>
#include <vector_functions.hpp>


void GradientTest::InitializeTest(const char* densityFile, const char* wallFile, int visualizationDecision)
{
	assert(m_wallInformation == NULL);
	assert(m_densityInformation == NULL);
	m_visualizationDecision = visualizationDecision;

	m_wallReader.ReadFile(wallFile);
	m_wallInformation = m_helper.UploadPicture(&m_wallReader, 0, m_wallStride);

	m_densityReader.ReadFile(densityFile);
	m_densityInformation = m_helper.UploadPictureAsFloat(&m_densityReader, INFINITY, 0.0f, 100.0f, m_densityStride);

	m_gradientModule.PreprareModule();
}

void GradientTest::UpdateSystem(uchar4* deviceMemory, double timePassedInSeconds)
{
	m_gradientModule.ComputeGradient(m_densityInformation, m_densityStride, m_wallInformation, m_wallStride);
	m_gradientModule.VisualizeField(m_visualizationDecision, 3.0f, deviceMemory);
}
