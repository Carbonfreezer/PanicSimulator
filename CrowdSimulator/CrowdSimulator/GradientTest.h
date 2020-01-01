#pragma once
#include "LogicClass.h"
#include "TransferHelper.h"
#include "TgaReader.h"
#include "GradientModule.h"


class GradientTest : public LogicClass
{
public:
	GradientTest() : m_densityInformation(NULL), m_wallInformation(NULL){};

	void InitializeTest(const char* densityFile, const char* wallFile, int visualizationDecision);

	virtual void UpdateSystem(uchar4* deviceMemory, double timePassedInSeconds);

private:
	TransferHelper m_helper;
	TgaReader m_densityReader;
	TgaReader m_wallReader;


	float* m_densityInformation;
	size_t m_densityStride;

	unsigned int* m_wallInformation;
	size_t m_wallStride;

	int m_visualizationDecision;

	GradientModule m_gradientModule;
	
};

