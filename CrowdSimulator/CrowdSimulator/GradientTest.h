#pragma once
#include "LogicClass.h"
#include "TransferHelper.h"
#include "TgaReader.h"
#include "GradientModule.h"
#include "MemoryStructs.h"


class GradientTest : public LogicClass
{
public:


	// 0: x derivative, 1 : y derivative
	void InitializeTest(const char* densityFile, const char* wallFile, int visualizationDecision);

	virtual void UpdateSystem(uchar4* deviceMemory, double timePassedInSeconds);

private:
	TransferHelper m_helper;
	TgaReader m_densityReader;
	TgaReader m_wallReader;

	FloatArray m_densityInformation;
	UnsignedArray m_wallInformation;

	int m_visualizationDecision;
	
	GradientModule m_gradientModule;
	
};

