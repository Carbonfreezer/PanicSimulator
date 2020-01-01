#pragma once
#include "LogicClass.h"
#include "TgaReader.h"
#include "VelocityManager.h"
#include "MemoryStructs.h"


// Test that checks, if the density is correctly mapped to the velocity.
class VelocityTest : public LogicClass
{
public:
	void PrepareTest(const char* densityFile, const char* wallFile, float isoLineDistance);

	virtual void UpdateSystem(uchar4* deviceMemory, double timePassedInSeconds);

private:
	TgaReader m_densityFile;
	VelocityManager m_velocityManager;
	TransferHelper m_helper;

	FloatArray m_densityData;
	
	float m_isoLineDistance;
	
};