#pragma once
#include "LogicClass.h"
#include "TgaReader.h"
#include "VelocityManager.h"

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

	float* m_densityData;
	size_t m_densityStride;
	float m_isoLineDistance;
	;
	
};