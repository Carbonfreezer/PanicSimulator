#pragma once
#include "LogicClass.h"
#include "TransferHelper.h"
#include "TgaReader.h"

class IsoLineTest :
	public LogicClass
{
public:
	IsoLineTest();
	~IsoLineTest();

	void LoadScalarPicture(const char* fileName);
	virtual void UpdateSystem(uchar4* deviceMemory, double timePassedInSeconds);

private:

	TgaReader m_scalarPicture;
	TransferHelper m_transferHelper;
	float* m_scalarMemory;
	size_t m_scalarStride;
	
};

