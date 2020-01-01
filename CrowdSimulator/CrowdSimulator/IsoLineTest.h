#pragma once
#include "LogicClass.h"
#include "TransferHelper.h"
#include "TgaReader.h"
#include  "MemoryStructs.h"

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
	FloatArray m_scalarMemory;
	
};

