#pragma once
#include "LogicClass.h"
#include "TgaReader.h"
#include "TransferHelper.h"
#include "MemoryStructs.h"

class FlagTest :
	public LogicClass
{
public:


	void LoadFlagPicture(const char* fileName);
	void LoadScalarPicture(const char* fileName);
	virtual void UpdateSystem(uchar4* deviceMemory, double timePassedInSeconds);

private:
	TgaReader m_maskPicture;
	TgaReader m_scalarPicture;
	TransferHelper m_transferHelper;
	UnsignedArray m_maskData;
	FloatArray m_scalarData;
};

