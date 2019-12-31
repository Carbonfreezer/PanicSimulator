#pragma once
#include "LogicClass.h"
#include "TgaReader.h"
#include "TransferHelper.h"

class FlagTest :
	public LogicClass
{
public:


	void LoadFlagPicture(const char* fileName);
	void LoadScalarPicture(const char* fileName);
	virtual void UpdateSystem(uchar4* deviceMemory);

private:
	TgaReader m_maskPicture;
	TgaReader m_scalarPicture;
	TransferHelper m_transferHelper;
	unsigned int* m_maskMemory;
	size_t m_maskStride;
	float* m_scalarMemory;
	size_t m_scalarStride;
};

