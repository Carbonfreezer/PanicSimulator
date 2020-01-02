#pragma once
#include "LogicClass.h"
#include "TransferHelper.h"
#include "TgaReader.h"
#include  "MemoryStructs.h"

/**
 * \brief Uses density data.
 */
class IsoLineTest : public LogicClass
{
public:

	virtual void UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase);

private:

	TransferHelper m_transferHelper;
	
};

