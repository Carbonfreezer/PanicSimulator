#pragma once
#include "LogicClass.h"
#include "TransferHelper.h"

/**
 * \brief Uses density data and wall data.
 */
class FlagTest :public LogicClass
{
public:

	virtual void UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase);

private:
	TransferHelper m_transferHelper;
};

