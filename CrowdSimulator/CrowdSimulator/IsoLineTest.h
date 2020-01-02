#pragma once
#include "LogicClass.h"
#include "TransferHelper.h"
#include "VisualizationHelper.h"

/**
 * \brief Uses density data.
 */
class IsoLineTest : public LogicClass
{
public:

	virtual void UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase);

private:
	VisualizationHelper m_visualizer;
};

