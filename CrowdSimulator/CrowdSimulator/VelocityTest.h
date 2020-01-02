#pragma once
#include "LogicClass.h"
#include "VelocityManager.h"


// 


/**
 * \brief Test that checks, if the density is correctly mapped to the velocity.
 * Uses initial density and wall data.
 */
class VelocityTest : public LogicClass
{
public:
	VelocityTest() : m_strideOnGradient(15) {};

	void SetStridesOnGradient(int stride) { m_strideOnGradient = stride; }
	virtual void ToolSystem(DataBase* dataBase);
	virtual void UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase);

private:
	VelocityManager m_velocityManager;
	int m_strideOnGradient;
};