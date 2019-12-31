#pragma once
#include "LogicClass.h"
class CheckerTest :
	public LogicClass
{
public:

	CheckerTest() : m_updateCounter(0) {}
	virtual void UpdateSystem(uchar4* deviceMemory);

private:

	unsigned char m_updateCounter;

};

