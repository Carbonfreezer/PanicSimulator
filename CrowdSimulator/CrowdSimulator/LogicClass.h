#pragma once
#include <surface_types.h>

/**
 * \brief Class used as an interface to bind in the logic for the kernel.
 */
class LogicClass
{	
public:
	virtual ~LogicClass() {};
	
	virtual void UpdateSystem(uchar4* deviceMemory){};


};
