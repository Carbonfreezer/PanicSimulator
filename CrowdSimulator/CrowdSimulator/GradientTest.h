#pragma once
#include "LogicClass.h"
#include "GradientModule.h"
#include "MemoryStructs.h"


/**
 * \brief Uses wall data. Simple test for the gradient module. Visualizes the gradient. Uses
 * a radial gradient.
 */
class GradientTest : public LogicClass
{
public:

	/**
	 * \brief Sets visualization mode
	 * \param visualizationDecision 0: visualize x derivative 1: visualize y derivative
	 */
	void PrepareTest(int visualizationDecision);

	/**
	 * \brief Is called for update and effectively rendering.
	 * \param deviceMemory The texture memory we write our data to.
	 * \param timePassedInSeconds The time in seconds passed since last update.
	 * \param dataBase The database with all the relevant information in.
	 */
	virtual void UpdateSystem(uchar4* deviceMemory, float timePassedInSeconds, DataBase* dataBase);

private:
	
	FloatArray m_densityInformation;
	int m_visualizationDecision;
	
	GradientModule m_gradientModule;
	
};

