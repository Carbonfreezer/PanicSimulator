#include "IconalTest.h"




void IconalTest::UpdateSystem(uchar4* deviceMemory)
{
	m_solver.PerformIterations(m_numOfGridIterations);
	size_t timeStride;
	float* timeField = m_solver.GetTimeField(timeStride);

	m_visualizer.VisualizeScalarField(timeField, m_maximumTime, timeStride, deviceMemory);
	m_visualizer.VisualizeIsoLines(timeField, 2.5f, timeStride, deviceMemory);
}
