#pragma once
#include  "MemoryStructs.h"
#include <surface_types.h>

class VisualizationHelper
{
public:
	// Marks the cells as being white on the texture memory.
	static void MarcColor(UnsignedArray data, uchar4* pixelMemory, uchar4 color);
	// Maps the data to a blue red flow.
	static void VisualizeScalarField(FloatArray deviceData, float maximumValue, uchar4* pixelMemory);
	// Maps the data to a blue red flow.
	static void VisualizeScalarFieldWithNegative(FloatArray deviceData, float maximumValue, uchar4* pixelMemory);
	// Visualizes some iso-lines on the indicated float memory field.
	void VisualizeIsoLines(FloatArray deviceData, float isoLineStepSize, uchar4* pixelMemory);

private:
	// Data on GPU for iso line buffering.
	// WARNING: Isoline data can not be static to avoid confusion on the GPU if several isolines get painted.
	FloatArray m_isoLineData;

	
};
