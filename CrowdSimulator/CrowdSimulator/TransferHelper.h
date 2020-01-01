#pragma once
#include <surface_types.h>
#include "GlobalConstants.h"
#include "MemoryStructs.h"

class TgaReader;

class TransferHelper
{
public:
	
	// Uploads the pictures as unsigned char array. Takes the red component. Picture must be 285 by 285. Resulting image is 287 by 287 adding the boundary structures.
	UnsignedArray  UploadPicture(TgaReader* reader, unsigned char boundaryValue);
	// Transforms all to float values.
	FloatArray UploadPictureAsFloat(TgaReader* reader, float boundaryValue, float minValueMapped, float maxValueMapped);
	// Simply reserves some space for the processing part in floats.
	FloatArray ReserveFloatMemory();


	// Visualization helpers.
	
	// Marks the cells as being white on the texture memory.
	void MarcColor(UnsignedArray data, uchar4* pixelMemory, uchar4 color);
	// Maps the data to a blue red flow.
	void VisualizeScalarField(FloatArray deviceData, float maximumValue, uchar4* pixelMemory);
	// Maps the data to a blue red flow.
	void VisualizeScalarFieldWithNegative(FloatArray deviceData, float maximumValue,  uchar4* pixelMemory);
	
	// Visualizes some iso-lines on the indicated float memory field.
	void VisualizeIsoLines(FloatArray deviceData, float isoLineStepSize,  uchar4* pixelMemory);
	
private:
	int m_intArea[gGridSizeExternal * gGridSizeExternal];
	float m_floatArea[gGridSizeExternal * gGridSizeExternal];

	// Data on GPU for iso line buffering.
	FloatArray m_isoLineData;
};
