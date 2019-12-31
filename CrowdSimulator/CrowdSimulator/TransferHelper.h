#pragma once
#include <surface_types.h>
#include "GlobalConstants.h"


class TgaReader;

class TransferHelper
{
public:
	TransferHelper() { m_isoLineData = NULL; }
	
	// Uploads the pictures as unsigned char array. Takes the red component. Picture must be 285 by 285. Resulting image is 287 by 287 adding the boundary structures.
	unsigned int*  UploadPicture(TgaReader* reader, unsigned char boundaryValue, size_t& pitch);
	// Transforms all to float values.
	float* UploadPictureAsFloat(TgaReader* reader, float boundaryValue, float minValueMapped, float maxValueMapped, size_t& pitch);
	// Simply reserves some space for the processing part in floats.
	float* ReserveFloatMemory(size_t& pitch);


	// Visualization helpers.
	
	// Marks the cells as being white on the texture memory.
	void MarcColor(unsigned int* deviceMemory,size_t devicePitch, uchar4* pixelMemory, uchar4 color);
	// Maps the data to a blue red flow.
	void VisualizeScalarField(float* deviceMemory, float maximumValue, size_t devicePitch, uchar4* pixelMemory);
	// Visualizes some iso-lines on the indicated float memory field.
	void VisualizeIsoLines(float* dataMemory, float isoLineStepSize, size_t rawDataStride, uchar4* pixelMemory);
	
private:
	int m_intArea[gGridSizeExternal * gGridSizeExternal];
	float m_floatArea[gGridSizeExternal * gGridSizeExternal];

	// Data on GPU for iso line buffering.
	unsigned int* m_isoLineData;
	size_t m_isoLineStride;
};
