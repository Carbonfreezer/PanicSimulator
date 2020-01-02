#pragma once
#include <surface_types.h>
#include "GlobalConstants.h"
#include "MemoryStructs.h"

class TgaReader;

/**
 * \brief Helper functions related to file loading, preparing memory on the GPU and copying memory on the GPU.
 */
class TransferHelper
{
public:

	//
	// Generating initial data fields.
	// 
	
	// Uploads the pictures as unsigned char array. Takes the red component. Picture must be 285 by 285. Resulting image is 287 by 287 adding the boundary structures.
	static UnsignedArray  UploadPicture(TgaReader* reader, unsigned char boundaryValue);
	// Transforms all to float values.
	static FloatArray UploadPictureAsFloat(TgaReader* reader, float boundaryValue, float minValueMapped, float maxValueMapped);

	// Simply reserves some space for the processing part in floats.
	static FloatArray ReserveFloatMemory();
	// Simply reserves unsigned memory (filled with zero).
	static UnsignedArray ReserveUnsignedMemory();
	
	// Reserves float memory filled with a defined constant value.
	static FloatArray UpfronFilledValue(float value);
	// Builds a horizontal gradient field decreasing in size.
	static FloatArray BuildHorizontalGradient(float startMax, int direction);
	// Builds a vertical gradient field decreasing in size.
	static FloatArray BuildVerticalGradient(float startMax, int direction);
	// Builds a radial gradient.
	static FloatArray BuildRadialGradient(float startMax, int direction);


	// Copys float data from to on the GPU.
	static void CopyDataFromTo(FloatArray source, FloatArray destination);
	
private:
	static int m_intArea[gGridSizeExternal * gGridSizeExternal];
	static float m_floatArea[gGridSizeExternal * gGridSizeExternal];

	
};
