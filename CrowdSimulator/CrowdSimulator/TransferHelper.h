#pragma once
#include "GlobalConstants.h"
#include "MemoryStructs.h"

class TgaReader;

/**
 * \brief Helper functions related to file loading, preparing memory on the GPU and copying memory on the GPU.
 */
class TransferHelper
{
public:

	/**
	 * \brief Uploads the pictures as unsigned char array. Takes the red component. Picture must be 288 by 288. Resulting image is 290 by 290 adding the boundary structures.
	 * \param reader Tga reader to read data from.
	 * \param boundaryValue The value that should be assumed at the outer pixel ring (boundary).
	 * \return Structure that describes the memory on the graphics card.
	 */
	static UnsignedArray  UploadPicture(TgaReader* reader, unsigned char boundaryValue);

	/**
	 * \brief Uploads the pictures as unsigned float array. Takes the red component. Picture must be 288 by 288. Resulting image is 290 by 290 adding the boundary structures.
	 * \param reader Tga Reader to read data from
	 * \param boundaryValue The boundary value assumed on the outer rim
	 * \param minValueMapped The float value that should get mapped to red 0.
	 * \param maxValueMapped The float value that should get mapped to red 255.
	 * \return Structure that describes the float array on the graphics card. 
	 */
	static FloatArray UploadPictureAsFloat(TgaReader* reader, float boundaryValue, float minValueMapped, float maxValueMapped);

	/**
	 * \brief Reserves a float array on the graphics card will with 0.
	 * \return Memory on graphics card.
	 */
	static FloatArray ReserveFloatMemory();

	/**
	 * \brief Reserves n unsigned  array on the graphics card will with 0.
	 * \return  Memory on graphics card.
	 */
	static UnsignedArray ReserveUnsignedMemory();
	
	/**
	 * \brief Reserves float memory filled with a defined constant value. 
	 * \param value Value to get filled into the memory.
	 * \return Float Arrayy on graphics card.
	 */
	static FloatArray UpfronFilledValue(float value);

	/**
	 * \brief Copies data from to on the GPU.
	 * \param source Source to copy data from.
	 * \param destination Destination to copy data to.
	 */
	static void CopyDataFromTo(FloatArray source, FloatArray destination);

	/**
	 * \brief Builds a horizontal gradient field decreasing in size.
	 * \param startMax The maximum value we have on one side (other is zero) 
	 * \param direction 0 or 1 differentiating x direction 1 increases in dimension 0 decreases
	 * \return Float Array with allocated memory on graphics card.
	 */
	static FloatArray BuildHorizontalGradient(float startMax, int direction);

	/**
	 * \brief Builds a horizontal gradient field decreasing in size.
	 * \param startMax The maximum value we have on one side (other is zero)
	 * \param direction direction 0 or 1 differentiating y direction 1 increases in dimension 0 decreases
	 * \return Array with allocated memory on graphics card.
	 */
	static FloatArray BuildVerticalGradient(float startMax, int direction);

	/**
	 * \brief Builds a radial gradient around the center.
	 * \param startMax The maximum value we get.
	 * \param direction 0 increasing with distance 1 decreasing
	 * \return Float value structure of the gradient field.
	 */
	static FloatArray BuildRadialGradient(float startMax, int direction);

	
private:
	static int m_intArea[gGridSizeExternal * gGridSizeExternal];
	static float m_floatArea[gGridSizeExternal * gGridSizeExternal];

	
};
