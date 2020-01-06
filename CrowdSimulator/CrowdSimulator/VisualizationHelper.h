#pragma once
#include  "MemoryStructs.h"
#include <surface_types.h>

/**
 * \brief Provides all kinds of visualization functions to visualize float and unsigned fields
 * on the graphics card. Pixel memory is the memory the content is written to.
 * In the current application mapped pixel buffer is used. All functions take the inner part of
 * the array (- 1 pixel boundary) and inflate the visualization by a certain factor.
 */
class VisualizationHelper
{
public:
	// Marks the cells as being white on the texture memory.
	/**
	 * \brief Marcs the pixels as being given in the array handed over. Meant for overpainting an existing visualization.
	 * \param data Data containing a 1 where a pixel should be set.
	 * \param color The color that should be set.
	 * \param pixelMemory The destination written to.
	*/
	static void MarcColor(UnsignedArray data, uchar4 color, uchar4* pixelMemory);


	// Maps the data to a blue red flow.
	/**
	 * \brief Visualizes a scalar field as a blue red gradient blue being 0 and red being a maximum value. Values above are clamped.
	 * \param deviceData The scalar field to be visualized.
	 * \param maximumValue The assumed maximum value for pure red.
	 * \param pixelMemory The pixel memory we write data to.
	 */
	static void VisualizeScalarField(FloatArray deviceData, float maximumValue, uchar4* pixelMemory);

	/**
	 * \brief Visualizes a scalar field as a blue red gradient blue being minimal value and red being a maximum value. Values above are clamped.
	 * \param deviceData The scalar field to be visualized.
	 * \param minimumValue The minimal value where we start with blue.
	 * \param maximumValue The assumed maximum value for pure red.
	 * \param pixelMemory The pixel memory we write data to.
	 */
	static void VisualizeScalarField(FloatArray deviceData, float minimumValue, float maximumValue, uchar4* pixelMemory);

	/**
	 * \brief Visualizes a scalar field as a blue red gradient blue being -maximum value and red being a maximum value. Values above / below are clamped.
	 * \param deviceData The scalar field to be visualized.
	 * \param maximumValue The assumed maximum value for pure red or -maximum for blue.
	 * \param pixelMemory The pixel memory we write data to.
	 */
	static void VisualizeScalarFieldWithNegative(FloatArray deviceData, float maximumValue, uchar4* pixelMemory);

	/**
	 * \brief Visualizes some iso-lines on the indicated float memory field.
	 * \param deviceData The data we want to compute iso lines from.
	 * \param isoLineStepSize The distance between lines in terms of value change.f
	 * \param pixelMemory The pixel memory where data is written to.
	 */
	static void VisualizeIsoLines(FloatArray deviceData, float isoLineStepSize, uchar4* pixelMemory);

private:
	// Data on GPU for iso line buffering.
	static FloatArray m_isoLineData;

	
};
