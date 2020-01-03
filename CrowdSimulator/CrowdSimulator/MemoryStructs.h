#pragma once


/**
 * \brief Float array as used on the graphics card.
 */
struct FloatArray
{
	FloatArray();
	/**
	 * \brief Frees the resources on the graphics card.
	 */
	void FreeArray();

	/**
	 * \brief Pointer to float array on graphics card.
	 */
	float* m_array;
	
	/**
	 * \brief Stride on the graphics card of the field.
	 */
	size_t m_stride;
};

/**
 * \brief Unsigned array as used on the graphics card.
 */
struct UnsignedArray
{
	UnsignedArray();
	/**
	 * \brief Frees the resources on the graphcis card.
	 */
	void FreeArray();

	/**
	 * \brief Pointer to unsigned array on graphics card.
	 */
	unsigned int* m_array;

	/**
	 * \brief Stride on the graphics card of the field.
	*/
	size_t m_stride;
};
