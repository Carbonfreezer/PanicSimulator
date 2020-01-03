#pragma once



/**
 * \brief Helper class to read a tga file. File has to exist, must not be RLE compressed and must not have an alpha channel.
 */
class TgaReader
{
public:
	TgaReader();
	~TgaReader();

	/**
	 * \brief Reads an TGA file without RLE coding and Alpha channel
	 * \param fileName The name of the file
	 */
	void ReadFile(const char* fileName);

	/**
	 * \brief Gets the pixels of the file with 3 bytes per pixel.
	 * \return The pixels.
	 */
	unsigned char* GetPixels() { return m_pixels; }

	/**
	 * \brief Gets the width of the image.
	 * \return Width of image
	 */
	int GetWidth() { return m_width; }


	/**
	 * \brief Gets the height of the image.
	 * \return Height of the image.
	 */
	int GetHeight() { return m_height; }

private:
	int m_width;
	int m_height;

	unsigned char* m_pixels;
};

