#pragma once



class TgaReader
{
public:
	TgaReader();
	~TgaReader();

	void ReadFile(const char* fileName);

	unsigned char* GetPixels() { return m_pixels; }

	int GetWidth() { return m_width; }
	int GetHeight() { return m_height; }

private:
	int m_width;
	int m_height;

	unsigned char* m_pixels;
};

