#pragma once

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include "DataBase.h"
#include "InputSystem.h"
class LogicClass;

/**
 * \brief Contains the framework for the simulation. It mainly does all the OpenGL stuff.
 */
class FrameWork
{
public:

	/**
	 * \brief Initializes the framework and sets all up.
	 * \param usedLogic The class used for execution. This contains either the simulation core or a test bed.
	 * \param windowTitle The title of the window.
	 * \param fileNames The name struct with all the typical files loaded at startup.
	 */
	void InitializeFramework(LogicClass* usedLogic, const char* windowTitle, BaseFileNames fileNames);

	/**
	 * \brief The core loop that runs all the updates.
	 */
	void RunCoreLoop();

	/**
	 * \brief Helper function to shutdown the framework.
	 */
	void ShutdownFramework();

private:
	GLuint m_pixelBuffer;
	cudaGraphicsResource* m_cuda_pixel_buffer_object;
	GLFWwindow* m_window;

	LogicClass* m_usedLogic;
	DataBase m_usedDataBase;

	InputSystem m_inputSystem;


	void InitializeGLEWandGLFW(const char* windowTitle);
	void InitializeCUDAndOpenGL();


};

