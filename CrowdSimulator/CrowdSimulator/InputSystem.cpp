#include "InputSystem.h"


void InputSystem::InitializeSystem(GLFWwindow* window)
{
	m_window = window;
	m_visualizationMode = 0;
	m_isSimulationPaused = false;
	m_wasSpacePressed = false;
	m_annotationMode = 0;
	m_wasAPRessed = false;
	m_showsEikonalSolution = false;
	m_wasEPressed = false;
	m_wasDownPressed = false;
	m_wasUpPressed = false;
	m_velocityCounter = 0;
	m_isoLineDistance = 7;
	m_wasLeftPressed = false;
	m_wasRightPressed = false;
}

void InputSystem::Update()
{
	m_resetToggled = (glfwGetKey(m_window, GLFW_KEY_R) == GLFW_PRESS);

	if (glfwGetKey(m_window, GLFW_KEY_1) == GLFW_PRESS)
		m_visualizationMode = 0;
	if (glfwGetKey(m_window, GLFW_KEY_2) == GLFW_PRESS)
		m_visualizationMode = 1;
	if (glfwGetKey(m_window, GLFW_KEY_3) == GLFW_PRESS)
		m_visualizationMode = 2;

	if (glfwGetKey(m_window, GLFW_KEY_SPACE) == GLFW_PRESS)
	{
		if (!m_wasSpacePressed)
		{
			m_isSimulationPaused = !m_isSimulationPaused;
			m_velocityCounter = 0;
		}
		m_wasSpacePressed = true;
	}
	else
	{
		m_wasSpacePressed = false;
	}

	if (glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS)
	{
		if (!m_wasAPRessed)
			m_annotationMode = !m_annotationMode;
		m_wasAPRessed = true;
	}
	else
	{
		m_wasAPRessed = false;
	}

	if (glfwGetKey(m_window, GLFW_KEY_E) == GLFW_PRESS)
	{
		if (!m_wasEPressed )
			m_showsEikonalSolution = !m_showsEikonalSolution;
		m_wasEPressed = true;
	}
	else
	{
		m_wasEPressed = false;
	}

	if (glfwGetKey(m_window, GLFW_KEY_DOWN) == GLFW_PRESS)
	{
		if ((!m_wasDownPressed) && (m_velocityCounter > 0))
			m_velocityCounter--;
		m_wasDownPressed = true;
	}
	else
	{
		m_wasDownPressed = false;
	}

	if (glfwGetKey(m_window, GLFW_KEY_UP) == GLFW_PRESS)
	{
		if ((!m_wasUpPressed) && (m_velocityCounter < 4))
			m_velocityCounter++;
		m_wasUpPressed = true;
	}
	else
	{
		m_wasUpPressed = false;
	}

	if (glfwGetKey(m_window, GLFW_KEY_RIGHT) == GLFW_PRESS)
	{
		if ((!m_wasRightPressed) && (m_showsEikonalSolution))
			m_isoLineDistance++;
		m_wasRightPressed = true;
	}
	else
	{
		m_wasRightPressed = false;
	}

	if (glfwGetKey(m_window, GLFW_KEY_LEFT) == GLFW_PRESS)
	{
		if ((!m_wasLeftPressed) && (m_showsEikonalSolution) && (m_isoLineDistance > 1))
			m_isoLineDistance--;
		m_wasLeftPressed = true;
	}
	else
	{
		m_wasLeftPressed = false;
	}

	
	
}

