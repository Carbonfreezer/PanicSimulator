#include "FrameWork.h"
#include "CheckerTest.h"
#include "FlagTest.h"
#include "IsoLineTest.h"
#include "IconalTest.h"
#include "VelocityTest.h"
#include "GradientTest.h"
#include "ContinuityEquationTest.h"

static FrameWork gBaseFrameWork;
static CheckerTest gChecker;
static FlagTest gFlagTest;
static IsoLineTest gIsoLineTest;
static IconalTest gIconal;
static VelocityTest gVelocity;
static GradientTest gGradientTest;
static ContinuityEquationTest gContinuityTest;

int main(int argc, char **argv)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);


	const int gridIterations = 25;
	
	if (argc != 2)
		return 0;

	if (argv[1][0] == '0')
	{
		gBaseFrameWork.InitializeFramework(&gChecker, "Checker Test");
	}
	else if (argv[1][0] == '1')
	{
		gFlagTest.LoadFlagPicture("Picture.tga");
		gFlagTest.LoadScalarPicture("Gradient.tga");
		gBaseFrameWork.InitializeFramework(&gFlagTest, "Visualization Test");
	}
	else if (argv[1][0] == '2')
	{
		gIsoLineTest.LoadScalarPicture("Gradient.tga");
		gBaseFrameWork.InitializeFramework(&gIsoLineTest, "Isoline Test 1");
	}
	else if (argv[1][0] == '3')
	{
		gIsoLineTest.LoadScalarPicture("BigCircle.tga");
		gBaseFrameWork.InitializeFramework(&gIsoLineTest, "Isoline Test 2");
	}
	
	else if (argv[1][0] == '4')
	{
		gIconal.PrepareTest("SourcePoint.tga", "Empty.tga");
		gIconal.SetIterations(gridIterations);
		gBaseFrameWork.InitializeFramework(&gIconal, "Eikonal Test 1");
	}
	else if (argv[1][0] == '5')
	{
		gIconal.PrepareTest("SourcePoint.tga", "WallComplex.tga");
		gIconal.SetIterations(gridIterations);
		gBaseFrameWork.InitializeFramework(&gIconal, "Eikonal Test 2");
	}
	else if (argv[1][0] == '6')
	{
		gIconal.PrepareTest("SourceLine.tga", "Empty.tga");
		gIconal.SetIterations(gridIterations);
		gBaseFrameWork.InitializeFramework(&gIconal, "EikonalTest 3");
	}
	else if (argv[1][0] == '7')
	{
		gIconal.PrepareTest("SourceLine.tga", "WallSlit.tga");
		gIconal.SetIterations(gridIterations);
		gBaseFrameWork.InitializeFramework(&gIconal, "EikonalTest 4");
	}
	else if (argv[1][0] == '8')
	{
		gIconal.PrepareTest("SourceLine.tga", "WallDoubleSlit.tga");
		gIconal.SetIterations(gridIterations);
		gBaseFrameWork.InitializeFramework(&gIconal, "EikonalTest 5");
	}
	else if (argv[1][0] == '9')
	{
		gIconal.PrepareTest("SourceLine.tga", "WallComplex.tga");
		gIconal.SetIterations(gridIterations);
		gBaseFrameWork.InitializeFramework(&gIconal, "EikonalTest 6");
	}
	else if (argv[1][0] == 'A')
	{
		gIconal.PrepareTest("SourceDoublePoint.tga", "Empty.tga");
		gIconal.SetIterations(gridIterations);
		gBaseFrameWork.InitializeFramework(&gIconal, "EikonalTest 7");
	}
	else if (argv[1][0] == 'B')
	{
		gIconal.PrepareTest("SourceDoublePoint.tga", "WallSlit.tga");
		gIconal.SetIterations(gridIterations);
		gBaseFrameWork.InitializeFramework(&gIconal, "EikonalTest 8");
	}
	else if (argv[1][0] == 'C')
	{
		gIconal.PrepareTest("SourceDoublePoint.tga", "WallDoubleSlit.tga");
		gIconal.SetIterations(gridIterations);
		gBaseFrameWork.InitializeFramework(&gIconal, "EikonalTest 9");
	}
	else if (argv[1][0] == 'D')
	{
		gIconal.PrepareTest("SourceDoublePoint.tga", "WallComplex.tga");
		gIconal.SetIterations(gridIterations);
		gBaseFrameWork.InitializeFramework(&gIconal, "EikonalTest 10");
	}
	else if (argv[1][0] == 'E')
	{
		gIconal.PrepareTest("SourceLine.tga", "WallConvoluted.tga");
		gIconal.SetIterations(60);
		gIconal.SetMaximumTime(300.0f);
		gBaseFrameWork.InitializeFramework(&gIconal, "EikonalTest 11");
	}
	else if (argv[1][0] == 'F')
	{
		gVelocity.PrepareTest("HorrGradient.tga", "WallComplex.tga", 0.03f);
		gBaseFrameWork.InitializeFramework(&gVelocity, "Velocity Test");
	}
	else if (argv[1][0] == 'G')
	{
		gIconal.PrepareTest("SourceLine.tga", "WallSlit.tga", "SpawnAreaSimple.tga");
		gIconal.SetIterations(gridIterations);
		gIconal.SetIsoLineDistance(5.0f);
		gBaseFrameWork.InitializeFramework(&gIconal, "Eikonal Test Density 1");
	}
	else if (argv[1][0] == 'H')
	{
		gIconal.PrepareTest("SourceLine.tga", "WallSlit.tga", "SpawnAreaSimple2.tga");
		gIconal.SetIterations(gridIterations);
		gIconal.SetIsoLineDistance(5.0f);
		gBaseFrameWork.InitializeFramework(&gIconal, "Eikonal Test Density 2");
	}
	else if (argv[1][0] == 'I')
	{
		gIconal.PrepareTest("SourceLine.tga", "WallSlit.tga", "SpawnAreaCircular.tga");
		gIconal.SetIterations(gridIterations);
		gIconal.SetIsoLineDistance(7.0f);
		gBaseFrameWork.InitializeFramework(&gIconal, "Eikonal Test Density 3");
	}
	else if (argv[1][0] == 'J')
	{
		gIconal.PrepareTest("SourceLine.tga", "WallConvoluted.tga", "SpawnAreaCircular.tga");

		gIconal.SetIterations(60);
		gIconal.SetMaximumTime(300.0f);
		
		gIconal.SetIsoLineDistance(7.0f);
		gBaseFrameWork.InitializeFramework(&gIconal, "Eikonal Test Density 4");
	}
	else if (argv[1][0] == 'K')
	{
		gVelocity.PrepareTest("SpawnAreaCircular.tga", "WallConvoluted.tga", 0.2f);
		gBaseFrameWork.InitializeFramework(&gVelocity, "Velocity Test 2");
	}
	else if (argv[1][0] == 'L')
	{
		gGradientTest.PrepareTest("BigCircle.tga", "WallConvoluted.tga", 0);
		gBaseFrameWork.InitializeFramework(&gGradientTest, "Gradient Test 1");
	}
	else if (argv[1][0] == 'M')
	{
		gGradientTest.PrepareTest("BigCircle.tga", "WallConvoluted.tga", 1);
		gBaseFrameWork.InitializeFramework(&gGradientTest, "Gradient Test 2");
	}
	else if (argv[1][0] == 'N')
	{
		gContinuityTest.PrepareTest(0, "Star.tga", "Empty.tga");
		gBaseFrameWork.InitializeFramework(&gContinuityTest, "ContinuityTest 1");
	}
	else if (argv[1][0] == 'O')
	{
		gContinuityTest.PrepareTest(1, "Star.tga", "Empty.tga");
		gBaseFrameWork.InitializeFramework(&gContinuityTest, "ContinuityTest 2");
	}
	else if (argv[1][0] == 'P')
	{
	gContinuityTest.PrepareTest(2, "Star.tga", "Empty.tga");
	gBaseFrameWork.InitializeFramework(&gContinuityTest, "ContinuityTest 3");
	}
	else if (argv[1][0] == 'Q')
	{
	gContinuityTest.PrepareTest(3, "Star.tga", "Empty.tga");
	gBaseFrameWork.InitializeFramework(&gContinuityTest, "ContinuityTest 4");
	}
	else if (argv[1][0] == 'R')
	{
		gContinuityTest.PrepareTest(0, "CenterBlob.tga", "Empty.tga");
		gBaseFrameWork.InitializeFramework(&gContinuityTest, "ContinuityTest 5");
	}
	else if (argv[1][0] == 'S')
	{
		gContinuityTest.PrepareTest(2, "CenterBlob.tga", "Empty.tga");
		gBaseFrameWork.InitializeFramework(&gContinuityTest, "ContinuityTest 6");
	}
	else if (argv[1][0] == 'T')
	{
		gContinuityTest.PrepareTest(3, "TopPixelLine.tga", "Empty.tga");
		gBaseFrameWork.InitializeFramework(&gContinuityTest, "ContinuityTest 7");
	}
	else if (argv[1][0] == 'U')
	{
		gContinuityTest.PrepareTest(0, "CenterBlob.tga", "MicroWall.tga");
		gBaseFrameWork.InitializeFramework(&gContinuityTest, "ContinuityTest 8");
	}
	else if (argv[1][0] == 'V')
	{
		gContinuityTest.PrepareTest(1, "CenterBlob.tga", "MicroWall.tga");
		gBaseFrameWork.InitializeFramework(&gContinuityTest, "ContinuityTest 9");
	}
	else if (argv[1][0] == 'W')
	{
		gContinuityTest.PrepareTest(2, "CenterBlob.tga", "MicroWall.tga");
		gBaseFrameWork.InitializeFramework(&gContinuityTest, "ContinuityTest 10");
	}
	else if (argv[1][0] == 'X')
	{
		gContinuityTest.PrepareTest(3, "CenterBlob.tga", "MicroWall.tga");
		gBaseFrameWork.InitializeFramework(&gContinuityTest, "ContinuityTest 11");
	}
	else if (argv[1][0] == 'Y')
	{
		gContinuityTest.PrepareTest(5, "CenterBlob.tga", "Empty.tga");
		gBaseFrameWork.InitializeFramework(&gContinuityTest, "ContinuityTest 12");
	}
	else if (argv[1][0] == 'Z')
	{
		gContinuityTest.PrepareTest(4, "OuterRing.tga", "Empty.tga");
		gBaseFrameWork.InitializeFramework(&gContinuityTest, "ContinuityTest 13");
	}
	else
		return 0;
	

	
	
	gBaseFrameWork.RunCoreLoop();
	gBaseFrameWork.ShutdownFramework();
	return 0;
	
}


