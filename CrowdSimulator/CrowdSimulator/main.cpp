#include "FrameWork.h"
#include "CheckerTest.h"
#include "FlagTest.h"
#include "IsoLineTest.h"
#include "IconalTest.h"
#include "VelocityTest.h"
#include "GradientTest.h"

static FrameWork gBaseFrameWork;
static CheckerTest gChecker;
static FlagTest gFlagTest;
static IsoLineTest gIsoLineTest;
static IconalTest gIconal;
static VelocityTest gVelocity;
static GradientTest gGradientTest;

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
		gIsoLineTest.LoadScalarPicture("Circle.tga");
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
		gGradientTest.InitializeTest("BigCircle.tga", "WallConvoluted.tga", 0);
		gBaseFrameWork.InitializeFramework(&gGradientTest, "Gradient Test 1");
	}
	else if (argv[1][0] == 'M')
	{
		gGradientTest.InitializeTest("BigCircle.tga", "WallConvoluted.tga", 1);
		gBaseFrameWork.InitializeFramework(&gGradientTest, "Gradient Test 2");
	}
	else
		return 0;
	

	
	
	gBaseFrameWork.RunCoreLoop();
	gBaseFrameWork.ShutdownFramework();
	return 0;
	
}


