#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/glDisplay.h>
#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/imageIO.h>
#include "LiteGstCamera.h"

#include <stdio.h>
#include <signal.h>

bool signal_recieved = false;


void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}


int main( int argc, char** argv )
{
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("can't catch SIGINT\n");

	void* imgRGB = NULL;
    int framenum = 0;
    int counter = 0;
    char filename[50];

	LiteGstCamera* cameraVIS = LiteGstCamera::Create(1920, 1080, "/dev/video0");
	if( !cameraVIS )
	{
		printf("[cameraVIS] failed to initialize VIS camera\n");
		return 0;
	}

	glDisplay* dis = glDisplay::Create(NULL, 800, 600);
	
	if( !cameraVIS->Open() )
	{
		printf("[cameraVIS] failed to open VIS camera\n");
		return 0;
	}
	printf("[cameraVIS] VIS camera streaming\n");
	
	while( !signal_recieved )
	{
		if( !cameraVIS->Capture(&imgRGB, IMAGE_RGB8, 1000))
			printf("[cameraVIS] failed to capture RGB image\n");

        dis->BeginRender();
        dis->RenderImage(imgRGB, cameraVIS->GetWidth(), cameraVIS->GetHeight(), IMAGE_RGB8, 0, 0);
        dis->EndRender();
        counter++;
        if(counter == 15)
        {
            framenum++;
            sprintf(filename, "../../saved_image/VIS/VIS-%08d.jpg", framenum);
            saveImage(filename, imgRGB, cameraVIS->GetWidth(), cameraVIS->GetHeight(), IMAGE_RGB8, 90);
            counter = 0;
        }

        char str[256];
        sprintf(str, "Camera Viewer | %.0f FPS",  dis->GetFPS());
        dis->SetTitle(str);	

        if( dis->IsClosed() )
            signal_recieved = true;
	}
	
	
	SAFE_DELETE(cameraVIS);
	SAFE_DELETE(dis);
	printf("[cameraVIS] shutdown complete.\n");
	return 0;
}
