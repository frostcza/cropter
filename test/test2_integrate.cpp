
#include <signal.h>
#include <chrono>

#include "integrate.h"

volatile bool signal_recieved = false;
mode keyboard_mode = Fuse;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}

void sig_handler_mode(int signo)
{
	if( signo == SIGTSTP )
	{
		printf("mode switch\n");
		if (keyboard_mode == IR)
			keyboard_mode = VI;
		else if (keyboard_mode == VI)
			keyboard_mode = Fuse;
		else if (keyboard_mode == Fuse)
			keyboard_mode = IR;
	}
}

void test()
{
	RunOption opt;
	opt.use_GPIO = false;
	opt.shrink_picture = false;
	Integrate* integrate = new Integrate(opt); 

	// The main cycle
	while(!signal_recieved)
	{
		if(!opt.use_GPIO)
			integrate->gpio->mode_result = keyboard_mode;
		
		integrate->mainLoop();

		if(integrate->dis->IsClosed())
        	signal_recieved = true;
	}
	delete integrate;
	integrate = NULL;
	printf("[Test] shut down\n");
}

int main(int argc, char** argv)
{
    // catch Ctrl + C signal
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("can't catch SIGINT\n");

	// catch Ctrl + Z signal
	if( signal(SIGTSTP, sig_handler_mode) == SIG_ERR)
		printf("can't catch SIGTSTP");
	
	test();
	return 0;
}
