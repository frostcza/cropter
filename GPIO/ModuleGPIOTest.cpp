#include <chrono>
#include <thread>
#include <queue>
#include <jetson-utils/Mutex.h>
#include <jetson-utils/Thread.h>
#include <signal.h>
#include <sys/time.h>
#include <JetsonGPIO.h>

/* It's just a test! The major functions are migrated to ModuleGPIO */

enum mode { IR = 0, VI = 1, Fuse = 2, Unknown = 3 };
static bool end_this_program = false;
void signalHandler(int s) { end_this_program = true; }
void delay(int s) { std::this_thread::sleep_for(std::chrono::seconds(s)); }


int main()
{
    signal(SIGINT, signalHandler);
    int pwm_input = 11; // BOARD pin 11
    int save_image_flag = 12; 
    GPIO::setmode(GPIO::BOARD);
    GPIO::setup(pwm_input, GPIO::IN);
    GPIO::setup(save_image_flag, GPIO::IN);
    struct timeval start, end;
    mode result;

    std::queue<mode> mode_queue;
    int countIR = 10;
    int countVI = 0;
    int countFuse = 0;
    for(int i=0; i<10; i++)
    {
        mode_queue.push(IR);
    }

    int test_pin = 12;

    while (!end_this_program) 
    {
        if(test_pin == 11)
        {
            printf("start to listen...\n");
            GPIO::wait_for_edge(pwm_input, GPIO::Edge::RISING);
            gettimeofday(&start,NULL);
            GPIO::wait_for_edge(pwm_input, GPIO::Edge::FALLING);
            gettimeofday(&end,NULL);
            int high = end.tv_usec - start.tv_usec;
            printf("high voltage time: %d us\n", high);

            if(high>800 && high<=1200)
            {
                mode_queue.push(IR);
                countIR++;
            }
            else if(high>1200 && high<=1600)
            {
                mode_queue.push(VI);
                countVI++;
            }
            else if(high>1600 && high<=2100)
            {
                mode_queue.push(Fuse);
                countFuse++;
            }
            else
            {
                continue;
            }
            switch(mode_queue.front())
            {
                case IR:
                {
                    countIR--;
                    break;
                }
                case VI:
                {
                    countVI--;
                    break;
                }
                case Fuse:
                {
                    countFuse--;
                    break;
                }
                default:
                    break;
            }
            mode_queue.pop();

            if(countIR >= 6) result = IR;
            if(countVI >= 6) result = VI;
            if(countFuse >= 6) result = Fuse;
            printf("mode: %d\n", result);
            //delay(1);
        }
        else if(test_pin == 12)
        {
            printf("start to listen...\n");
            GPIO::wait_for_edge(save_image_flag, GPIO::Edge::RISING);
            gettimeofday(&start,NULL);
            GPIO::wait_for_edge(save_image_flag, GPIO::Edge::FALLING);
            gettimeofday(&end,NULL);
            int high = end.tv_usec - start.tv_usec;
            printf("high voltage time: %d us\n", high);
            if(high>1500 && high<=2000)
            {
                printf("received save flag\n");
                //delay(3);
            }
            //delay(1);
        }
        
    }

    GPIO::cleanup();

    return 0;
}