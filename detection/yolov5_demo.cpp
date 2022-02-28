#include "yolov5.h"
#include <jetson-utils/imageIO.h>
#include <jetson-utils/glDisplay.h>
#include <jetson-utils/cudaFont.h>
#include <jetson-utils/cudaDraw.h>
#include <string>
#include <vector>
#include <chrono>

using namespace std;

int main()
{
    const char* image_path = "../bus.jpg";
    string engine_path = "../yolov5n.engine";
    const char* save_path = "../bus_detect.jpg";

    void *img_buffer = NULL;
    int width = 0;
    int height = 0;
    Yolov5* yolo = new Yolov5(engine_path);
    vector<Detection> res;
    cudaFont* font = cudaFont::Create(30);

    for(int i = 0; i < 20; i++)
    {
        // loadImage is time consuming!
        // without loadImage, runtime = 25ms, with loadImage, runtime > 60ms

        loadImage(image_path, &img_buffer, &width, &height, IMAGE_RGB8);
        auto start = std::chrono::system_clock::now();
        yolo->doInference(img_buffer, width, height, res);
        auto end = std::chrono::system_clock::now();
        int runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << runtime << "ms" << std::endl;

    }

    // glDisplay* dis = glDisplay::Create(NULL, width, height);
    // dis->BeginRender();

    int x, y, w, h;
    std::vector< std::pair< std::string, int2 > > labels;
    for (size_t i = 0; i < res.size(); i++)
    {
        x = (int)res[i].bbox[0];
        y = (int)res[i].bbox[1];
        w = (int)res[i].bbox[2];
        h = (int)res[i].bbox[3];
        float4 color = make_float4(0.0f, 0.0f, 255.0f, 255.0f);
        cudaDrawLine(img_buffer, width, height, IMAGE_RGB8, x, y, x+w, y, color);
        cudaDrawLine(img_buffer, width, height, IMAGE_RGB8, x, y, x, y+h, color);
        cudaDrawLine(img_buffer, width, height, IMAGE_RGB8, x+w, y, x+w, y+h, color);
        cudaDrawLine(img_buffer, width, height, IMAGE_RGB8, x, y+h, x+w, y+h, color);

        string a = "class ";
        string b = to_string((int)res[i].class_id);
        labels.push_back(std::pair<std::string, int2>(a+b, {x+5,y+5}));
    }

    font->OverlayText(img_buffer, IMAGE_RGB8, width, height, labels, make_float4(255,0,0,255));

    // dis->RenderImage(img_buffer, width, height, IMAGE_RGB8, 0, 0);
    // dis->EndRender();

    saveImage(save_path, img_buffer, width, height, IMAGE_RGB8);

    cudaFreeHost(img_buffer);
    SAFE_DELETE(yolo);
    SAFE_DELETE(font);
    // SAFE_DELETE(dis);

    return 0;
}