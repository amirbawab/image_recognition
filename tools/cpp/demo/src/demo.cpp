#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <sstream>
#include <string>

using namespace cv;
using namespace std;

int main( int argc, char** argv ) {
    std::ifstream inExample("/tmp/img.csv");
    int rows = 64;
    int cols = 64;
    std::string lineMat;
    while (std::getline(inExample, lineMat)) {
        Mat out = Mat::zeros(rows, cols, CV_64F);
        int row = 0;
        int col = 0;
        std::istringstream iss(lineMat);
        std::string token;
        while(std::getline(iss, token, ',')) {
            out.at<double>(row, col) = atof(token.c_str())/100;
            col++;
            if(col == 64) {
                row++;
                col = 0;
            }
        }

        std::cout << out << std::endl;
        std::stringstream winName;
        winName << "Image " << row << ":" << col;
        namedWindow(winName.str(), WINDOW_AUTOSIZE );
        imshow(winName.str(), out);
    }
    waitKey(8000);
    return 0;
}

