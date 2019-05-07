#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <stdio.h>
#include <iostream>
#include <limits.h>
#include <fstream>

using namespace cv;
using namespace std;


int** calc_disp(string img1_filename, string img2_filename, int window_size, int l_offset, int r_offset, int* rows, int* cols) {
    int i = 0, j = 0, k = 0, h, w;
    int base = window_size / 2;
    int best_point[2] = {0, 0};
    int disp = 0;
    int best_diff = INT_MAX;
    int diff = 0;
    Mat window0(window_size, window_size, CV_8UC1);
    Mat window1(window_size, window_size, CV_8UC1);
    

    Mat img1 = imread(img1_filename, CV_8UC1);
    Mat img2 = imread(img2_filename, CV_8UC1);

    h = img1.rows - 2 * base;
    w = img1.cols - 2 * base;

    *rows = h;
    *cols = w;

    int** disparity = new int*[h];
    for (int i = 0; i < h; ++i)
        disparity[i] = new int[w];

    for (i = base; i < h + base; i++){
        if (i % (h / 10) == 0) {
            cout << (100 * i)/h << "%" << endl;
        }
        for (j = base; j < w + base; j++){
            window0 = img1(Range(i - base, i + base), Range(j - base, j + base));
            
            best_point[0] = 0;
            best_point[1] = 0;
            best_diff = INT_MAX;

            for (k = min(j, max(base, j - l_offset)); k < min(j + r_offset, w + base); k+= 1){
                window1 = img2(Range(i - base, i + base), Range(k - base, k + base));

                diff = sum(abs(window0 - window1))[0];

                if (diff < best_diff){
                    best_diff = diff;
                    best_point[0] = i;
                    best_point[1] = k;
                }
            }
            disp = best_point[1] - j;
            disparity[i-base][j-base] = disp;
        }
    }
    return disparity;
}

int main() {
    int ** disp;
    int rows, cols;
    int i, j;
    
    // Ajustar na chamada abaixo os parâmetros de teste
    disp = calc_disp("data/Middlebury/Jadeplant-perfect/im0.png", "data/Middlebury/Jadeplant-perfect/im1.png", 15, 640, 0, &rows, &cols);

    int min = INT_MAX;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if (disp[i][j] < min) {
                min = disp[i][j];
            }
        }
    }

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if (disp[i][j] < min) {
                disp[i][j] -= min;
            }
        }
    }

    int max = INT_MIN;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if (disp[i][j] > max) {
                max = disp[i][j];
            }
        }
    }

    Mat dispar(rows, cols, CV_8UC1);

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            dispar.at<unsigned char>(i, j) = (unsigned char) (255 * disp[i][j]) / max;
        }
    }

    imshow("Disparidade", dispar);
    waitKey();
    imwrite("data/output/dispSAD.png", dispar);

    // Ele salva em um arquivo txt os dados brutos
    // para abrir depois caso preciso.
    // Não quero rodar esse programa muitas vezes,
    // ele é lento.
    // Pode abrir o arquivo em um programa em Python,
    // que é mais fácil de lidar com os dados.
    ofstream disparity("data/output/dispSAD.txt", ofstream::out);
    if (disparity.is_open()) {
        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++){
                disparity << disp[i][j] << ' ';
            }
            disparity << endl;
        }
        disparity.close();
    }
    else{
        cout << "Não conseguiu abrir o arquivo" << endl;
    }


    for (int i = 0; i < rows; ++i)
        delete [] disp[i];
    delete [] disp;

    return 0;
}
