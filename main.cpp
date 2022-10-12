//
// Created by dcr on 22-10-6.
//
#include <iostream>
#include <string>
#include <cmath>
#include <stdio.h>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "image_processor.h"

#define PI acos(-1)

int main() {
    std::string file_path = "../picture.jpg";

    auto target_image = new ImageProcessor(file_path);
    target_image->ResizeImage(512, 512);
    target_image->CvtToGray();
    target_image->AddNoise(0, std::sqrt(10), 0.1);

    Eigen::Matrix3d filter_core;
    const double mean = 0.0;
    const double sigma = 1;
    double sum = 0;
    /** 构建卷积核，此处为 mean = 0, sigma = 1 **/
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            filter_core(i, j) = (1 / std::sqrt(2 * PI * sigma * sigma)) *
                                exp(-pow((std::sqrt(pow(i - 1, 2) + pow(j - 1, 2) - mean)), 2) / (2 * sigma * sigma));
            sum += filter_core(i, j);
        }
    }
    /** 归一化 **/
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            filter_core(i, j) = filter_core(i, j) / sum;
        }
    }
    target_image->MyGaussFilter(filter_core);
    std::cout << target_image->ComputeSNR(0) << std::endl;
    std::cout << target_image->ComputeSNR(1) << std::endl;

    cv::Mat srcImg;
    cv::Mat dstImg;
    target_image->GetGaussImage(srcImg);
    target_image->MiddleFilter(srcImg, dstImg);
//    ImageProcessor::ViewImage(dstImg);
    target_image->GetSaltImage(srcImg);
    target_image->MiddleFilter(srcImg, dstImg);
//    ImageProcessor::ViewImage(srcImg);
//    ImageProcessor::ViewImage(dstImg);
    cv::Mat edge_detect;
    target_image->SobelDetector(edge_detect);
//    cv::imshow("The edge", edge_detect);
}