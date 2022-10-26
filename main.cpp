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
    int rows, cols;

    auto target_image = new ImageProcessor(file_path);
    std::cout << "Please input the target image size" << std::endl;
//    std::cin >> rows >> cols;
    std::cout << "Begin to process the image" << std::endl;
    target_image->ResizeImage(512, 512);
    target_image->CvtToGray();
    target_image->AddNoise(0, sqrt(10), 0.1);

    const double mean = 0.0;
    double sigma = 0.1;
    double sum = 0;
    double SNR_1[50] = {0};
    double SNR_2[50] = {0};
    for (int t = 0; t < 30; ++t) {
        Eigen::MatrixXd filter_core(9, 9);
        std::cout << t << std::endl;
        sum = 0;
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                filter_core(i, j) = (1 / (2 * PI * sigma * sigma)) *
                                    exp(-(pow(i - 4, 2) + pow(j - 4, 2)) / (2 * sigma * sigma));
                sum += filter_core(i, j);
            }
        }
        filter_core /= filter_core.sum();
        /** 归一化 **/
//        for (int i = 0; i < 9; ++i) {
//            for (int j = 0; j < 9; ++j) {
//                filter_core(i, j) = filter_core(i, j) / sum;
//            }
//        }
        std::cout << "Core built" << std::endl;
        sigma = sigma + 0.1;
        target_image->MyGaussFilter(filter_core);
        std::cout << "Finish filter" << std::endl;
        SNR_1[t] = target_image->ComputeSNR(0);
        SNR_2[t] = target_image->ComputeSNR(1);
    }
    for (int i = 0; i < 30; ++i) {
        std::cout << SNR_1[i] << ',';
    }
    std::cout << std::endl;
    for (int i = 0; i < 30; ++i) {
        std::cout << SNR_2[i] << ',';
    }
    target_image->ViewImage(GAUSSIAN_FILTERED);
    target_image->ViewImage(SALT_PEPPER_FILTERED);
//    cv::Mat srcImg;
//    cv::Mat dstImg;
//    target_image->GetGaussImage(srcImg);
////    target_image->MiddleFilter(srcImg, dstImg);
//    cv::medianBlur(srcImg, dstImg, 3);
//    cv::imshow("result3", dstImg);
//    cv::waitKey(0);
//    target_image->GetSaltImage(srcImg);
////    target_image->MiddleFilter(srcImg, dstImg);
//    cv::medianBlur(srcImg, dstImg, 3);
//    cv::imshow("result4", dstImg);
//    cv::waitKey(0);
////    target_image->ComputeSNR()
////    ImageProcessor::ViewImage(dstImg);
//    target_image->GetSaltImage(srcImg);
//    target_image->MiddleFilter(srcImg, dstImg);
//    ImageProcessor::ViewImage(srcImg);
//    ImageProcessor::ViewImage(dstImg);
//    cv::Mat edge_detect;
//    target_image->SobelDetector(edge_detect);
//
//    cv::imshow("M", edge_detect);
//    cv::waitKey(0);
}