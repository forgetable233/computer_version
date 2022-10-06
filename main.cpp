//
// Created by dcr on 22-10-6.
//
#include <iostream>
#include <string>
#include <cmath>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "image_processor.h"

int main() {
    std::string file_path = "/home/dcr/CLionProjects/computer_version/picture.jpg";

    auto target_image = new ImageProcessor(file_path);
    target_image->ResizeImage(512, 512);
    target_image->CvtToGray();
    target_image->AddNoise(0, std::sqrt(10), 0.1);
}