//
// Created by dcr on 22-10-6.
//
#include <iostream>
#include <algorithm>
#include <string>
#include <random>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#ifndef COMPUTER_VISION_IMAGE_PROCESSOR_H
#define COMPUTER_VISION_IMAGE_PROCESSOR_H

enum ImageChoose {
    ORIGIN,
    RESIZED,
    GRAY,
    GAUSSIAN_NOISE,
    SALT_PEPPER_NOISE,
    NOISE_COMBINED
};

class ImageProcessor {
private:
    cv::Mat image_;
    cv::Mat resized_image_;
    cv::Mat grey_image_;
    cv::Mat gaussian_noise_image_;
    cv::Mat salt_pepper_noise_image_;
    cv::Mat filtered_image_;

    bool have_finished_resize_ = false;
public:
    ImageProcessor() = default;

    explicit ImageProcessor(const std::string &file_path);

    void ViewImage(ImageChoose choose);

    void ResizeImage(int row, int cols);

    void CvtToGray();

    void AddNoise(const double mean, const double sigma, const double dis);

    void Filter(Eigen::Matrix3d filter_core);
};


#endif //COMPUTER_VISION_IMAGE_PROCESSOR_H
