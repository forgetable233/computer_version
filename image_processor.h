//
// Created by dcr on 22-10-6.
//
#include <iostream>
#include <algorithm>
#include <string>
#include <random>

#include <Eigen/Eigen>
#include <Eigen/Dense>
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
    GAUSSIAN_FILTERED,
    SALT_PEPPER_FILTERED
};

class ImageProcessor {
private:
    cv::Mat image_;
    cv::Mat resized_image_;
    cv::Mat grey_image_;
    cv::Mat gaussian_noise_image_;
    cv::Mat salt_pepper_noise_image_;
    cv::Mat gaussian_filtered_image_;
    cv::Mat salt_pepper_filtered_image_;

    bool finished_resize_ = false;
    bool noise_added_ = false;
public:
    ImageProcessor() = default;

    explicit ImageProcessor(const std::string &file_path);

    void ViewImage(ImageChoose choose);

    void ResizeImage(int row, int cols);

    void CvtToGray();

    void AddNoise(const double mean, const double sigma, const double dis);

    // TODO 更改函数格式，使用template
    void MyGaussFilter(const Eigen::Matrix3d &filter_core);

    static double Convolution(const Eigen::Matrix3d &core, Eigen::Matrix3d &matrix);

    double ComputeSNR(int choose);
};


#endif //COMPUTER_VISION_IMAGE_PROCESSOR_H
