//
// Created by dcr on 22-10-6.
//
#include <iostream>
#include <algorithm>
#include <string>
#include <random>
#include <cmath>

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

    static void ViewImage(const cv::Mat &image);

    void ResizeImage(int row, int cols);

    void CvtToGray();

    void AddNoise(const double mean, const double sigma, const double dis);

    void MyGaussFilter(const Eigen::Matrix3d &filter_core);

    void MiddleFilter(cv::Mat &srcImg, cv::Mat &dstImg);

    void GetGaussImage(cv::Mat &dst_img);

    void GetSaltImage(cv::Mat &dst_img);

    static double Convolution(const Eigen::Matrix3d &core, Eigen::Matrix3d &matrix);

    double ComputeSNR(int choose);

    static inline double GetMiddleValue(Eigen::Matrix3d &matrix);

    void SobelDetector(cv::Mat &dstImg);

    void NMS(Eigen::Matrix<uchar, -1, -1> &M, Eigen::MatrixXd &angle);
};


#endif //COMPUTER_VISION_IMAGE_PROCESSOR_H
