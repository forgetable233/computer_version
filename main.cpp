//
// Created by dcr on 22-10-6.
//

#include "image_processor.h"

#define PI acos(-1)

int main() {
    std::string file_path = "../bw.jpeg";
    cv::Mat input_image = cv::imread(file_path);
    cv::Mat resized;
    cv::resize(input_image, resized, cv::Size(300, 300));
    cv::Mat t = resized.clone();
    std::vector<cv::Point2i> feature_points;
    std::vector<cv::Mat> features;
    ImageProcessor::HarrisDetector(resized, feature_points, features);
    std::cout << "Have detected " << feature_points.size() << " feature points" << std::endl;
    for (auto &point: feature_points) {
        cv::circle(resized, cv::Point(point.x, point.y), 2, cv::Scalar(0, 0, 255), 2, 8, 0);
    }
    std::cout << input_image.rows << ' ' << input_image.cols << std::endl;
    cv::imshow("points", resized);
    cv::waitKey(0);
    cv::Mat cv_harris;
    cv::Mat grey_img;
    cv::cvtColor(t, grey_img, cv::COLOR_BGR2GRAY);
    cv::cornerHarris(grey_img, cv_harris, 2, 3, 0.01);
    cv::normalize(cv_harris, cv_harris, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(cv_harris, cv_harris);
    for (int i = 0; i < t.rows; ++i) {
        for (int j = 0; j < t.cols; ++j) {
            if (cv_harris.at<double>(i, j) > 130) {
                cv::circle(t, cv::Point(i, j), 2, cv::Scalar(0, 0, 255), 2, 8, 0);
            }
        }
    }
    cv::imshow("harris", t);
    cv::waitKey(0);
//
//    auto target_image = new ImageProcessor(file_path);
//    std::cout << "Please input the target image size" << std::endl;
//    std::cin >> rows >> cols;
//    std::cout << "Begin to process the image" << std::endl;
//    target_image->ResizeImage(512, 512);
//    target_image->CvtToGray();
//    target_image->AddNoise(0, sqrt(10), 0.1);
//
//    const double mean = 0.0;
//    double sigma = 0.1;
//    double sum = 0;
//    double SNR_1[50] = {0};
//    double SNR_2[50] = {0};
//    for (int t = 0; t < 30; ++t) {
//        Eigen::MatrixXd filter_core(9, 9);
//        std::cout << t << std::endl;
//        sum = 0;
//        for (int i = 0; i < 9; ++i) {
//            for (int j = 0; j < 9; ++j) {
//                filter_core(i, j) = (1 / (2 * PI * sigma * sigma)) *
//                                    exp(-(pow(i - 4, 2) + pow(j - 4, 2)) / (2 * sigma * sigma));
//                sum += filter_core(i, j);
//            }
//        }
//        filter_core /= filter_core.sum();
    /** 归一化 **/
//        for (int i = 0; i < 9; ++i) {
//            for (int j = 0; j < 9; ++j) {
//                filter_core(i, j) = filter_core(i, j) / sum;
//            }
//        }
//        std::cout << "Core built" << std::endl;
//        sigma = sigma + 0.1;
//        target_image->MyGaussFilter(filter_core);
//        std::cout << "Finish filter" << std::endl;
//        SNR_1[t] = target_image->ComputeSNR(0);
//        SNR_2[t] = target_image->ComputeSNR(1);
//    }
//    for (int i = 0; i < 30; ++i) {
//        std::cout << SNR_1[i] << ',';
//    }
//    std::cout << std::endl;
//    for (int i = 0; i < 30; ++i) {
//        std::cout << SNR_2[i] << ',';
//    }
//    target_image->ViewImage(GAUSSIAN_FILTERED);
//    target_image->ViewImage(SALT_PEPPER_FILTERED);
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
    return 0;
}