//
// Created by dcr on 22-10-6.
//

#include "image_processor.h"

#define PI acos(-1)

int main() {
    std::string file_path = "../picture.jpg";
    cv::Mat input_image = cv::imread(file_path);
    cv::Mat resized;
    cv::resize(input_image, resized, cv::Size(512, 512));
    std::vector<cv::Point2i> feature_points;
    std::vector<cv::Mat> features;
    ImageProcessor::HarrisDetector(resized, feature_points, features);
    for (auto &point: feature_points) {
        resized.at<cv::Vec3b>(point.x, point.y)[0] = 0;
        resized.at<cv::Vec3b>(point.x, point.y)[1] = 0;
        resized.at<cv::Vec3b>(point.x, point.y)[2] = 0xff;
    }
    std::cout << input_image.rows << ' ' << input_image.cols << std::endl;
    cv::imshow("points", resized);
    cv::waitKey(0);
    return 0;
}