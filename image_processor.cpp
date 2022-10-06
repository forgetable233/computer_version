//
// Created by dcr on 22-10-6.
//

#include "image_processor.h"

ImageProcessor::ImageProcessor(const std::string &file_path) {
    image_ = cv::imread(file_path);

    std::cout << "Have read the target image." << std::endl;
    std::cout << "The size of the image is : row = " << image_.rows << " col = " << image_.cols << std::endl;
}

/**
 * 为方便调试增加的查看图片函数
 * @param choose
 */
void ImageProcessor::ViewImage(ImageChoose choose) {
    switch (choose) {
        case ORIGIN:
            std::cout << "The origin image" << std::endl;
            cv::imshow("origin image", image_);
            break;
        case RESIZED:
            std::cout << "The resized image" << std::endl;
            cv::imshow("resized image", resized_image_);
            break;
        case GRAY:
            std::cout << "The grey image" << std::endl;
            cv::imshow("grey image", grey_image_);
            break;
        case GAUSSIAN_NOISE:
            std::cout << "The gaussian noise image" << std::endl;
            cv::imshow("gaussian noise image", gaussian_noise_image_);
            break;
        case SALT_PEPPER_NOISE:
            std::cout << "The salt and pepper noise image" << std::endl;
            cv::imshow("gaussian noise image", salt_pepper_noise_image_);
            break;
        default:
            std::cerr << "Can not find the target image" << std::endl;
            return;
    }
    cv::waitKey(0);
}

/**
 * 对图像进行resize，如需要放大则使用插值进行，还没有完成相关代码
 * @param row
 * @param cols
 */
void ImageProcessor::ResizeImage(int row, int cols) {
    if (row <= image_.rows && cols <= image_.cols) {
        cv::Mat temp_resized_image(cv::Size(512, 512), CV_8UC3);
        temp_resized_image.copyTo(resized_image_);
        double row_jump = static_cast<double>(image_.rows) / row;
        double cols_jump = static_cast<double>(image_.cols) / cols;

        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < cols; ++j) {
                resized_image_.at<cv::Vec3b>(i, j) =
                        image_.at<cv::Vec3b>(static_cast<int>(i * row_jump), static_cast<int>(j * cols_jump));
            }
        }
        have_finished_resize_ = true;
        std::cout << "Have finished the resize of the image" << std::endl;
    } else {
        std::cerr << "The target rows and columns are too large" << std::endl;
    }
}

/**
 * 将彩色图像转化为灰度图，这里使用到了加权平均法
 */
void ImageProcessor::CvtToGray() {
    if (!have_finished_resize_) {
        std::cerr << "Haven't finished the resize" << std::endl;
        return;
    }
    cv::Mat temp_grey_image(cv::Size(resized_image_.rows, resized_image_.cols), CV_8UC1);
    temp_grey_image.copyTo(grey_image_);
    for (int i = 0; i < resized_image_.rows; ++i) {
        for (int j = 0; j < resized_image_.cols; ++j) {
            grey_image_.at<u_char>(i, j) = static_cast<u_char>(
                    static_cast<double>(resized_image_.at<cv::Vec3b>(i, j)[0]) * 0.11 +
                    static_cast<double>(resized_image_.at<cv::Vec3b>(i, j)[1]) * 0.59 +
                    static_cast<double>(resized_image_.at<cv::Vec3b>(i, j)[2]) * 0.3);
        }
    }
    std::cout << "Have finished converting to grey image" << std::endl;
}

/**
 * 对图片进行添加噪声的处理
 * @param mean
 * @param sigma
 * @param dis
 */
void ImageProcessor::AddNoise(const double mean, const double sigma, const double dis) {
    cv::Mat temp_image(cv::Size(grey_image_.rows, grey_image_.cols), CV_8UC1);
    grey_image_.copyTo(gaussian_noise_image_);
    grey_image_.copyTo(salt_pepper_noise_image_);

    /** 首先进行高斯噪音的添加 **/
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean, sigma);

    for (int i = 0; i < grey_image_.rows; ++i) {
        for (int j = 0; j < grey_image_.cols; ++j) {
            gaussian_noise_image_.at<u_char>(i, j) = static_cast<u_char>(grey_image_.at<u_char>(i, j) +
                                                                         distribution(generator));
        }
    }

    /** 下面进行椒盐分布噪音的添加 **/
    for (int i = 0; i < grey_image_.rows; ++i) {
        for (int j = 0; j < grey_image_.cols; ++j) {
            if (rand() <= RAND_MAX * dis) {
                if (rand() <= RAND_MAX / 2) {
                    salt_pepper_noise_image_.at<u_char>(i, j) = 0;
                } else {
                    salt_pepper_noise_image_.at<u_char>(i, j) = 255;
                }
            } else {
                salt_pepper_noise_image_.at<u_char>(i, j) = grey_image_.at<u_char>(i, j);
            }
        }
    }
}

void ImageProcessor::Filter(Eigen::Matrix3d filter_core) {

}
