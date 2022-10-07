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
            cv::imshow("salt and pepper noise image", salt_pepper_noise_image_);
            break;
        case GAUSSIAN_FILTERED:
            std::cout << "The gaussian filtered noise image" << std::endl;
            cv::imshow("gaussian filtered image", gaussian_filtered_image_);
            break;
        case SALT_PEPPER_FILTERED:
            std::cout << "The salt and pepper filtered noise image" << std::endl;
            cv::imshow("salt and pepper filtered image", salt_pepper_filtered_image_);
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
        finished_resize_ = true;
        std::cout << "==============================================================" << std::endl;
        std::cout << "Have finished the resize of the image" << std::endl;
        std::cout << "==============================================================" << std::endl << std::endl;
    } else {
        std::cerr << "The target rows and columns are too large" << std::endl;
    }
}

/**
 * 将彩色图像转化为灰度图，这里使用到了加权平均法
 */
void ImageProcessor::CvtToGray() {
    if (!finished_resize_) {
        std::cerr << "没有完成图像resize" << std::endl;
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
    std::cout << "==============================================================" << std::endl;
    std::cout << "完成灰度图转化" << std::endl;
    std::cout << "==============================================================" << std::endl << std::endl;
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

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean, sigma);

    for (int i = 0; i < grey_image_.rows; ++i) {
        for (int j = 0; j < grey_image_.cols; ++j) {
            /** 首先进行高斯噪音的添加 **/
            gaussian_noise_image_.at<u_char>(i, j) = static_cast<u_char>(grey_image_.at<u_char>(i, j) +
                                                                         distribution(generator));
            /** 下面进行椒盐分布噪音的添加 **/
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
    std::cout << "==============================================================" << std::endl;
    std::cout << "已完成图像噪声的添加" << std::endl;
    std::cout << "==============================================================" << std::endl << std::endl;
    noise_added_ = true;
}

void ImageProcessor::MyGaussFilter(const Eigen::Matrix3d &filter_core) {
    if (!noise_added_) {
        std::cerr << "没有完成图像噪声的添加" << std::endl;
        return;
    }
    const int rows = gaussian_noise_image_.rows;
    const int cols = gaussian_noise_image_.cols;

    Eigen::MatrixXd gaussian_matrix(rows + 2, cols + 2);
    Eigen::MatrixXd salt_matrix(rows + 2, cols + 2);
    Eigen::MatrixXd mat2eigen(rows, cols);
    Eigen::MatrixXd gaussian_result_matrix(rows, cols);
    Eigen::MatrixXd salt_result_matrix(rows, cols);
    Eigen::Matrix3d temp_matrix;

    cv::cv2eigen(gaussian_noise_image_, mat2eigen);
    gaussian_matrix.block(1, 1, rows, cols) = mat2eigen;
    cv::cv2eigen(salt_pepper_noise_image_, mat2eigen);
    salt_matrix.block(1, 1, rows, cols) = mat2eigen;

    for (int i = 1; i < rows + 1; ++i) {
        for (int j = 1; j < cols + 1; ++j) {
            /** 首先进行高斯的卷积 **/
            temp_matrix = gaussian_matrix.block<3, 3>(i - 1, j - 1);
            gaussian_result_matrix(i - 1, j - 1) = Convolution(filter_core, temp_matrix);
            /** 下面进行盐椒的卷积 **/
            temp_matrix = salt_matrix.block<3, 3>(i - 1, j - 1);
            salt_result_matrix(i - 1, j - 1) = Convolution(filter_core, temp_matrix);
        }
    }
    cv::eigen2cv(gaussian_result_matrix, gaussian_filtered_image_);
    cv::eigen2cv(salt_result_matrix, salt_pepper_filtered_image_);
    gaussian_filtered_image_.convertTo(gaussian_filtered_image_, CV_8UC1);
    salt_pepper_filtered_image_.convertTo(salt_pepper_filtered_image_, CV_8UC1);
}

double ImageProcessor::Convolution(const Eigen::Matrix3d &core, Eigen::Matrix3d &matrix) {
    double re = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            re += core(i, j) * matrix(i, j);
        }
    }
    return re;
}

double ImageProcessor::ComputeSNR(int choose) {
    Eigen::MatrixXd origin_image(resized_image_.rows, resized_image_.cols);
    Eigen::MatrixXd filtered_image(resized_image_.rows, resized_image_.cols);
    cv::cv2eigen(grey_image_, origin_image);
    if (choose == 0) {
        cv::cv2eigen(gaussian_filtered_image_, filtered_image);
    } else if (choose == 1) {
        cv::cv2eigen(salt_pepper_filtered_image_, filtered_image);
    }
    return 20 * log(origin_image.norm() / (origin_image - filtered_image).norm());
}

void ImageProcessor::GetGaussImage(cv::Mat &dst_img) {
    this->gaussian_noise_image_.copyTo(dst_img);
}

void ImageProcessor::GetSaltImage(cv::Mat &dst_img) {
    this->salt_pepper_noise_image_.copyTo(dst_img);
}

void ImageProcessor::MiddleFilter(cv::Mat &srcImg, cv::Mat &dstImg) {
    const int rows = srcImg.rows;
    const int cols = srcImg.cols;
    Eigen::MatrixXd src_eigen(rows, cols);
    Eigen::MatrixXd dst_eigen(rows, cols);
    Eigen::MatrixXd temp_Eigen(rows + 2, cols + 2);

    cv::cv2eigen(srcImg, src_eigen);
    temp_Eigen.block(1, 1, rows, cols) = src_eigen;
    for (int i = 1; i < rows + 1; ++i) {
        for (int j = 1; j < cols + 1; ++j) {
            Eigen::Matrix3d temp = temp_Eigen.block<3, 3>(i - 1, j - 1);
            dst_eigen(i - 1, j - 1) = GetMiddleValue(temp);
        }
    }
    cv::eigen2cv(dst_eigen, dstImg);
    dstImg.convertTo(dstImg, CV_8UC1);
    std::cout << "==============================================================" << std::endl;
    std::cout << "已完成图像中值去噪声" << std::endl;
    std::cout << "==============================================================" << std::endl << std::endl;
}

void ImageProcessor::ViewImage(const cv::Mat &image) {
    std::cout << image.rows << ' ' << image.cols << std::endl;
    cv::imshow("input image", image);
    cv::waitKey(0);
}

double ImageProcessor::GetMiddleValue(Eigen::Matrix3d &matrix) {
    double array[] = {matrix(0, 0), matrix(0, 1), matrix(0, 2),
                      matrix(1, 0), matrix(1, 1), matrix(1, 2),
                      matrix(2, 0), matrix(2, 1), matrix(2, 2)};
    std::sort(array, array + 9);
    return array[4];
}
