#pragma once

#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Image{
private:
    static unsigned int m_uniq_id;

    std::shared_ptr<cv::Mat> m_mat;
    std::vector< std::vector<cv::Point>> m_contours;
    unsigned int m_id = m_uniq_id++;

    /**
     * Deskew objects
     */
    void _deskew(cv::Mat &mat);
public:
    Image(std::shared_ptr<cv::Mat> mat) : m_mat(mat){}

    /**
     * Display image
     */
    void display();

    /**
     * Clean background noise
     */
    void cleanNoise();

    /**
     * Extract letters/digits from image
     */
    void extract();

    /**
     * Conver to binary
     */
    void binarize();

    /**
     * Draw contour around objects
     */
    void contour();

    /**
     * Recreate matrix
     * @param mat
     * @return recreated matrix
     */
    std::shared_ptr<cv::Mat> recreate();

    /**
     * Close windows on input
     */
    static void wait();
};