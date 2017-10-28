#pragma once

#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Image{
private:
    std::shared_ptr<cv::Mat> m_mat;
    static unsigned int m_uniq_id;
    unsigned int m_id = m_uniq_id++;
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
     * Draw contour around objects
     */
    void contour();

    /**
     * Close windows on input
     */
    static void wait();
};