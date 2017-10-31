#pragma once

#include <memory>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Image{
private:
    static unsigned int m_uniq_id;

    std::shared_ptr<cv::Mat> m_mat;
    std::vector<std::vector<cv::Point>> m_contours;
    unsigned int m_id = m_uniq_id++;

    /**
     * Deskew objects
     */
    void _deskew(cv::Mat &mat);

    /**
     * Generate matrix permutatios
     * @param images
     * @param indices
     */
    void _permutation(std::vector<std::shared_ptr<Image>> &images, std::vector<int> &indices);

    /**
     * Generate a new image
     * @parm indcies
     * @return generate image
     */
    std::shared_ptr<Image> _buildImage(const std::vector<int> &indices);
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
     * Conver to binary
     */
    void binarize();

    /**
     * Detect elements in matrix
     */
    void detectElements();

    /**
     * Draw contour around objects
     */
    void contour();

    /**
     * Recreate matrices
     * @param mat
     * @return vector of matrices
     */
    std::vector<std::shared_ptr<Image>> permutation();

    /**
     * Close windows on input
     */
    static void wait();
};