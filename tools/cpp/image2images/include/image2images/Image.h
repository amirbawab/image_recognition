#pragma once

#include <memory>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define NO_VALUE 777

class Image{
private:
    static long m_uniq_id;

    std::shared_ptr<cv::Mat> m_mat;
    std::vector<std::vector<cv::Point>> m_contours;
    int m_value;
    long m_id = m_uniq_id++;

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
     * @param rows
     * @param cols
     * @parm indcies
     * @return generate image
     */
    std::shared_ptr<Image> _buildImage(int rows, int cols, const std::vector<int> &indices);
public:
    Image(std::shared_ptr<cv::Mat> mat, int value = NO_VALUE) : m_mat(mat), m_value(value){}

    /**
     * Display image
     */
    void display();

    /**
     * Clean background noise
     */
    void cleanNoise();

    /**
     * Convert to binary
     * @param threshold
     */
    void binarize(int threshold);

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
     * Align objects
     * @return Image pointer
     */
    std::shared_ptr<Image> align();

    /**
     * Close windows on input
     */
    static void wait();

    /**
     * Get value
     * @return value or -1 if not set
     */
    int getValue() const {return m_value;}

    /**
     * Generate an image per detected element
     * @return vector of images
     */
    std::vector<std::shared_ptr<Image>> split();

    /**
     * Get matrix
     * @return matrix
     */
    std::shared_ptr<cv::Mat> getMat() const {return m_mat;}

    /**
     * Get id
     * @return unique image id
     */
    long getId() const { return m_id;}

    /**
     * Rotate matrix
     * @param angle
     */
    void rotate(int angle);
};