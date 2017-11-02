#pragma once

#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define NO_VALUE 777

struct charMatch {
    cv::Point2i position;
    cv::Mat image;
};

class Image : public std::enable_shared_from_this<Image>{
private:
    static long m_uniq_id;

    std::string m_name;
    std::shared_ptr<cv::Mat> m_mat;
    std::vector<std::vector<cv::Point>> m_contours;
    int m_value;
    long m_id = m_uniq_id++;

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

    /**
     * Generate binary image
     * @param image
     * @param threshold
     * @return binary image
     */
    void _binarize(std::shared_ptr<Image> image, int threshold);

    /**
     * Get the average pixel value in matrix
     * @return average
     */
    double _averagePixelVal();

    /**
     * Apply MNIST manipulations
     */
    void _mnist();

    /**
     * Reduce image to K colors
     * @param K
     */
    void _reduceColors(int K);
public:
    Image(std::shared_ptr<cv::Mat> mat = nullptr, int value = NO_VALUE) : m_mat(mat), m_value(value){}

    /**
     * Set image name
     * @param name
     */
    void setName(std::string name) { m_name = name;}

    /**
     * Get name
     * @return image name
     */
    std::string getName() const { return m_name;}

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
    std::shared_ptr<Image> binarize(int threshold);

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
    std::shared_ptr<Image> rotate(int angle);

    /**
     * Method that would manipulate MNIST
     * dataset.
     * NOTE: Do not use this function
     * if the input is not the MNIST dataset
     * @return
     */
    std::vector<std::shared_ptr<Image>> mnist();

    /**
     * Deep clone image
     * @return image pointer
     */
    std::shared_ptr<Image> clone();

    /**
     * Scale image
     * @param val
     * @return new image
     */
    std::shared_ptr<Image> scale(double val);

    /**
     * Recognize images
     * @param kNN
     * @return recognize images
     */
    std::string recognize(cv::Ptr<cv::ml::KNearest> kNN);

    /**
     * Extract characters
     * @return vector of matched chars
     */
    std::vector<charMatch> extractChars();
};