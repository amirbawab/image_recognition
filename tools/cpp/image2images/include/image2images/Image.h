#pragma once

#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

struct charMatch {
    cv::Point2i position;
    cv::Mat image;
};

class Image : public std::enable_shared_from_this<Image>{
private:
    static long m_uniq_id;
    std::string m_name;
    std::shared_ptr<cv::Mat> m_mat;
    int m_label;
    long m_id = m_uniq_id++;

    /**
     * Generate matrix permutatios
     * @param images
     * @param indices
     * @param charsContours
     */
    void _permutation(std::vector<std::shared_ptr<Image>> &images, std::vector<int> &indices,
                      std::vector<std::vector<cv::Point>> &charsContours);

    /**
     * Generate a new image
     * @parm indcies
     * @return generate image
     */
    std::shared_ptr<Image> _align(const std::vector<int> &indices, std::vector<std::vector<cv::Point>> &charsContours);

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
     * Deep copy of the matrix
     * @return matrix pointer
     */
    std::shared_ptr<cv::Mat> _cloneMat();

    // Detect elements
    std::vector<std::vector<cv::Point>> _charsControus();

    /**
     * Deep clone image
     * @return image pointer
     */
    std::shared_ptr<Image> _cloneImage();

    /**
     * Reduce colors
     * @param colors
     * @return Largest pixel value
     */
    int _reduceColors(int colors);

public:
    Image(int label, std::shared_ptr<cv::Mat> mat) : m_label(label), m_mat(mat){}

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
     * @return new image
     */
    std::shared_ptr<Image> cleanNoise();

    /**
     * Convert to binary
     * @param threshold
     */
    std::shared_ptr<Image> binarize();

    /**
     * Draw contour around objects
     */
    std::shared_ptr<Image> drawContour();

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
     * Get label
     * @return label
     */
    int getLabel() const {return m_label;}

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
     * Resize image
     * @param side
     * @return new image
     */
    std::shared_ptr<Image> size(int side);

    /**
     * Erode image
     * @param size
     * @return new image
     */
    std::shared_ptr<Image> erode(int size);

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

    /**
     * Get matrix side
     * @return side
     */
    int getSide() const {return m_mat->rows;}
};