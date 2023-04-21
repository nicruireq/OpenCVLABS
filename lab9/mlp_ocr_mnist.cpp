// MLP code (partial) for students of the MSEEI
// Author: Juan Pedro Bandera Rubio
// part of this code is stolen from Eric Yuan (http://eric-yuan.me/), who stole part from http://compvisionlab.wordpress.com/

#include "opencv2/core/core.hpp"
#include <opencv2/core/mat.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/ml/ml.hpp>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>

using namespace cv;
using namespace std;
using namespace ml;

int ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

// returns the cv::Mat container with all images arranged as files
void read_Mnist(string filename, cv::Mat &vec)
{
    ifstream file(filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char *)&number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        unsigned int vec_number_images = vec.size().height;
        unsigned int vec_pixels_per_image = vec.size().width;
        if (number_of_images < vec_number_images)
        {
            // Setup a rectangle to define your region of interest
            Rect myROI(0, 0, vec_pixels_per_image - 1, number_of_images - 1);
            // Crop the full image to that image contained by the rectangle myROI
            // Note that this doesn't copy the data
            vec = vec(myROI);
        }
        if (number_of_images > vec_number_images)
        {
            cout << " WARNING: There are " << number_of_images << " samples in " << filename << " but only the first " << vec_number_images << " will be considered" << endl;
        }

        for (int i = 0; i < vec_number_images; ++i)
        {
            for (int r = 0; r < n_rows; ++r)
            {
                for (int c = 0; c < n_cols; ++c)
                {
                    // cout << " image n: " << i << " row: " << r << " column: " << c;
                    unsigned char temp = 0;
                    file.read((char *)&temp, sizeof(temp));
                    int width = vec.size().width;
                    int index = r * n_cols + c;
                    // cout << " temp: " << (float)temp << " index: " << index << " vec height: " << vec.size().height << " width: " << vec.size().width << endl;
                    vec.at<float>(i, index) = ((float)temp) / 255.0;
                }
            }
        }
    }
}

// returns the cv::Mat container with all labels arranged as files with one column per output neuron (1.0 for the correct output, 0.0 for the rest)
void read_Mnist_Label(string filename, cv::Mat &labels)
{
    ifstream file(filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char *)&number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);

        unsigned int labels_height = labels.size().height;
        unsigned int labels_width = labels.size().width;
        if (number_of_images < labels_height)
        {
            // Setup a rectangle to define your region of interest
            Rect myROI(0, 0, labels_width - 1, number_of_images - 1);
            // Crop the full image to that image contained by the rectangle myROI
            // Note that this doesn't copy the data
            labels = labels(myROI);
        }
        if (number_of_images > labels_height)
        {
            cout << " WARNING: There are " << number_of_images << " samples in " << filename << " but only the first " << labels_height << " will be considered" << endl;
        }

        for (int i = 0; i < labels_height; ++i)
        {
            unsigned char temp = 0;
            file.read((char *)&temp, sizeof(temp));
            for (int j = 0; j < labels_width; j++)
            {
                labels.at<float>(i, j) = 0.0;
            }
            labels.at<float>(i, (unsigned int)temp) = 1.0;
        }
    }
    file.close();
}

// returns number of errors and hits in predicted labels vs
// real labels, each row of matrix is a label of a pattern
class PredResult
{
private:
    vector<int> results;
    Mat &predicted, &labels;
    unsigned int hits_, errors_, total_;

public:
    PredResult(Mat &predicted, Mat &labels)
        : predicted(predicted), labels(labels), hits_(0),
          errors_(0), total_(labels.rows){};

    unsigned int hits() { return hits_; }
    unsigned int errors() { return errors_; }
    unsigned int total() { return total_; }

    // Another form to calculate hits and errors
    // using only std functions
    void compute()
    {
        int nhits = 0;
        vector<int> pos_prediction(total_), pos_labels(total_);
        vector<float> linePred, lineLabel;
        size_t indexPred, indexLabel;
        for (size_t i = 0; i < total_; i++)
        {
            linePred = predicted.row(i);
            indexPred = distance(linePred.begin(), max_element(linePred.begin(), linePred.end()));
            lineLabel = labels.row(i);
            indexLabel = distance(lineLabel.begin(), max_element(lineLabel.begin(), lineLabel.end()));

            if (indexPred == indexLabel)
                hits_++;
            if (indexPred != indexLabel)
                errors_++;
        }
    }

    void operator()()
    {
        // search pairs (max_val, index) for each
        // row in predictions and labels, then compares
        // to know if there is a hit or an error
        // Using OpenCV function: minMaxIdx 
        double *minVal = new double(), *maxVal = new double();
        int *minPos = new int(), *maxPos = new int();
        int prediction;
        int trueLabel;

        for (size_t i = 0; i < total_; i++)
        {
            minMaxIdx(predicted.row(i).t(), minVal, maxVal, minPos, maxPos);
            prediction = *maxPos;
            minMaxIdx(labels.row(i).t(), minVal, maxVal, minPos, maxPos);
            trueLabel = *maxPos;
            if (prediction == trueLabel)
                hits_++;
            if (prediction != trueLabel)
                errors_++;
        }

        if (minVal != nullptr)
            delete minVal;
        if (maxVal != nullptr)
            delete maxVal;
        if (minPos != nullptr)
            delete minPos;
        if (maxPos != nullptr)
            delete maxPos;
    }
};


int main()
{
    string filename_train_images = "train-images.idx3-ubyte";
    string filename_train_labels = "train-labels.idx1-ubyte";
    int number_of_train_images = 60000;

    string filename_images = "t10k-images.idx3-ubyte";
    string filename_image_labels = "t10k-labels.idx1-ubyte";
    int number_of_images = 10000;
    int image_size = 28 * 28;

    // TRAIN IMAGES

    // read MNIST label into Mat
    cv::Mat mat_train_labels;
    mat_train_labels.create(number_of_train_images, 10, CV_32F);
    read_Mnist_Label(filename_train_labels, mat_train_labels);
    cout << "N. train labels: " << mat_train_labels.size() << endl;

    // read MNIST image into Mat
    cv::Mat mat_train_images;
    mat_train_images.create(number_of_train_images, image_size, CV_32F);
    read_Mnist(filename_train_images, mat_train_images);
    cout << "N. train images: " << mat_train_images.size() << endl;

    // TEST IMAGES

    // read MNIST label into Mat
    cv::Mat mat_test_labels;
    mat_test_labels.create(number_of_images, 10, CV_32F);
    read_Mnist_Label(filename_image_labels, mat_test_labels);
    cout << "N. test labels: " << mat_test_labels.size() << endl;

    // read MNIST image into Mat
    cv::Mat mat_test_images;
    mat_test_images.create(number_of_images, image_size, CV_32F);
    read_Mnist(filename_images, mat_test_images);
    cout << "N. test images: " << mat_test_images.size() << endl;

    // Create an OpenCV MLP
    Ptr<ANN_MLP> mlp = ANN_MLP::create();

    // Set number of hidden layers, and number of neurons in each layer
    //  Note: the number of input neurons must match the number of pixels
    //        in each image, and the number of output neurons must match the number of classes
    // For this example we create 2 hidden layers with 512 neurons
    mlp->setLayerSizes(Mat({image_size, 512, 512, 10}));

    // Set sigmoid as activation function
    mlp->setActivationFunction(ANN_MLP::ActivationFunctions::SIGMOID_SYM, 2.5, 1.0);
    // Set training algorithm
    mlp->setTrainMethod(ANN_MLP::TrainingMethods::BACKPROP);
    // Set learning rate (tasa de aprendizaje)
    mlp->setBackpropWeightScale(0.0001);

    // Set termination criteria
    mlp->setTermCriteria(TermCriteria(TermCriteria::Type::COUNT + TermCriteria::Type::EPS, 10, 0.01));

    // Arrange training data and...
    // Train!
    Ptr<TrainData> trainingData = TrainData::create(
        mat_train_images,
        SampleTypes::ROW_SAMPLE,
        mat_train_labels);
    mlp->train(trainingData);

    // // Get predictions for training images
    Mat mat_predicted_train(mat_train_labels.size(), CV_32F);
    mlp->predict(mat_train_images, mat_predicted_train);

    // Compute results for training images
    float percentage_hits = 0.0;
    float percentage_errors = 0.0;
    PredResult train_set_res(mat_predicted_train, mat_train_labels);
    train_set_res();
    unsigned int hits = train_set_res.hits();
    unsigned int errors = train_set_res.errors();
    unsigned int total = train_set_res.total();

    percentage_hits = (float)hits / (float)total;
    percentage_errors = (float)errors / (float)total;
    cout << " RESULTS TRAINING: Number of samples: " << total << " hits: " << percentage_hits * 100 << " \%, errors: " << percentage_errors * 100 << "\%" << endl;

    // Get predictions for evaluation images
    cv::Mat mat_predicted_test(mat_test_labels.size(), CV_32F);
    mlp->predict(mat_test_images, mat_predicted_test);

    // Compute results for evaluation images
    percentage_hits = 0.0;
    percentage_errors = 0.0;
    PredResult test_set_res(mat_predicted_test, mat_test_labels);
    test_set_res();
    hits = test_set_res.hits();
    errors = test_set_res.errors();
    total = test_set_res.total();

    percentage_hits = (float)hits / (float)total;
    percentage_errors = (float)errors / (float)total;
    cout << " RESULTS EVALUATION: Number of samples: " << total << " hits: " << percentage_hits * 100 << " \%, errors: " << percentage_errors * 100 << "\%" << endl;

    cout << "Press any key to exit" << endl;
    waitKey();

    return 0;
}