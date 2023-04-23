//
// Intro example caffe framework with opencv >3.3
// Author: NRR
// how to use opencv_dnn module for image classification
// by using GoogLeNet trained network from Caffe model zoo.

// Based on https://github.com/opencv/opencv/blob/master/samples/dnn/classification.cpp

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "common.hpp"

std::string keys =
    "{ help  h         |   | Print help message. }"
    "{ N     N         | 1 | number of most likely classes classified to show }"
    "{ @image          |   | Path to input image or video file. }";

using namespace cv;
using namespace dnn;

std::vector<std::string> classes;

int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, keys);

    parser = CommandLineParser(argc, argv, keys);
    parser.about("Use this script to run classification deep learning networks using OpenCV.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    int N = parser.get<int>("N");
    int rszWidth = 0;
    int rszHeight = 0;
    float scale = 1.0;
    Scalar mean(104, 117, 123);
    Scalar std;
    bool swapRB = true;
    bool crop = 0;
    int inpWidth = 224;
    int inpHeight = 224;
    String model = findFile("bvlc_googlenet.caffemodel");
    String config = findFile("bvlc_googlenet.prototxt");
    String framework;
    int backendId = 0;
    int targetId = 0;

    // Open file with classes names.
    std::string file = "classification_classes_ILSVRC2012.txt";
    std::ifstream ifs(file.c_str());
    if (!ifs.is_open())
        CV_Error(Error::StsError, "File " + file + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    CV_Assert(!model.empty());

    //! [Read and initialize network]
    Net net = readNet(model, config, framework);
    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);

    //! [Open a video file or an image file or a camera stream]
    VideoCapture cap;
    if (parser.has("@image"))
        cap.open(parser.get<String>("@image"));
    else
        cap.open(0);

    // Process frames.
    Mat frame, blob;
    cap >> frame;
    if (frame.empty())
    {
        std::cerr << "Cannot read the image." << std::endl;
        return 1;
    }

    if (rszWidth != 0 && rszHeight != 0)
    {
        resize(frame, frame, Size(rszWidth, rszHeight));
    }

    //! [Create a 4D blob from a frame]
    blobFromImage(frame, blob, scale, Size(inpWidth, inpHeight), mean, swapRB, crop);

    // Check std values.
    if (std.val[0] != 0.0 && std.val[1] != 0.0 && std.val[2] != 0.0)
    {
        // Divide blob by std.
        divide(blob, std, blob);
    }
 
    //! [Set input blob]
    net.setInput(blob);

    Mat prob = net.forward();
    double t1;
    prob = net.forward();

    Mat probSortedIndex(prob.size(), CV_32F);
    sortIdx(prob, probSortedIndex, SORT_EVERY_ROW + SORT_DESCENDING);

    // Print first N predicted classes.
    std::cout << "Showing the " << N << " most likely classes: " << std::endl << std::endl;
    int classId;
    float confidence;
    std::string label;
    for (size_t i = 0; i < N; i++)
    {
        classId = probSortedIndex.col(i).at<int>(0);
        confidence = prob.col(probSortedIndex.col(i).at<int>(0)).at<float>(0);
        label = format("%s: %.6f", (classes.empty() ? format("Class #%d", classId).c_str() : classes[classId].c_str()),
                       confidence);
        std::cout << label << std::endl;
    }

    return 0;
}
