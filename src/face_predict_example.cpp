/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Xiao Liu, pertusa, caprice-j
 * \file image_classification-predict.cpp
 * \brief C++ predict example of mxnet
 *
 * This is a simple predictor which shows how to use c api for image classification. It uses
 * opencv for image reading.
 *
 * Created by liuxiao on 12/9/15.
 * Thanks to : pertusa, caprice-j, sofiawu, tqchen, piiswrong
 * Home Page: www.liuxiao.org
 * E-mail: liuxiao@foxmail.com
*/

#include <vector>
#include <opencv2/opencv.hpp>
// Path for c_predict_api
#include "mxnet/c_predict_api.h"

#include "face_predict.h"

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "No test image here." << std::endl
              << "Usage: ./image-classification-predict apple.jpg" << std::endl;
    return EXIT_FAILURE;
  }

  std::string image_path(argv[1]);

  // Models path for your model, you have to modify it
  std::string json_file = "/Users/moon/abc/AI/Models/model-r50-am-lfw/model-symbol.json";
  std::string param_file = "/Users/moon/abc/AI/Models/model-r50-am-lfw/model-0000.params";

  // Image size and channels
  int width = 112;
  int height = 112;
  int channels = 3;
  InputShape input_shape (width, height, channels);
  
  // Load model
  PredictorHandle pred_hnd = nullptr;
  LoadMXNetModel(&pred_hnd, json_file, param_file, input_shape);

  // Read Image Data
  auto image_size = static_cast<std::size_t>(width * height * channels);
  std::vector<mx_float> image_data(image_size);
  GetImageFile(image_path, image_data.data(), channels, cv::Size(width, height));

  // Inference
  std::vector<float> data;
  Infer(pred_hnd, image_data, data);

  // normalize the output vector
  std::vector<float> output(data.size());
  cv::normalize(data, output);

  // Print Output Data
  PrintOutputResult(output);

  // Release Predictor
  MXPredFree(pred_hnd);

  return EXIT_SUCCESS;
}
