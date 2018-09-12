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

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <iomanip>
#include <opencv2/opencv.hpp>

#include <glog/logging.h>
#include "face_predict.h"
#include "utils.h"

int m_channel=0;
int m_width=0;
int m_height=0;

BufferFile::BufferFile(const std::string& file_path):file_path_(file_path) {

    std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
    if (!ifs) {
      std::cerr << "Can't open the file. Please check " << file_path << ". \n";
      return;
    }

    ifs.seekg(0, std::ios::end);
    length_ = static_cast<std::size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    std::cout << file_path.c_str() << " ... " << length_ << " bytes\n";

    buffer_.reset(new char[length_]);
    ifs.read(buffer_.get(), length_);
    ifs.close();
}

InputShape::InputShape(int width, int height, int channels) {
    input_shape_data[0] = 1;
    input_shape_data[1] = static_cast<mx_uint>(channels); 
    input_shape_data[2] = static_cast<mx_uint>(height);
    input_shape_data[3] = static_cast<mx_uint>(width);
}

// input opencv image matrix
// output image vecotr for mxnet
// input mat will be resize if it does not have a shape equal to the mxnet model
void imgFormConvert( const cv::Mat input, std::vector<mx_float> & img_vec) {
  cv::Mat ori,im;
  if(!input.isContinuous()){
    input.copyTo(ori);
    if(!ori.isContinuous()){
      std::cerr << "not enough continous memory for opencv mat\n";
      exit(1);
    }
  } else {
    ori = input;
  }
  if(ori.channels()!=m_channel){
      std::cerr << "mat channel is " << ori.channels()  << " ,can not proccess!\n";
      exit(1);
  }

  if(ori.cols*ori.rows!= m_width*m_height) {
    cv::resize(ori,im,cv::Size(m_width,m_height));
  } else
    im = ori;

  int size = m_channel * m_width * m_height;
  img_vec.resize(size);
  mx_float* image_data  = img_vec.data();
  mx_float* ptr_image_r = image_data;
  mx_float* ptr_image_g = image_data + size / 3;
  mx_float* ptr_image_b = image_data + size / 3 * 2;

  for (int i = 0; i < im.rows; i++) {
    auto data = im.ptr<uchar>(i);

    for (int j = 0; j < im.cols; j++) {
        *ptr_image_b++ = static_cast<mx_float>(*data++);
        *ptr_image_g++ = static_cast<mx_float>(*data++);
        *ptr_image_r++ = static_cast<mx_float>(*data++);
    }
  }

}

void GetImageFile(const std::string& image_file,
                  mx_float* image_data, int channels,
                  cv::Size resize_size) {
  // Read all kinds of file into a BGR color 3 channels image
  // the shape(height, width, channels)
  cv::Mat im_ori = cv::imread(image_file, cv::IMREAD_COLOR);

  if (im_ori.empty()) {
    std::cerr << "Can't open the image. Please check " << image_file << ". \n";
    assert(false);
  }

  cv::Mat im;

  // after resize, shape(112, 112, 3)
  resize(im_ori, im, resize_size);

  int size = im.rows * im.cols * channels;

  mx_float* ptr_image_r = image_data;
  mx_float* ptr_image_g = image_data + size / 3;
  mx_float* ptr_image_b = image_data + size / 3 * 2;

  for (int i = 0; i < im.rows; i++) {
    auto data = im.ptr<uchar>(i);

    for (int j = 0; j < im.cols; j++) {
        *ptr_image_b++ = static_cast<mx_float>(*data++);
        *ptr_image_g++ = static_cast<mx_float>(*data++);
        *ptr_image_r++ = static_cast<mx_float>(*data++);
    }
  }
}

void Infer ( PredictorHandle pred_hnd,         /* mxnet model */
           std::vector<mx_float> &image_data,  /* input data */
           std::vector<float> &data) {         /* output vector */



  // Set Input Image
  MXPredSetInput(pred_hnd, "data", image_data.data(), static_cast<mx_uint>(image_data.size()));

  // Do Predict Forward
  MXPredForward(pred_hnd);

  mx_uint output_index = 0;
  mx_uint* shape = nullptr;
  mx_uint shape_len;

  // Get Output Result
  MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);

  // std::cout << "output shape_len: " << shape_len << std::endl;
  // for(mx_uint index=0; index<shape_len; index++) {
  //   std::cout << shape[index] << std::endl;
  // }

  std::size_t size = 1;
  for (mx_uint i = 0; i < shape_len; ++i) { size *= shape[i]; }

  data.resize(size);

  MXPredGetOutput(pred_hnd, output_index, &(data[0]), static_cast<mx_uint>(size));

}

void PrintOutputResult(const std::vector<float>& output) {
    std::cout<< "embedding size: " << output.size() <<"\n";
    for(int i=0; i < output.size(); ++i) {
    std::cout << output[i];
    if((i+1) % 16 == 0) {
      std::cout << std::endl;
    } else {
      std::cout << " ";
    }
  }
}

/*
 * Load mxnet model
 *
 * Inputs:
 * - json_file:  path to model-symbol.json
 * - param_file: path to model-0000.params
 * - shape: input shape to mxnet model (1, channels, height, width)
 * - dev_type: 1: cpu, 2:gpu
 * - dev_id: 0: arbitary
 *
 * Output:
 * - PredictorHandle
 */
void LoadMXNetModel ( PredictorHandle* pred_hnd, /* Output */
                      std::string json_file,     /* path to model-symbol.json */
                      std::string param_file,    /* path to model-0000.params */
                      InputShape shape,          /* input shape to mxnet model (1, channels, height, width) */
                      int dev_type,              /* 1: cpu, 2:gpu */
                      int dev_id ) {             /* 0: arbitary */

  BufferFile json_data(json_file);
  BufferFile param_data(param_file);

  if (json_data.GetLength() == 0 || param_data.GetLength() == 0) {
    std::cerr << "Cannot load mxnet model" << std::endl;
    std::cerr << "\tjson file: " << json_file << std::endl;
    std::cerr << "\tparams file: " << param_file << std::endl;
    exit(EXIT_FAILURE);
  }

  // Parameters
  // int dev_type = 1;  // 1: cpu, 2: gpu
  // int dev_id = 0;  // arbitrary.
  mx_uint num_input_nodes = 1;  // 1 for feedforward
  const char* input_key[1] = { "data" };
  const char** input_keys = input_key;

  // Create Predictor
  MXPredCreate(static_cast<const char*>(json_data.GetBuffer()),
               static_cast<const char*>(param_data.GetBuffer()),
               static_cast<int>(param_data.GetLength()),
               dev_type,
               dev_id,
               num_input_nodes,
               input_keys,
               shape.input_shape_indptr,
               shape.input_shape_data,
               pred_hnd);
  assert(pred_hnd);
}

void LoadMxModelConf(std::string mx_model_conf) {
  try {
      auto g = cpptoml::parse_file(mx_model_conf);
      m_width   = g->get_qualified_as<int>("shape.width").value_or(112);
      m_height  = g->get_qualified_as<int>("shape.height").value_or(112);
      m_channel = g->get_qualified_as<int>("shape.channel").value_or(3);
  }
  catch (const cpptoml::parse_exception& e) {
      std::cerr << "Failed to parse mxModel.toml: " << e.what() << std::endl;
      exit(1);
  }

  InputShape input_shape(m_width, m_height, m_channel);
  std::cout << "load mx model shape: " << m_width << ","
                                                 << m_height << ","
                                                 << m_channel << "\n";

}

