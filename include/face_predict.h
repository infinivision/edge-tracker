#ifndef _FACE_PREDICT_H_
#define _FACE_PREDICT_H_

#include <vector>
#include "cpptoml.h"

// Path for c_predict_api
#include "mxnet/c_predict_api.h"
#include "camera.h"
//#include "vectodb.hpp"

extern int m_channel;
extern int m_width;
extern int m_height;
extern int output_feature;
extern PredictorHandle embd_hd;
extern PredictorHandle age_hd;
extern int n_age_sample;
extern bool age_enable;
// Read file to buffer
class BufferFile {
 public :
  std::string file_path_;
  std::size_t length_ = 0;
  std::unique_ptr<char[]> buffer_;

  BufferFile(const std::string& file_path);

  inline std::size_t GetLength() {
    return length_;
  }

  inline char* GetBuffer() {
    return buffer_.get();
  }
};

class InputShape {
public:
  mx_uint input_shape_indptr[2] = {0,4};
  mx_uint input_shape_data[4];

  InputShape(int width, int height, int channels);
};

void GetImageFile(const std::string& image_file,
                  mx_float* image_data, int channels,
                  cv::Size resize_size);

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
                      int dev_type = 1,          /* 1: cpu, 2:gpu */
                      int dev_id = 0             /* 0: arbitary */
                    );

void Infer ( PredictorHandle pred_hnd,           /* mxnet model */
	         std::vector<mx_float> &image_data,  /* input data */
	         std::vector<float> &data);          /* output vector */

void PrintOutputResult(const std::vector<float>& output);

void imgFormConvert( const cv::Mat input, std::vector<mx_float> & img_vec);

void LoadMxModelConf(std::string mx_model_conf);

int proc_embd_vec(std::vector<float> &data, const CameraConfig & camera,int frameCount,int faceId);

#endif // _FACE_PREDICT_H_