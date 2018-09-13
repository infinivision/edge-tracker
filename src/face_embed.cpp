#include "face_embed.h"
#include "face_predict.h"
#include "vector_search.h"
#include "utils.h"

#include <glog/logging.h>
#include "cpptoml.h"

static PredictorHandle infer_hd = nullptr;

static std::string infer_mode = "local";
static std::string svc_url = "";
static CURL *curl = NULL;
static curl_mime * form;
static curl_mimepart *field;

#ifdef BENCH_EDGE
static long  sum_t_infer  = 0;
static long  infer_count  = 0;
#endif

void LoadEmbedConf(std::string mx_model_conf) {
  int w,h,c;
  try {
      auto g = cpptoml::parse_file(mx_model_conf);
      w = g->get_qualified_as<int>("shape.width").value_or(112);
      h = g->get_qualified_as<int>("shape.height").value_or(112);
      c = g->get_qualified_as<int>("shape.channel").value_or(3);
  }
  catch (const cpptoml::parse_exception& e) {
      std::cerr << "Failed to parse mxModel.toml: " << e.what() << std::endl;
      exit(1);
  }

  InputShape input_shape(w, h, c);

  try {
    auto g = cpptoml::parse_file(mx_model_conf);
    infer_mode  = g->get_qualified_as<std::string>("embedding.infer_mode").value_or("local");
    if(infer_mode=="local"){
        std::string json_file  = g->get_qualified_as<std::string>("embedding.json").value_or("");
        std::string param_file = g->get_qualified_as<std::string>("embedding.param").value_or("");
        LoadMXNetModel(&infer_hd, json_file, param_file, input_shape);
        std::cout << "embedding model has been loaded!\n";
    } else if (infer_mode=="service") {
        svc_url = g->get_qualified_as<std::string>("embedding.url").value_or("");
        std::cout << "embedding service url: " << svc_url <<"\n";
        curl_global_init(CURL_GLOBAL_ALL);
    }
  }
  catch (const cpptoml::parse_exception& e) {
      std::cerr << "Failed to parse mxModel.toml: " << e.what() << std::endl;
      exit(1);
  }

}

bool infer_svc_embed(cv::Mat & face, std::vector<float> & embed_vec, std::string & remote_file) ;

int proc_embeding(cv::Mat & face, std::vector<mx_float> & face_vec, face_tracker & target, 
                 const CameraConfig & camera, int frameCounter, int thisFace) {

    std::vector<float> face_embed_vec;
    std::string remote_file = camera.ip + "_" + to_string(frameCounter) + "_" + to_string(thisFace) + ".jpg";
    #ifdef BENCH_EDGE
    struct timeval  tv;
    gettimeofday(&tv,NULL);
    long t_ms1 = tv.tv_sec * 1000 * 1000 + tv.tv_usec;
    #endif

    bool infer_success = true;

    if(infer_mode == "local")
        Infer(infer_hd,face_vec,face_embed_vec);
    else if (infer_mode == "service") {
        infer_success = infer_svc_embed(face, face_embed_vec, remote_file);
    }
    #ifdef BENCH_EDGE
    if(infer_success){
        gettimeofday(&tv,NULL);
        long t_ms2 = tv.tv_sec * 1000 * 1000 + tv.tv_usec;
        infer_count++;
        if(infer_count>1){
            sum_t_infer += t_ms2-t_ms1;
            LOG(INFO) << "face infer embeding performance: [" << (sum_t_infer/1000.0 ) / (infer_count-1) << "] mili second latency per time";
        }
    }
    #endif
    if(infer_success) {
        target.reid = proc_embd_vec(face_embed_vec, camera, frameCounter, thisFace);
        return target.reid;
    } else{
        LOG(WARNING) << "face infer embeding svc failed!";
        return -1;
    }
}

// to do use cv::mat format do infer
bool infer_svc_embed(cv::Mat & face, std::vector<float> & embed_vec, std::string & remote_file) {

    std::vector<uchar> buff;//buffer for coding
    std::vector<int> param(2);
    param[0] = cv::IMWRITE_JPEG_QUALITY;
    param[1] = 100;//default(95) 0-100
    cv::imencode(".jpg", face, buff);

    curl = curl_easy_init();
    CURLcode res;
    if(curl) {
        std::string readBuffer;
        curl_mime * form = curl_mime_init(curl);
        curl_mimepart *field = curl_mime_addpart(form);
        curl_mime_name(field, "data");
        curl_mime_filename(field, remote_file.c_str());
        curl_mime_data(field, (char *) &buff[0], buff.size());

        curl_easy_setopt(curl, CURLOPT_URL, svc_url.c_str());
        curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 1L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        // to do: set a small threshold for out of time
        /* Perform the request, res will get the return code */ 
        res = curl_easy_perform(curl);
        /* Check for errors */ 
        if(res != CURLE_OK){
            LOG(WARNING) << "access embeding service failed, curl error code: " << res;
            curl_easy_cleanup(curl);
            curl_mime_free(form);            
            return false;
        }

        long http_response_code;
        curl_easy_getinfo( curl, CURLINFO_RESPONSE_CODE, &http_response_code);
        if(http_response_code!=200){
            LOG(WARNING) << "access embeding service failed, http result code: " << http_response_code;
            curl_easy_cleanup(curl);
            curl_mime_free(form);            
            return false;
        }
        auto json_res = json::parse(readBuffer);
        auto feature_json_array = json_res["prediction"];
        
        embed_vec.resize(feature_json_array.size());
        for(size_t i = 0; i<embed_vec.size();i++ ){
            embed_vec[i] = feature_json_array[i];
        }

        curl_easy_cleanup(curl);
        curl_mime_free(form);
        return true;

    } else {
        LOG(WARNING) << "access embeding service failed, curl init failed";
        curl_easy_cleanup(curl);
        curl_mime_free(form);        
        return false;        
    }
}