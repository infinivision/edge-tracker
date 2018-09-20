#include "face_age.h"
#include "face_predict.h"
#include "vector_search.h"
#include "utils.h"

#include <glog/logging.h>
#include "cpptoml.h"

static PredictorHandle infer_hd = nullptr;

static std::string infer_mode = "local";
static std::string svc_url = "";

int  n_age_sample = 2;
bool age_enable = true;

#ifdef BENCH_EDGE
static long  sum_t_infer  = 0;
static long  infer_count  = 0;
#endif

void LoadAgeConf(std::string mx_model_conf) {
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
    infer_mode  = g->get_qualified_as<std::string>("age.infer_mode").value_or("local");
    if(infer_mode=="local"){
        auto g = cpptoml::parse_file(mx_model_conf);
        std::string json_file  = g->get_qualified_as<std::string>("age.json").value_or("");
        std::string param_file = g->get_qualified_as<std::string>("age.param").value_or("");
        n_age_sample = g->get_qualified_as<int>("age.n_age_sample").value_or(2);
        age_enable = g->get_qualified_as<bool>("age.enable").value_or(true);
        LoadMXNetModel(&infer_hd, json_file, param_file, input_shape);
        std::cout << "age model has been loaded!\n";
    } else if(infer_mode=="service") {
        svc_url = g->get_qualified_as<std::string>("age.url").value_or("");
        std::cout << "age service url: " << svc_url <<"\n";
        curl_global_init(CURL_GLOBAL_ALL);
    }
  }
  catch (const cpptoml::parse_exception& e) {
      std::cerr << "Failed to parse mxModel.toml: " << e.what() << std::endl;
      exit(1);
  }

}

int infer_svc_age(cv::Mat & face, std::string & remote_file);

int proc_age(cv::Mat & face, vector<mx_float> & face_vec, face_tracker & target) {
    if(target.infer_age_count < n_age_sample) {
        int age=0;
        if(age_enable){
            std::vector<float> age_vec;
            std::string remote_file = to_string(target.faceId) + ".jpg";
            #ifdef BENCH_EDGE
            struct timeval  tv_age;
            gettimeofday(&tv_age,NULL);
            long t_ms1_age = tv_age.tv_sec * 1000 * 1000 + tv_age.tv_usec;
            #endif
            if(infer_mode == "local"){
                Infer(infer_hd,face_vec, age_vec);
                for(size_t j = 2; j<age_vec.size()-1; j+=2){
                    if(age_vec[j]<age_vec[j+1])
                        age++;
                }
            } else if(infer_mode == "service") {
                age = infer_svc_age(face, remote_file);
            }
            if(age!=-1){
                LOG(INFO) << "target infer age: " << age;
                #ifdef BENCH_EDGE
                gettimeofday(&tv_age,NULL);
                long t_ms2_age = tv_age.tv_sec * 1000 * 1000 + tv_age.tv_usec;
                infer_count++;
                if(infer_count>1){
                    sum_t_infer += t_ms2_age-t_ms1_age;
                    LOG(INFO) << "face infer age performance: [" << (sum_t_infer/1000.0 ) / (infer_count-1) << "] mili second latency per time";
                }
                #endif
            } else{
                LOG(WARNING) << "face infer age svc failed!";
                return -1;
            }
        } else 
            age = 20;
        target.infer_age_count++;
        target.age_sum += age;
    }
    return target.age_sum / target.infer_age_count;
}

int infer_svc_age(cv::Mat & face, std::string & remote_file) {
    std::vector<uchar> buff;//buffer for coding
    std::vector<int> param(2);
    param[0] = cv::IMWRITE_JPEG_QUALITY;
    param[1] = 100;//default(95) 0-100
    cv::imencode(".jpg", face, buff);

    CURL *curl = NULL;
    curl_mime * form;
    curl_mimepart *field;
    
    curl = curl_easy_init();
    CURLcode res;
    if(curl) {
        std::string readBuffer;
        curl_mime * form = curl_mime_init(curl);
        curl_mimepart *field = curl_mime_addpart(form);
        curl_mime_name(field, "data");
        curl_mime_filename(field, remote_file.c_str());
        curl_mime_data(field, (char *) &buff[0], buff.size());
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 1L);
        curl_easy_setopt(curl, CURLOPT_URL, svc_url.c_str());
        curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);

        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        // to do: set a small threshold for out of time
        /* Perform the request, res will get the return code */ 
        res = curl_easy_perform(curl);
        /* Check for errors */ 
        if(res != CURLE_OK){
            LOG(WARNING) << "access age service failed, curl error code: " << res;
            curl_easy_cleanup(curl);
            curl_mime_free(form);
            return -1;
        }

        long http_response_code;
        curl_easy_getinfo( curl, CURLINFO_RESPONSE_CODE, &http_response_code);
        if(http_response_code!=200){
            LOG(WARNING) << "access age service failed, http result code: " << http_response_code;
            curl_easy_cleanup(curl);
            curl_mime_free(form);            
            return -1;
        }
        auto json_res = json::parse(readBuffer);
        int age    = json_res["prediction"]["age"];
        int gender = json_res["prediction"]["gender"];
        curl_easy_cleanup(curl);
        curl_mime_free(form);
        return age;

    } else {
        LOG(WARNING) << "access embeding service failed, curl init failed";
        curl_easy_cleanup(curl);
        curl_mime_free(form);        
        return -1;        
    }
}
