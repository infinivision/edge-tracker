#include "face_age.h"
#include "face_predict.h"
#include "utils.h"

#include "cpptoml.h"

PredictorHandle age_hd = nullptr;
int  n_age_sample = 2;
bool age_enable = true;

#ifdef BENCH_EDGE
long  sum_t_infer_age  = 0;
long  infer_count_age  = 0;
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
      std::string json_file  = g->get_qualified_as<std::string>("age.json").value_or("");
      std::string param_file = g->get_qualified_as<std::string>("age.param").value_or("");
      n_age_sample = g->get_qualified_as<int>("age.n_age_sample").value_or(2);
      age_enable = g->get_qualified_as<bool>("age.enable").value_or(true);
      LoadMXNetModel(&age_hd, json_file, param_file, input_shape);
      std::cout << "age model has been loaded!\n";
  }
  catch (const cpptoml::parse_exception& e) {
      std::cerr << "Failed to parse mxModel.toml: " << e.what() << std::endl;
      exit(1);
  }

}

int proc_age(vector<mx_float> face_vec, face_tracker & target) {
    if(target.infer_age_count < n_age_sample) {
        int age=0;
        if(age_enable){
            std::vector<float> age_vec;
            #ifdef BENCH_EDGE
            struct timeval  tv_age;
            gettimeofday(&tv_age,NULL);
            long t_ms1_age = tv_age.tv_sec * 1000 * 1000 + tv_age.tv_usec;
            #endif
            Infer(age_hd,face_vec, age_vec);
            for(size_t j = 2; j<age_vec.size()-1; j+=2){
                if(age_vec[j]<age_vec[j+1])
                    age++;
            }
            LOG(INFO) << "target infer age: " << age;
            #ifdef BENCH_EDGE
            gettimeofday(&tv_age,NULL);
            long t_ms2_age = tv_age.tv_sec * 1000 * 1000 + tv_age.tv_usec;
            infer_count_age++;
            if(infer_count_age>1){
                sum_t_infer_age += t_ms2_age-t_ms1_age;
                LOG(INFO) << "face infer age performance: [" << (sum_t_infer_age/1000.0 ) / (infer_count_age-1) << "] mili second latency per time";
            }
            #endif
        } else 
            age = 20;
        target.infer_age_count++;
        target.age_sum += age;
    }
    return target.age_sum / target.infer_age_count;
}