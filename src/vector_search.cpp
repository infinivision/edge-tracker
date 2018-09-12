#include "vector_search.h"

#include <vector>
#include <map>
#include <numeric>
#include <algorithm>
#include <functional>
#include <mutex>

#include <glog/logging.h>
#include "cpptoml.h"

#include "utils.h"

typedef struct {
  std::vector<float> vec;
  int idx;
  float score;
} vec_element;

std::mutex vec_mutex;
// VectoDB * dbp;
std::map<long, std::vector<vec_element>> identifies;
int merge_count=0;
int append_count=0;
int full_count=0;
int none_count=0;
float sim_threshold = 0.4;
float merge_threshold = 0.8;
float append_threshold = 0.55;
int max_vector_size = 6;
int amt_reid = 2000;

void LoadVecSearchConf(std::string mx_model_conf) {

  try {
      auto g = cpptoml::parse_file(mx_model_conf);
      sim_threshold = g->get_qualified_as<double>("vector.sim_threshold").value_or(0.4);
      merge_threshold = g->get_qualified_as<double>("vector.merge_threshold").value_or(0.8); 
      append_threshold = g->get_qualified_as<double>("vector.append_threshold").value_or(0.55); 
      max_vector_size = g->get_qualified_as<int>("vector.max_vector_size").value_or(6);
      amt_reid = g->get_qualified_as<int>("vector.amt_reid").value_or(2000);
      std::cout << "load vector search conf!\n" ;
  }
  catch (const cpptoml::parse_exception& e) {
      std::cerr << "Failed to parse vector search conf in mxModel.toml: " << e.what() << std::endl;
      exit(1);
  }
  
}

#ifdef BENCH_EDGE
long  sum_t_search = 0;
long  search_count = 0;
#endif

void vec_norm(std::vector<float> &in, std::vector<float> &out){
  float sqare_sum=0;
  for(size_t i=0;i<in.size();i++){
    sqare_sum += in[i]*in[i];
  }
  float magnititue = sqrt(sqare_sum);
  out.resize(in.size());
  for(size_t i=0;i<out.size();i++){
    out[i] = in[i] / magnititue;
  }
}


void update_sim_score(std::vector<vec_element> & vec_list){
  for(size_t i=0;i<vec_list.size();i++){
    int    max_ids  = 0;
    float  max_sims = -1.0;
    for(size_t j=0;j<vec_list.size();j++){
      if(i==j) continue;
      float sim = std::inner_product(vec_list[i].vec.begin(), vec_list[i].vec.end(), vec_list[j].vec.begin(), 0.0);
      if(sim>max_sims){
        max_sims = sim;
        max_ids = j;
      }
    }
    vec_list[i].idx   = max_ids;
    vec_list[i].score = max_sims;
  }
}

void insert_vec(std::vector<vec_element> & vec_list, std::vector<float> & i_vec, int i_id, float i_score){
  if(i_score>merge_threshold){
    std::transform(i_vec.begin(), i_vec.end(), 
                                  vec_list[i_id].vec.begin(), i_vec.begin(),std::plus<float>());
    std::vector<float> new_vec_norm;
    vec_norm(i_vec,new_vec_norm);
    vec_list[i_id].vec = new_vec_norm;
    update_sim_score(vec_list);
    merge_count++;
  } else if(i_score<append_threshold) {
    if(vec_list.size()<max_vector_size){
      vec_element element;
      element.vec = i_vec;
      vec_list.push_back(element);
      update_sim_score(vec_list);
      append_count++;
    } else {
      full_count++;
      float max_score = 0.0;
      int   max_id = 0;
      for(size_t i=0;i<vec_list.size();i++){
        if(max_score<vec_list[i].score){
          max_score = vec_list[i].score;
          max_id    = i;
        }
      }
      if(max_score>i_score){
        int vec2_id = vec_list[max_id].idx;
        std::transform(vec_list[vec2_id].vec.begin(), vec_list[vec2_id].vec.end(),
                                      vec_list[max_id].vec.begin(), vec_list[vec2_id].vec.begin(),std::plus<float>());
        std::vector<float> new_vec_norm;
        vec_norm(vec_list[vec2_id].vec,new_vec_norm);
        vec_list[max_id].vec  = new_vec_norm;
        vec_list[vec2_id].vec = i_vec;
        update_sim_score(vec_list);
      } else {
        std::transform(i_vec.begin(), i_vec.end(),
                       vec_list[i_id].vec.begin(), i_vec.begin(),std::plus<float>());
        std::vector<float> new_vec_norm;
        vec_norm(i_vec,new_vec_norm);
        vec_list[i_id].vec  = new_vec_norm;
        update_sim_score(vec_list);
      }
    }
  } else {
    none_count++;
  }
}

long reid = 0;

// Find the similar vector in the list, if no similar vector found, insert into the list head
// If list len is bigger than threshold, delete vecotors from the list tail
// output id,vector,coordinate,timestamp?


int proc_embd_vec(std::vector<float> &data, const CameraConfig & camera,int frameCount,int faceId) {
  int new_id = -1;
  std::vector<float> i_vec;
  vec_norm(data,i_vec);
  vec_mutex.lock();

#ifdef BENCH_EDGE
  struct timeval  tv;
  gettimeofday(&tv,NULL);
  long t_ms1 = tv.tv_sec * 1000 * 1000 + tv.tv_usec;
#endif

  if(identifies.size()>= amt_reid){
    identifies.clear();
    reid = 0;
    LOG(INFO) << "reid amount up to "<< amt_reid <<", clear map";
  }

  if(identifies.empty()) {
    vec_element element;
    element.vec = i_vec;
    element.idx = 0;
    element.score = 1;
    std::vector<vec_element> vec_list;
    vec_list.push_back(element);
    identifies[0] = vec_list;
    reid++;
    vec_mutex.unlock();
    LOG(INFO) << camera.ip << " frame["<< frameCount << "]faceId[" << faceId
              << "]add first face vec ";
    return 0;
  }

  std::map<long, std::vector<vec_element>>::iterator it;
  int   max_reid = -1;
  int   max_idx  = -1;
  float max_sim_score  = -1.0;
  for(it=identifies.begin(); it!= identifies.end(); it++){
    for(size_t j=0;j<it->second.size();j++){
      float sim = std::inner_product(i_vec.begin(), i_vec.end(), it->second[j].vec.begin(), 0.0);
      if(sim > max_sim_score){
        max_sim_score = sim;
        max_idx = j;
        max_reid = it->first;
      }
    }
  }

  if(max_sim_score<sim_threshold){
    vec_element element;
    element.vec = i_vec;
    element.idx = 0;
    element.score = 1;
    std::vector<vec_element> vec_list;
    vec_list.push_back(element);
    identifies[reid] = vec_list;
    LOG(INFO) << camera.ip << " frame["<< frameCount << "]faceId[" << faceId
              << "]add new face vec,distance[" << max_sim_score <<"], reid[" << reid <<"]";
    new_id = reid;
    reid++;
  } else {
    new_id = max_reid;
    insert_vec(identifies[max_reid], i_vec, max_idx, max_sim_score);
    LOG(INFO) << camera.ip << " frame["<< frameCount << "]faceId[" << faceId
              << "]find old face vect,distance[" << max_sim_score <<"], reid[" << max_reid <<"] vec_list idx["<< max_idx <<"]";
  }

#ifdef BENCH_EDGE
    gettimeofday(&tv,NULL);
    long t_ms2 = tv.tv_sec * 1000 * 1000 + tv.tv_usec;
    sum_t_search += t_ms2-t_ms1;
    search_count++;
    LOG(INFO) << "vector search performance: [" << (sum_t_search/1000.0 ) / search_count << "] mili second latency per time";
#endif

  vec_mutex.unlock();
  return new_id;
}
