#include "face_embed.h"
#include "face_predict.h"
#include "vector_search.h"
#include "utils.h"

#include <curl/curl.h>
#include <stdio.h>

#include "cpptoml.h"

PredictorHandle embd_hd = nullptr;
std::string embed_infer_mode = "local";
std::string embed_svc_url = "";

#ifdef BENCH_EDGE
long  sum_t_infer_embed  = 0;
long  infer_count_embed  = 0;
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
    embed_infer_mode  = g->get_qualified_as<std::string>("embedding.infer_mode").value_or("local");
    if(embed_infer_mode=="local"){
        std::string json_file  = g->get_qualified_as<std::string>("embedding.json").value_or("");
        std::string param_file = g->get_qualified_as<std::string>("embedding.param").value_or("");
        LoadMXNetModel(&embd_hd, json_file, param_file, input_shape);
        std::cout << "embedding model has been loaded!\n";
    } else if (embed_infer_mode=="service") {
        embed_svc_url = g->get_qualified_as<std::string>("embedding.url").value_or("");
        std::cout << "embedding service url: " << embed_svc_url <<"\n";
        CURL *curl;
        CURLcode res;
        curl = curl_easy_init();
        if(curl) {
            /* First set the URL that is about to receive our POST. This URL can
            just as well be a https:// URL if that is what should receive the
            data. */ 
            curl_easy_setopt(curl, CURLOPT_URL, "http://postit.example.com/moo.cgi");
            /* Now specify the POST data */ 
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, "name=daniel&project=curl");
        
            /* Perform the request, res will get the return code */ 
            res = curl_easy_perform(curl);
            /* Check for errors */ 
            if(res != CURLE_OK)
            fprintf(stderr, "curl_easy_perform() failed: %s\n",
                    curl_easy_strerror(res));
        
            /* always cleanup */ 
            curl_easy_cleanup(curl);
        }
        curl_global_cleanup();        
    }
  }
  catch (const cpptoml::parse_exception& e) {
      std::cerr << "Failed to parse mxModel.toml: " << e.what() << std::endl;
      exit(1);
  }

}

int proc_embeding(std::vector<mx_float> face_vec, face_tracker & target, 
                 const CameraConfig & camera, int frameCounter, int thisFace) {

    std::vector<float> face_embed_vec;

    #ifdef BENCH_EDGE
    struct timeval  tv;
    gettimeofday(&tv,NULL);
    long t_ms1 = tv.tv_sec * 1000 * 1000 + tv.tv_usec;
    #endif

    Infer(embd_hd,face_vec,face_embed_vec);

    #ifdef BENCH_EDGE
    gettimeofday(&tv,NULL);
    long t_ms2 = tv.tv_sec * 1000 * 1000 + tv.tv_usec;
    infer_count_embed++;
    if(infer_count_embed>1){
        sum_t_infer_embed += t_ms2-t_ms1;
        LOG(INFO) << "face infer embeding performance: [" << (sum_t_infer_embed/1000.0 ) / (infer_count_embed-1) << "] mili second latency per time";
    }
    #endif

    int new_id = proc_embd_vec(face_embed_vec, camera, frameCounter, thisFace);
    target.reid = new_id;
    return new_id;
}


