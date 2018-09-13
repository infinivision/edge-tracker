#include <string.h>
#include <curl/curl.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <vector>

size_t WriteCallback(char *contents, size_t size, size_t nmemb, void *userp)
{
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

int main(int argc, char * argv[]){

    if(argc!=4)
        std::cout << "input format error! Usage: test_infer_service [service type] [service url] [data input_file]\n";

    int service_type = atoi(argv[1]);
    char * service_url = argv[2];
    char * filePath = argv[3];

    std::ifstream input_file(filePath, std::ios::binary | std::ios::ate);
    std::streamsize file_size = input_file.tellg();
    input_file.seekg(0, std::ios::beg);

    std::vector<char> buffer(file_size);
    if (! input_file.read(buffer.data(), file_size)) {
        std::cout << "read file error!\n";
        exit(-1);
    }

    CURL *curl;
    CURLcode res;
    curl = curl_easy_init();
    if(curl) {
        /* First set the URL that is about to receive our POST. This URL can
        just as well be a https:// URL if that is what should receive the
        data. */ 
        std::string readBuffer;
       
        /* Now specify the POST data */ 
        curl_mime * form = curl_mime_init(curl);
        curl_mimepart *field = curl_mime_addpart(form);
        curl_mime_name(field, "data");
        // curl_mime_data(field, buffer.data(), buffer.size());
        curl_mime_filedata(field, filePath);
        
        curl_easy_setopt(curl, CURLOPT_URL, service_url );
        curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);

        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        /* Perform the request, res will get the return code */ 
        res = curl_easy_perform(curl);
        /* Check for errors */ 
        if(res != CURLE_OK)
        fprintf(stderr, "curl_easy_perform() failed: %s\n",
                curl_easy_strerror(res));

        long http_response_code;    
        curl_easy_getinfo( curl, CURLINFO_RESPONSE_CODE, &http_response_code);
        std::cout << "http result code: " << http_response_code << std::endl;

        std::cout << "result body: " << readBuffer ;

        /* always cleanup */ 
        curl_easy_cleanup(curl);
        curl_mime_free(form);
    }
    curl_global_cleanup(); 

}