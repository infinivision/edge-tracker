#include <clickhouse/client.h>

#include <iostream>
#include <stdexcept>

using namespace clickhouse;

/// Initialize client connection.

int main(){

    try{
        Client client(ClientOptions().SetHost("172.19.0.101"));

        std::cout << "init client for click house server: 172.19.0.105\n" ;

        Block block;

        auto sample_date = std::make_shared<ColumnDate>();
        sample_date->Append(123);

        auto time_stamp = std::make_shared<ColumnUInt64>();
        time_stamp->Append(456);

        block.AppendColumn("sample_date" , sample_date);
        block.AppendColumn("time_stamp", time_stamp);

        client.Insert("tracker.sample123", block);

        std::cout << "insert one row into table tracker.sample\n";
    } 
    catch (clickhouse::ServerException & exception) {
        std::cout << "click house access exception " << exception.what() << std::endl;
    }
    catch(std::system_error & exception){
        std::cout << "click house access exception, system error: " << exception.what() << std::endl;
    }
}
