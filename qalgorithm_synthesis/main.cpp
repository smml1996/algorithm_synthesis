#include <iostream>
#include <ctime>
#include "pomdp.cpp"
#include "json.hpp"
#include <type_traits>
#include <cassert>
#include <filesystem>  // C++17 and above

// for convenience
using json = nlohmann::json;

using namespace  std;

auto keys = {"name", "embeddings_path", "min_horizon", "max_horizon", "output_dir", "pomdps_path"};
/// @brief 
/// @param argc 
/// @param argv 
/// @return 
int main(int argc, char **argv) {
    string arg1 = argv[1];
    if ( arg1.compare("bellmaneq") == 0) {
        cout << "opening config file: " << argv[2] << endl;

        std::ifstream f(argv[2]); // parse configuration file
        json config_json = json::parse(f);
        f.close();
        
        for (auto key : keys) { // check that config file has all we need
            if (!config_json.contains(key)) {
                string key_ = key;
                throw std::invalid_argument(key_ + "not in config file");
            }
        }


        // print config file
        for (auto& el : config_json.items()) {
            std::cout << el.key() << " : " << el.value() << "\n\n";
        }

        int min_horizon = config_json["min_horizon"];
        int max_horizon = config_json["max_horizon"];
        string embeddings_path = config_json["embeddings_path"];
        string output_dir = config_json["output_dir"];
        string pomdps_path = config_json["pomdps_path"];

        // check embedding file exists
        if (!std::filesystem::exists(embeddings_path)) {
            throw std::runtime_error("Embedding files does not exist");
        } 

        // open embeddings file
        std::ifstream embeddings_file(embeddings_path);
        json all_embeddings = json::parse(embeddings_file);
        embeddings_file.close();

        // checking output dir exists (or create)
        if (!std::filesystem::exists(output_dir)) {
            std::cout << "Output dir does not exists. Creating directory..." << std::endl;
            if (std::filesystem::create_directory(output_dir)) {
                std::cout << "Directory created successfully." << std::endl;
            } else {
                std::cout << "Failed to create directory or it already exists.\n" << std::endl;
            }
        } else {
            cout << "output directory exists" << endl;
        }
        
        // create directory where algorithms should be stored (if it does not already exists)
        if (!std::filesystem::exists(output_dir+"algorithms/")) {
            std::cout << "algorithms dir does not exists. Creating directory..." << std::endl;
            if (std::filesystem::create_directory(output_dir+"algorithms/")) {
                std::cout << "algorithms directory created successfully." << std::endl;
            } else {
                std::cout << "Failed to create algorithms directory or it already exists.\n" << std::endl;
            }
        } else {
            cout << "algorithms directory exists" << endl;
        }

        // we output the computed lambdas in the following file:
        ofstream lambdas_file( output_dir + "lambdas.txt");
        lambdas_file << "embedding,horizon,lambda,time\n";
        lambdas_file.flush();

        for (auto& el : all_embeddings.items()) {
            if (el.key() == "count") continue;

            string hardware = el.key();
            int count = el.value()["count"];

            for (int embedding_index = 0; embedding_index < count; embedding_index ++) {
                auto pomdp = parse_pomdp_file(pomdps_path+hardware+"_"+ to_string(embedding_index)) ;
                Belief initial_belief = get_initial_belief(pomdp);
                for (int horizon = min_horizon; horizon < max_horizon+1; horizon++) {
                    cout << "Running experiment: " << hardware << embedding_index << " h="<< horizon << endl;
                    long time_before = time(nullptr);
                    auto result = get_bellman_value(pomdp, initial_belief, horizon);
                    long time_after = time(nullptr);
                    auto lambda = result.second;
                    lambdas_file << embedding_index << "," << horizon << "," << lambda << ","
                                << (time_after-time_before) << "\n";
                    lambdas_file.flush();
                    write_algorithm_file(result.first,
                                        output_dir+"algorithms/"+hardware+"_"+ to_string(embedding_index)+"_"+ to_string(horizon));
                }
            }
        }
        lambdas_file.close();
    } else if (arg1.compare("exact") == 0){

        // auto experiment_index = to_string(argv[2][0] - '0');
        // if (experiment_index.compare("0") == 0){
        //     experiment_index="";
        // }
        // string experiment_folder = "analysis_results" +  experiment_index + '/';
        // cout << experiment_folder << endl;
        // ofstream output_file(experiment_folder + "backends_vs.csv");
        // output_file << "horizon,diff_index,real_hardware,acc\n";

        // for (int horizon = 4; horizon < 8; horizon++) {

        //     // getting the algorithms for a given horizon and experiment index
        //     std::ifstream f(experiment_folder + "diff" + to_string(horizon )+ ".json");
        //     cout << experiment_folder + "diff" + to_string(horizon)+ ".json" << endl;
        //     json algorithms_data = json::parse(f);
        //     int count_algorithms = algorithms_data["count"];
        //     auto algorithms = algorithms_data["algorithms"];
        //     f.close();
        //     // we iterate over all algorithms to test them in all embeddings
        //     for(auto alg_index = 0; alg_index < count_algorithms; alg_index++) {
        //         Algorithm* algorithm = new Algorithm(algorithms[to_string(alg_index)]);
        //         for(auto it = backends_to_embeddings.begin(); it != backends_to_embeddings.end(); it++) {
        //             auto backend = it->first;
        //             for(auto embedding_index = 0; embedding_index < it->second; embedding_index++) {
        //                 auto pomdp = parse_pomdp_file(
        //                         "pomdps" + experiment_index + "/"+backend+"_"+ to_string(embedding_index));
        //                 Belief initial_belief = get_initial_belief(pomdp);
        //                 unordered_map<int, int> mapping = get_mapping(backend, embedding_index, experiment_index);
        //                 auto acc = get_algorithm_acc(pomdp, algorithm, initial_belief, mapping);
        //                 output_file << horizon << "," << alg_index << "," << backend << embedding_index << "," << acc << "\n";
        //             }
        //         }
        //         output_file.flush();
        //     }

        // }

        // output_file.close();
    } else {
        cout << "nothing matches" << endl;
    }

    return 0;
}
