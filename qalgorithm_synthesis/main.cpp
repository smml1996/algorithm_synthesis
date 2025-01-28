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

auto all_keys_required = {"name", "min_horizon", "max_horizon", "output_dir", "opt_technique", "verbose"};

/// @brief 
/// @param argc 
/// @param argv 
/// @return 
int main(int argc, char **argv) {
    string arg1 = argv[1];
    
    if ( arg1.compare("bellmaneq") == 0) {
        cerr << "opening config file: " << argv[2] << endl; 

        std::ifstream f(argv[2]); // parse configuration file
        json config_json = json::parse(f);
        f.close();
        
        for (auto key : all_keys_required) { // check that config file has all we need
            if (!config_json.contains(key)) {
                string key_ = key;
                throw std::invalid_argument(key_ + "not in config file");
            }
        }


        // print config file
        for (auto& el : config_json.items()) {
            std::cerr << el.key() << " : " << el.value() << "\n\n";
        }

        string experiment_name = config_json["name"];
        string experiment_id = config_json["experiment_id"];
        int min_horizon = config_json["min_horizon"];
        int max_horizon = config_json["max_horizon"];
        string opt_technique = config_json["opt_technique"];
        filesystem::path project_path = get_project_path();
        filesystem::path output_dir = project_path / config_json["output_dir"];
        filesystem::path embeddings_file_ = "embeddings.json";
        filesystem::path embeddings_path  = output_dir / embeddings_file_;
        
        filesystem::path pomdps_path = output_dir / "pomdps/";
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
            std::cerr << "Output dir does not exists. Creating directory..." << std::endl;
            if (std::filesystem::create_directory(output_dir)) {
                std::cerr << "Directory created successfully." << std::endl;
            } else {
                std::cerr << "Failed to create directory or it already exists.\n" << std::endl;
            }
        } else {
            cerr << "output directory exists" << endl;
        }
  
        filesystem::path algorithms_path = output_dir / "algorithms";
        // create directory where algorithms should be stored (if it does not already exists)
        if (!std::filesystem::exists(algorithms_path)) {
            std::cerr << "algorithms dir does not exists. Creating directory..." << std::endl;
            if (std::filesystem::create_directory(algorithms_path)) {
                std::cerr << "algorithms directory created successfully." << std::endl;
            } else {
                std::cerr << "Failed to create algorithms directory or it already exists.\n" << std::endl;
            }
        } else {
            cerr << "algorithms directory exists" << endl;
        }

        // we output the computed lambdas in the following file:
        filesystem::path lambdas_file_path = output_dir / "lambdas.csv";
        ofstream lambdas_file(lambdas_file_path);
        lambdas_file << "hardware,embedding,horizon,lambda,time\n";
        lambdas_file.flush();

        for (auto& el : all_embeddings.items()) {
            if (el.key() == "count") continue;

            string hardware = el.key();
            int count = el.value()["count"];
            for (int embedding_index = 0; embedding_index < count; embedding_index ++) {
                filesystem::path instance_pomdp_path = pomdps_path / (hardware+"_"+ to_string(embedding_index) + ".txt");
                auto pomdp = parse_pomdp_file(instance_pomdp_path);

                Belief initial_belief = get_initial_belief(pomdp);
                for (int horizon = min_horizon; horizon < max_horizon+1; horizon++) {
                    cerr << "Running experiment: " << hardware << embedding_index << " h="<< horizon << endl;
                    long time_before = time(nullptr);
                    auto result = get_bellman_value(pomdp, initial_belief, horizon, opt_technique);
                    long time_after = time(nullptr);
                    auto lambda = result.second;
                    cout << lambda << endl;
                    lambdas_file << hardware << "," << embedding_index << "," << horizon << "," << lambda << ","
                                << (time_after-time_before) << "\n";
                    lambdas_file.flush();
                    filesystem::path instance_algo_path = algorithms_path / (hardware+"_"+ to_string(embedding_index)+"_"+ to_string(horizon)+".json");
                    write_algorithm_file(result.first, instance_algo_path);
                }
            }
        }
        lambdas_file.close();
    } else if (arg1.compare("exact") == 0){
        string algorithm_path = argv[2]; 
        string pomdp_path = argv[3];
        // getting the algorithms for a given horizon and experiment index
        std::ifstream f(algorithm_path);
        json algorithms_data = json::parse(f);
        f.close();
        Algorithm* algorithm = new Algorithm(algorithms_data);
        auto pomdp = parse_pomdp_file(pomdp_path);
        Belief initial_belief = get_initial_belief(pomdp);
        auto acc = get_algorithm_acc(pomdp, algorithm, initial_belief);
        cout << acc << endl;

    } else {
        cerr << "nothing matches" << endl;
    }

    return 0;
}
