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

auto all_keys_required = {"name", "min_horizon", "max_horizon", "output_dir"};

/// @brief 
/// @param argc 
/// @param argv 
/// @return 
int main(int argc, char **argv) {
    string arg1 = argv[1];

    cout << "opening config file: " << argv[2] << endl; 

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
        std::cout << el.key() << " : " << el.value() << "\n\n";
    }

    string experiment_name = config_json["name"];
    string experiment_id = config_json["experiment_id"];
    int min_horizon = config_json["min_horizon"];
    int max_horizon = config_json["max_horizon"];
    filesystem::path output_dir = config_json["output_dir"];
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
        std::cout << "Output dir does not exists. Creating directory..." << std::endl;
        if (std::filesystem::create_directory(output_dir)) {
            std::cout << "Directory created successfully." << std::endl;
        } else {
            std::cout << "Failed to create directory or it already exists.\n" << std::endl;
        }
    } else {
        cout << "output directory exists" << endl;
    }
    
    if ( arg1.compare("bellmaneq") == 0) {
  
        filesystem::path algorithms_path = output_dir / "algorithms";
        // create directory where algorithms should be stored (if it does not already exists)
        if (!std::filesystem::exists(algorithms_path)) {
            std::cout << "algorithms dir does not exists. Creating directory..." << std::endl;
            if (std::filesystem::create_directory(algorithms_path)) {
                std::cout << "algorithms directory created successfully." << std::endl;
            } else {
                std::cout << "Failed to create algorithms directory or it already exists.\n" << std::endl;
            }
        } else {
            cout << "algorithms directory exists" << endl;
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
                    cout << "Running experiment: " << hardware << embedding_index << " h="<< horizon << endl;
                    long time_before = time(nullptr);
                    auto result = get_bellman_value(pomdp, initial_belief, horizon);
                    long time_after = time(nullptr);
                    auto lambda = result.second;
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

        string algorithms_path = config_json["algorithms_file"];
        // check embedding file exists
        if (!std::filesystem::exists(algorithms_path)) {
            throw std::runtime_error("algorithms file does not exist");
        } 

        filesystem::path accuracies_file_path = output_dir / "exact_accuracies.csv";
        ofstream output_file(accuracies_file_path);
        output_file << "horizon,diff_index,real_hardware,acc\n";

        for (int horizon = min_horizon; horizon < max_horizon+1; horizon++) {
            // getting the algorithms for a given horizon and experiment index
            std::ifstream f(algorithms_path);
            json algorithms_data = json::parse(f);
            int count_algorithms = algorithms_data["count"];
            auto algorithms = algorithms_data["algorithms"];
            f.close();
            // we iterate over all algorithms to test them in all embeddings
            for(auto alg_index = 0; alg_index < count_algorithms; alg_index++) {
                Algorithm* algorithm = new Algorithm(algorithms[to_string(alg_index)]);
                for (auto& el : all_embeddings.items()) {
                    if (el.key() == "count") continue;
                    string hardware = el.key();
                    int count = el.value()["count"];
                    for (int embedding_index = 0; embedding_index < count; embedding_index ++) {
                        filesystem::path instance_pomdp_path = pomdps_path / (hardware+"_"+ to_string(embedding_index) + ".txt");
                        auto pomdp = parse_pomdp_file(instance_pomdp_path);
                        Belief initial_belief = get_initial_belief(pomdp);
                        auto acc = get_algorithm_acc(pomdp, algorithm, initial_belief);
                        output_file << horizon << "," << alg_index << "," << hardware << embedding_index << "," << acc << "\n";
                    }
                }
                output_file.flush();
            }

        }

        output_file.close();
    } else {
        cout << "nothing matches" << endl;
    }

    return 0;
}
