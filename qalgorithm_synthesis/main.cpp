#include <iostream>
#include <ctime>
#include "pomdp.cpp"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

using namespace  std;
int main(int argc, char **argv) {
    string arg1 = argv[1];
    if ( arg1.compare("bellmaneq") == 0) {
        cout << "running EXP" << EXP_INDEX << endl;
        string backend = argv[2];
        backend = "fake_" + backend;
        cout << "backend: "<< backend << endl;
        ofstream lambdas_file(DIR_PREFIX + "lambdas"+ EXP_INDEX + "/" + backend + ".txt");
        lambdas_file << "embedding,horizon,lambda,time\n";
        lambdas_file.flush();
        for (int embedding_index = 0; embedding_index < backends_to_embeddings[backend]; embedding_index ++) {
            auto pomdp = parse_pomdp_file(
                    DIR_PREFIX + "pomdps"+ EXP_INDEX + "/"+backend+"_"+ to_string(embedding_index));
            Belief initial_belief = get_initial_belief(pomdp);
            for (int horizon = 4; horizon < 8; horizon++) {
                long time_before = time(nullptr);
                auto result = get_bellman_value(pomdp, initial_belief, horizon);
                long time_after = time(nullptr);
                auto lambda = result.second;
                lambdas_file << embedding_index << "," << horizon << "," << lambda << ","
                             << (time_after-time_before) << "\n";
                lambdas_file.flush();
                write_algorithm_file(result.first,
                                     DIR_PREFIX+"algorithms" + EXP_INDEX+ "/"+backend+"_"+ to_string(embedding_index)+"_"+ to_string(horizon),
                                     get_inverse_mapping(backend, embedding_index));
            }

        }
        lambdas_file.close();
    } else if (arg1.compare("exact") == 0){

        auto experiment_index = to_string(argv[2][0] - '0');
        if (experiment_index.compare("0") == 0){
            experiment_index="";
        }
        string experiment_folder = "analysis_results" +  experiment_index + '/';
        cout << experiment_folder << endl;
        ofstream output_file(experiment_folder + "backends_vs.csv");
        output_file << "horizon,diff_index,real_hardware,acc\n";

        for (int horizon = 4; horizon < 8; horizon++) {

            // getting the algorithms for a given horizon and experiment index
            std::ifstream f(experiment_folder + "diff" + to_string(horizon )+ ".json");
            cout << experiment_folder + "diff" + to_string(horizon)+ ".json" << endl;
            json algorithms_data = json::parse(f);
            int count_algorithms = algorithms_data["count"];
            auto algorithms = algorithms_data["algorithms"];
            f.close();
            // we iterate over all algorithms to test them in all embeddings
            for(auto alg_index = 0; alg_index < count_algorithms; alg_index++) {
                Algorithm* algorithm = new Algorithm(algorithms[to_string(alg_index)]);
                for(auto it = backends_to_embeddings.begin(); it != backends_to_embeddings.end(); it++) {
                    auto backend = it->first;
                    for(auto embedding_index = 0; embedding_index < it->second; embedding_index++) {
                        auto pomdp = parse_pomdp_file(
                                "pomdps" + experiment_index + "/"+backend+"_"+ to_string(embedding_index));
                        Belief initial_belief = get_initial_belief(pomdp);
                        unordered_map<int, int> mapping = get_mapping(backend, embedding_index, experiment_index);
                        auto acc = get_algorithm_acc(pomdp, algorithm, initial_belief, mapping);
                        output_file << horizon << "," << alg_index << "," << backend << embedding_index << "," << acc << "\n";
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
