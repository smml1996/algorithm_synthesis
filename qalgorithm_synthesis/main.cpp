#include <iostream>
#include <ctime>
#include "pomdp.cpp"

using namespace  std;
int main() {
    cout << "running EXP" << EXP_INDEX << endl; 
    string backend;
    cin >> backend;
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
    return 0;
}
