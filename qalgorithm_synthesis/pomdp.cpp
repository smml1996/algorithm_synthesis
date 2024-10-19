//
// Created by Stefanie Muroya Lei on 28.01.24.
//
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <string>
#include "utils.cpp"
#include <fstream>
#include <cassert>

using namespace  std;

static auto HALT_ACTION = "halt";
static auto HALT = new Algorithm(HALT_ACTION, nullptr, nullptr, nullptr, 0);

class POMDP {
public:
    //            from                      I         to    prob
    unordered_map< int , unordered_map< string, map< int, MyFloat > > > probabilities;
    int initial_state{};
    unordered_set<string> actions{};
    unordered_map<int, int> gamma{};
    unordered_map<int, MyFloat> rewards{}; // maps a vertex to its reward

    void insert_probability(const int &from,const string &action, const int &to, const MyFloat &prob) {
        this->probabilities[from][action][to] = prob;
    }

    void safe_insert(const int &from, const string &action, const int &to, const MyFloat &prob) {
        if ((this->probabilities).find(from) != this->probabilities.end()){
            if (this->probabilities[from].find(action) != this->probabilities[from].end()) {
                assert((this->probabilities)[from][action].find(to) == (this->probabilities)[from][action].end());
            } else{
                this->probabilities[from][action] = map<int, MyFloat>();
            }
        } else{
            this->probabilities[from] = unordered_map<string, map<int, MyFloat>>();
        }
        return this->insert_probability(from, action, to, prob);
    }

    void safe_insert_action(const string &action){
        if (this->actions.find(action) == this->actions.end()) {
            this->actions.insert(action);
        } else {
            throw std::runtime_error("action " + action + " is being inserted more than once");
        }
    }

    void insert_gamma(const int &v, const int &obs) {
        assert(this->gamma.find(v) == this->gamma.end());
        this->gamma[v] = obs;
    }

    MyFloat get_vertex_reward(const int &id) {
        return this->rewards[id];
    }

    bool insert_reward(const int &id, const MyFloat&reward ) {
        if (this->rewards.find(id) == this->rewards.end()) {
            this->rewards.insert({id, reward});
            return true;
        } else {
            return false;
        }
    }
};

void fill_pomdp_gamma(POMDP &pomdp, const string &line){
    vector<string> out1;
    split_str(line, ' ', out1);

    assert(out1.size() == 2);
    assert(out1[0] == "GAMMA:");

    vector<string> out2;
    split_str(out1[1], ',', out2);

    for(const auto & i : out2){
        vector<string> out3;
        split_str(i, ':', out3);
        assert(out3.size() == 2);
        int v = stoi(out3[0]);
        int obs = stoi(out3[1]);
        pomdp.insert_gamma(v, obs);
    }
}

void fill_rewards(POMDP &pomdp, const string &line) {
    vector<string> elements;
    split_str(line, ' ', elements);

    assert(elements.size() ==  2);
    assert(elements[0] == "REWARDS:");

    vector<string> vertices_to_rewards;
    split_str(elements[1], ',', vertices_to_rewards);

    for (const auto & vertex_to_reward : vertices_to_rewards) {
        vector<string> elements;
        split_str(vertex_to_reward, ':', elements);
        assert(elements.size() == 2);
        int v = stoi(elements[0]);
        MyFloat r(elements[1]);
        assert(pomdp.insert_reward(v, r));
    }

}

POMDP parse_pomdp_file (const string& fname) {
    POMDP pomdp;

    ifstream f(fname);
    string line;
    if (getline (f, line)) {
        assert (line == "BEGINPOMDP");
    } else {
        throw std::runtime_error("Error reading POMDP file: "+ fname);
    }


    // INITIALSTATE:
    string str_initial_state;
    getline(f, str_initial_state);
    vector<string> temp_elements;
    split_str(str_initial_state, ' ', temp_elements);
    assert(temp_elements[0] == "INITIALSTATE:");
    int initial_state = stoi(temp_elements[1]);
    pomdp.initial_state = initial_state;

    // STATES:
    getline(f, line); // this is the line of all the states of the POMDP

    // target vertices
    getline(f, line);
    fill_rewards(pomdp, line);

    // GAMMA:
    getline(f, line);
    fill_pomdp_gamma(pomdp, line);

    // Actions
    getline(f, line);
    assert(line == "BEGINACTIONS");
    getline(f, line);
    while(line != "ENDACTIONS"){
        vector<string> elements;
        split_str(line, ' ', elements);
        assert(elements.size() == 1);
        string action = elements[0];
        pomdp.safe_insert_action(action);
        getline(f, line);
    }

    getline(f,line);

    while(line != "ENDPOMDP") {
        vector<string> elements;
        split_str(line, ' ', elements);
        assert(elements.size() == 4);
        int fromv = stoi(elements[0]);
        string channel = elements[1];
        int tov = stoi(elements[2]);
        MyFloat prob(elements[3]);
//        cout << fromv << " " << channel << " " << tov << " " << elements[3] << endl;
        pomdp.safe_insert(fromv, channel, tov, prob);
        getline(f,line);
    }
    return pomdp;
}


// TODO: change POMDP to const

pair<Algorithm*, MyFloat> get_bellman_value(POMDP &pomdp, Belief &current_belief, const int &horizon, const string &opt_technique) {
    MyFloat curr_belief_val = current_belief.get_belief_reward(pomdp.rewards);


    if (horizon == 0) {
        return make_pair(HALT, curr_belief_val);
    }

    vector< pair< Algorithm*, MyFloat > > bellman_values;

    bellman_values.emplace_back(HALT, curr_belief_val);

    for(auto it = pomdp.actions.begin(); it != pomdp.actions.end(); it++) {
        string action = *it;

        // build next_beliefs, separate them by different observables
        map<int, Belief> obs_to_next_beliefs;

        MyFloat zero;
        for(auto & prob : current_belief.probs) {
            int current_v = prob.first;
            if(prob.second > zero) {
                for (auto &it_next_v: pomdp.probabilities[current_v][action]) {
                    if (it_next_v.second > zero) {
                        auto successor = it_next_v.first;
                        obs_to_next_beliefs[pomdp.gamma[it_next_v.first]].add_val(successor,
                                                                                  prob.second * it_next_v.second);
                    }
                }
            }
        }

        assert(obs_to_next_beliefs.size() < 3);
        
        if (!obs_to_next_beliefs.empty()) {
            vector<Algorithm *>next_algorithms;
            MyFloat bellman_val;

            for(auto & obs_to_next_belief : obs_to_next_beliefs) {
                auto temp = get_bellman_value(pomdp, obs_to_next_belief.second, horizon-1, opt_technique);
                next_algorithms.push_back(temp.first);
                bellman_val = bellman_val + temp.second;
            }

            assert(!next_algorithms.empty());
            assert(next_algorithms.size() < 3);

            Algorithm *new_alg_node = new Algorithm(*it, nullptr, nullptr, nullptr);
            if (next_algorithms.size() == 1) {
                new_alg_node->next_ins = next_algorithms[0];
                new_alg_node->depth = next_algorithms[0]->depth + 1;
            } else {
                // since maps are ordered by values:
                new_alg_node->case0 = next_algorithms[0]; // this should be a measure to 0
                new_alg_node->case1 = next_algorithms[1]; // This should be a measure to 1
                new_alg_node->depth = max(next_algorithms[0]->depth, next_algorithms[1]->depth) +1;
            }

            bellman_values.emplace_back(new_alg_node, bellman_val);
        }
    }

    MyFloat max_val; // this is initialized as zero
    for(auto & bellman_value : bellman_values) {
        if (opt_technique == "max") {
            max_val = max(max_val, bellman_value.second);
        } else {
            assert(opt_technique == "min");
            max_val = min(max_val, bellman_value.second);
        }
        
    }

    int shortest_alg_with_max_val = -1;
    for(auto & bellman_value : bellman_values) {
        if (bellman_value.second == max_val) {
            if (shortest_alg_with_max_val == -1) {
                shortest_alg_with_max_val = bellman_value.first->depth;
            } else {
                shortest_alg_with_max_val = min(shortest_alg_with_max_val, bellman_value.first->depth);
            }
        }
    }

    for(auto & bellman_value : bellman_values) {
        if (bellman_value.second == max_val and bellman_value.first->depth == shortest_alg_with_max_val) {
            return bellman_value;
        }
    }
    assert(false);
}

Belief get_initial_belief(POMDP &pomdp) {
    Belief initial_belief;

    if (pomdp.probabilities[pomdp.initial_state].find("INIT_") !=  pomdp.probabilities[pomdp.initial_state].end()) {
        for(auto it : pomdp.probabilities[pomdp.initial_state]["INIT_"]) {
            initial_belief.add_val(it.first, it.second);
        }
    } else {
        initial_belief.set_val(pomdp.initial_state, MyFloat("1"));
    }
    return initial_belief;
}

MyFloat get_algorithm_acc(POMDP &pomdp, Algorithm*& algorithm, Belief &current_belief) {
    MyFloat curr_belief_val = current_belief.get_belief_reward(pomdp.rewards);
    

    string action = algorithm->action;
    if ((action == HALT_ACTION) or (algorithm == nullptr)) {
        return curr_belief_val;
    }

    // build next_beliefs, separate them by different observables
    map<int, Belief> obs_to_next_beliefs;

    MyFloat zero;
    for(auto & prob : current_belief.probs) {
        int current_v = prob.first;
        if(prob.second > zero) {
            for (auto &it_next_v: pomdp.probabilities[current_v][action]) {
                if (it_next_v.second > zero) {
                    obs_to_next_beliefs[pomdp.gamma[it_next_v.first]].add_val(it_next_v.first,
                                                                              prob.second * it_next_v.second);
                }
            }
        }
    }

    assert(obs_to_next_beliefs.size() < 3);

    if (!obs_to_next_beliefs.empty()) {
        MyFloat bellman_val;
        for(auto & obs_to_next_belief : obs_to_next_beliefs) {
            MyFloat temp;
            if (algorithm->next_ins != nullptr) {
                assert(algorithm->case0 == nullptr);
                assert(algorithm->case1 == nullptr);
                temp = get_algorithm_acc(pomdp, algorithm->next_ins, obs_to_next_belief.second);
            }else{
                if(obs_to_next_belief.first == 0) {
                    temp = get_algorithm_acc(pomdp, algorithm->case0, obs_to_next_belief.second);
                } else {
                    assert(obs_to_next_belief.first == 1);
                    temp =  get_algorithm_acc(pomdp, algorithm->case1, obs_to_next_belief.second);
                }
            }

            bellman_val = bellman_val + temp;
        }

        return bellman_val;
    } else {
        return curr_belief_val;
    }
}

