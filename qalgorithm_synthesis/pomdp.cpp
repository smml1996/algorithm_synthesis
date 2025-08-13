//
// Created by Stefanie Muroya Lei on 28.01.24.
//
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <iostream>
#include <string>
#include "utils.cpp"
#include <fstream>
#include <cassert>


using namespace  std;

static auto HALT_ACTION = "halt";

struct Condition {
    int state;
    string relation;
    MyFloat prob;

    Condition(int state, std::string relation, MyFloat prob)
        : state(state), relation(std::move(relation)), prob(prob) {}
};


class POMDP {
public:
    //            from                      I         to    prob
    unordered_map< int , unordered_map< string, map< int, MyFloat > > > probabilities;
    int initial_state{};
    unordered_set<string> actions{};
    map<int, string> map_actions{};
    unordered_map<int, int> gamma{};
    unordered_map<int, MyFloat> rewards{}; // maps a vertex to its reward
    vector<Condition> terminal_belief_conditions{};

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

    double satisfies_condition(Belief &current_belief) {
        for (auto condition : terminal_belief_conditions) {
            MyFloat state_value = current_belief.get(condition.state);
            if (condition.relation.compare(">=")) {
                if ((state_value > condition.prob) || (state_value == condition.prob)) {
                    return 1.00;
                }
            } else {
                assert(false); // other relations must be implemented
            }
        }
        return 0.00;
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
        pomdp.map_actions[pomdp.map_actions.size()] = action;
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

    // conditions for target belief
    assert(line == "BEGINCONDITIONS");
    getline(f,line);
    while(line != "ENDCONDITIONS") {
        vector<string> elements;
        split_str(line, ' ', elements);
        assert(elements.size() == 3);
        int state = stoi(elements[0]);
        string relation = elements[1];
        MyFloat prob(elements[2]);

        pomdp.terminal_belief_conditions.emplace_back(state, relation, prob);
        getline(f,line);
    }

    // target_vertices
    return pomdp;
}


// TODO: change POMDP to const

pair<Algorithm*, MyFloat> get_bellman_value(POMDP &pomdp, Belief &current_belief, const int &horizon, const string &opt_technique, const MyFloat &threshold) {
    MyFloat curr_belief_val = current_belief.get_belief_reward(pomdp.rewards, opt_technique, threshold);
    int current_classical_state = -1;
    for(auto & prob : current_belief.probs) {
        if (current_classical_state == -1) {
            current_classical_state = pomdp.gamma[prob.first];
        } else {
            assert(pomdp.gamma[prob.first] == current_classical_state);
        }
    }
    assert(current_classical_state >= 0);
    auto halt_algorithm = new Algorithm(HALT_ACTION, current_classical_state, 0);
    if (horizon == 0) {
        return make_pair(halt_algorithm, curr_belief_val);
    }

    vector< pair< Algorithm*, MyFloat > > bellman_values;

    bellman_values.emplace_back(halt_algorithm, curr_belief_val);
    
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
        
        if (!obs_to_next_beliefs.empty()) {
            Algorithm *new_alg_node = new Algorithm(*it, current_classical_state);
            MyFloat bellman_val;

            int max_depth = 0;
            for(auto & obs_to_next_belief : obs_to_next_beliefs) {
                auto temp = get_bellman_value(pomdp, obs_to_next_belief.second, horizon-1, opt_technique, threshold);
                new_alg_node->children.push_back(temp.first);
                max_depth = max(temp.first->depth, max_depth);
                bellman_val = bellman_val + temp.second;
            }

            new_alg_node->depth = max_depth + 1;
            bellman_values.emplace_back(new_alg_node, bellman_val);
        }
    }

    MyFloat max_val; // this is initialized as zero
    for(auto & bellman_value : bellman_values) {
        if (opt_technique == "max" or opt_technique == "target") {
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

vector<int> get_initial_states(POMDP &pomdp) {
    vector<int> answer;

    if (pomdp.probabilities[pomdp.initial_state].find("INIT_") !=  pomdp.probabilities[pomdp.initial_state].end()) {
        for(auto it : pomdp.probabilities[pomdp.initial_state]["INIT_"]) {
            answer.push_back(it.first);
        }
    } else {
        answer.push_back(pomdp.initial_state);
    }
    return answer;
}

MyFloat get_algorithm_acc(POMDP &pomdp, Algorithm*& algorithm, Belief &current_belief, const string &opt_technique, const MyFloat &threshold) {
    MyFloat curr_belief_val = current_belief.get_belief_reward(pomdp.rewards, opt_technique, threshold);

    if (algorithm == nullptr) {
        return curr_belief_val;
    }
    string action = algorithm->action;
    if (action == HALT_ACTION) {
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
                }else {
                    assert(it_next_v.second == zero);
                }
            }
        }
    }
    assert(algorithm->children.size() == obs_to_next_beliefs.size());

    if (!obs_to_next_beliefs.empty()) {
        MyFloat bellman_val;
        
        for (int i = 0; i < algorithm->children.size(); i++) {
            assert(obs_to_next_beliefs.find(algorithm->children[i]->classical_state) != obs_to_next_beliefs.end());
            bellman_val = bellman_val + get_algorithm_acc(pomdp, algorithm->children[i], obs_to_next_beliefs[algorithm->children[i]->classical_state], opt_technique, threshold);
        }
        return bellman_val;
    } else {
        return curr_belief_val;
    }
}


// maximin code

double does_strategy_satisfies_conditions(POMDP &pomdp, const Algorithm *algorithm, Belief current_belief) {
    double curr_belief_val = pomdp.satisfies_condition(current_belief);

    if (algorithm == nullptr) {
        return curr_belief_val;
    }
    string action = algorithm->action;
    if (action == HALT_ACTION) {
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
                }else {
                    assert(it_next_v.second == zero);
                }
            }
        }
    }

    assert(algorithm->children.size() == obs_to_next_beliefs.size());

    if (!obs_to_next_beliefs.empty()) {
        for (int i = 0; i < algorithm->children.size(); i++) {
            assert(obs_to_next_beliefs.find(algorithm->children[i]->classical_state) != obs_to_next_beliefs.end());
            if (does_strategy_satisfies_conditions(pomdp, algorithm->children[i], obs_to_next_beliefs[algorithm->children[i]->classical_state]) == 0) {
                return 0.00;
            }
        }
        return 1.00;
    } else {
        return curr_belief_val;
    }
}

void set_minimax_values(POMDP &pomdp, 
    Algorithm* algorithm, 
    const vector<int> &initial_states,
    unordered_map<int, unordered_map<int, double>> &minimax_matrix,
    unordered_map<int, Algorithm*> &mapping_index_algorithm) {

    int current_alg_index = minimax_matrix.size();
    mapping_index_algorithm[current_alg_index] = deep_copy_algorithm(algorithm);
    
    // the current algorithm index should not exist
    assert(minimax_matrix.find(current_alg_index) == minimax_matrix.end());
    minimax_matrix[current_alg_index] = unordered_map<int, double>();

    for (int index = 0; index < initial_states.size(); index++) {
        assert(minimax_matrix[current_alg_index].find(index) == minimax_matrix[current_alg_index].end());

        Belief initial_belief;
        initial_belief.set_val(initial_states[index], MyFloat("1"));

        minimax_matrix[current_alg_index][index] = does_strategy_satisfies_conditions(pomdp, algorithm, initial_belief);
    }
}

int get_next_classical_state_to_try(Algorithm *algorithm) {
    set<int> succ_cstates = algorithm->get_successor_classical_states(algorithm->classical_state);

    if (algorithm->children.size() > 0) {
        int max_ = algorithm->children[0]->classical_state;
        for (auto child : algorithm->children) {
            max_ = max(max_, child->classical_state);
        }
        auto it = succ_cstates.upper_bound(max_);
        if (it == succ_cstates.end()) {
            return -1;
        }
        return *it;
    } else {
        auto it = succ_cstates.begin();
        if (it == succ_cstates.end()) {
            return -1;
        }
        return *it;
    }
    
}

void get_matrix_maximin(POMDP &pomdp, 
    const vector<int> &initial_states, 
    Algorithm *current_algorithm, 
    unordered_map<int, unordered_map<int, double>> &minimax_matrix,
    const int &max_horizon,
    unordered_map<int, Algorithm*> &mapping_index_algorithm) {
        set_minimax_values(pomdp, current_algorithm, initial_states, minimax_matrix, mapping_index_algorithm);

        vector<Algorithm *> end_nodes;
        if (current_algorithm != nullptr)
            get_algorithm_end_nodes(current_algorithm, end_nodes);

        if (end_nodes.size() == 0) {
            if (max_horizon >= 1) {
                for (auto action : pomdp.actions) {
                    Algorithm * new_node = new Algorithm(action, 0, 1); // assumes initial classical state is 0
                    get_matrix_maximin(pomdp, initial_states, new_node, minimax_matrix, max_horizon, mapping_index_algorithm);
                    delete new_node;
                }
            }
            
        } else {
            for (auto end_node : end_nodes) {
                if (end_node->depth < max_horizon) {
                    for (string action : pomdp.actions) {
                        if (end_node->is_measurement){
                            int next_classical_state = get_next_classical_state_to_try(end_node);
                            if (next_classical_state > -1) {
                                Algorithm * new_node = new Algorithm(action, next_classical_state, end_node->depth + 1);
                                end_node->children.push_back(new_node);
                                get_matrix_maximin(pomdp, initial_states, current_algorithm, minimax_matrix, max_horizon, mapping_index_algorithm);
                                end_node->children.pop_back();
                                delete new_node;
                            }
                            
                        } else {
                            assert(end_node->children.size() == 0);
                            Algorithm * new_node = new Algorithm(action, end_node->classical_state, end_node->depth + 1);
                            
                            end_node->children.push_back(new_node);
                            get_matrix_maximin(pomdp, initial_states, current_algorithm, minimax_matrix, max_horizon, mapping_index_algorithm);
                            end_node->children.pop_back();
                            delete new_node;
                        }
                    }
                }
            }   
        }
}

