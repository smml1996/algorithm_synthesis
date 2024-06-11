//
// Created by Stefanie Muroya Lei on 28.01.24.
//
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <string>
#include "utils.cpp"
#include <fstream>

using namespace  std;

class POMDP {
public:
    //            from                      I         to    prob
    unordered_map< int , unordered_map< string, map< int, MyFloat > > > probabilities;
    int initial_state{};

    unordered_map<int, int> gamma{};
    unordered_set<int> target_vertices{};
    unordered_map< string, vector< Instruction > > actions_to_instructions;

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

    void insert_gamma(const int &v, const int &obs) {
        assert(this->gamma.find(v) == this->gamma.end());
        this->gamma[v] = obs;
    }

    bool is_target_vertex(const int &id) {
        return this->target_vertices.find(id) != this->target_vertices.end();
    }

    bool insert_target_v(const int &id) {
        if (this->is_target_vertex(id)) {
            return false;
        }

        this->target_vertices.insert(id);
        return true;
    }
};

pair<int, int> check_states_integrity(const string &line) {
    vector<string> out1;
    split_str(line, ' ', out1);

    assert(out1.size() == 2);
    assert(out1[0] == "STATES:");

    vector<string> out2;
    split_str(out1[1], ',', out2);
    int prev = stoi(out2[0]);
    int state0 = prev;
    for (int i = 1; i < out2.size(); i++) {
        assert(stoi(out2[i]) == prev + 1);
        prev = stoi(out2[i]);
    }
    return make_pair(state0, prev); // all the states-ids are within this range
}

void fill_pomdp_gamma(POMDP &pomdp, const pair<int, int> &states_range, const string &line){
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
        assert(v >= states_range.first);
        assert(v <= states_range.second);
        pomdp.insert_gamma(v, obs);
    }
}

void fill_target_vertices(POMDP &pomdp, const string &line) {
    vector<string> elements;
    split_str(line, ' ', elements);

    assert(elements.size() ==  2);
    assert(elements[0] == "TARGETV:");

    vector<string> vertices;
    split_str(elements[1], ',', vertices);

    for (const auto & vertice : vertices) {
        assert(pomdp.insert_target_v(stoi(vertice)));
    }

}

POMDP parse_pomdp_file (const string& fname) {
    POMDP pomdp;
    ifstream f(fname + ".txt");
    cout << fname << endl;
    string line;
    if (getline (f, line)) {
        assert (line == "BEGINPOMDP");
    } else {
        cout << "error reading file" << endl;
        assert(false);
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
    getline(f, line);
    pair<int, int> states_range = check_states_integrity(line); // this is the line of all the states of the POMDP
    assert(initial_state == states_range.first);

    // target vertices
    getline(f, line);
    fill_target_vertices(pomdp, line);

    // GAMMA:
    getline(f, line);
    fill_pomdp_gamma(pomdp, states_range, line);

    // Actions
    getline(f, line);
    assert(line == "BEGINACTIONS");
    getline(f, line);
    while(line != "ENDACTIONS"){
        vector<string> elements;
        split_str(line, ' ', elements);
        assert(elements.size() == 4);
        string channel_name = elements[0];
        assert(pomdp.actions_to_instructions.find(channel_name) == pomdp.actions_to_instructions.end()); // it should be a unique name that we havent already used
        pomdp.actions_to_instructions[channel_name] = vector<Instruction>();

        vector<string> instructions;
        split_str(elements[1], ',', instructions);

        vector<string> controls;
        split_str(elements[2], ',', controls);

        vector<string> targets;
        split_str(elements[3], ',', targets);

        assert(instructions.size() == controls.size());
        assert(targets.size() == controls.size());

        for (int i = 0; i < instructions.size(); i++) {
            int control;
            if (controls[i] == "None") {
                control = -1;
            } else {
                control = stoi(controls[i]);
            }
            int target = stoi(targets[i]);


            Instruction ins;
            ins.instruction = instructions[i];
            ins.target = target;
            ins.control = control;

            pomdp.actions_to_instructions[channel_name].push_back(ins);
        }

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

pair<Algorithm *, Algorithm*> get_instructions_algorithm(const vector<Instruction> &instructions) {
    assert(!instructions.empty());

    auto *initial_node = new Algorithm(instructions[0], nullptr, nullptr, nullptr);
    Algorithm *current_node = initial_node;
    for (int i = 1; i < instructions.size(); i++){
        current_node = new Algorithm(instructions[i], nullptr, nullptr, nullptr);
        initial_node->next_ins = current_node;
    }

    return make_pair(initial_node, current_node);

}

pair<Algorithm*, MyFloat> get_bellman_value(POMDP &pomdp, Belief &current_belief, const int &horizon) {

    MyFloat curr_belief_val = current_belief.get_vertices_probs(pomdp.target_vertices);

    if (horizon == 0) {
        return make_pair((Algorithm *) nullptr, curr_belief_val);
    }

    vector< pair< Algorithm*, MyFloat > > bellman_values;

    bellman_values.emplace_back((Algorithm *) nullptr, curr_belief_val);

    for(auto it = pomdp.actions_to_instructions.begin(); it != pomdp.actions_to_instructions.end(); it++) {
        string action = it->first;

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
            vector<Algorithm *>next_algorithms;
            MyFloat bellman_val;

            for(auto & obs_to_next_belief : obs_to_next_beliefs) {
                auto temp = get_bellman_value(pomdp, obs_to_next_belief.second, horizon-1);
                next_algorithms.push_back(temp.first);
                bellman_val = bellman_val + temp.second;
            }

            assert(!next_algorithms.empty());
            assert(next_algorithms.size() < 3);

            Algorithm *new_alg_node;
            pair<Algorithm*, Algorithm*> algs_pointer = get_instructions_algorithm(it->second);
            new_alg_node = algs_pointer.first;
            if (next_algorithms.size() == 1) {
                algs_pointer.second->next_ins = next_algorithms[0];
            } else {
                // since maps are ordered by values:
                algs_pointer.second->case0 = next_algorithms[0]; // this should be a measure to 0
                algs_pointer.second->case1 = next_algorithms[1]; // This should be a measure to 1
            }

            bellman_values.emplace_back(new_alg_node, bellman_val);
        }
    }

    MyFloat max_val;
    for(auto & bellman_value : bellman_values) {
        max_val = max(max_val, bellman_value.second);
    }

    for(auto & bellman_value : bellman_values) {
        if (bellman_value.second == max_val) {
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

