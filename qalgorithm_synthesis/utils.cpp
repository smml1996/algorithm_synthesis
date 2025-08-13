#include <string>
#include <utility>
#include <vector>
#include <set>
#include <unordered_set>
#include <sstream>
#include <fstream>
#include <cassert>
#include <map>
#include "fp.cpp"
#include "json.hpp"
#include "ortools/linear_solver/linear_solver.h"

// for convenience
using json = nlohmann::json;

using namespace std;


void split_str(string const &str, const char delim, vector<string> &out) {
    stringstream s(str);

    string s2;

    while(getline(s, s2, delim)){
        out.push_back(s2);
    }
}

class Belief {
    MyFloat get_sum(){
        MyFloat result;

        for (auto & prob : this->probs) {
            result = result + prob.second;
        }

        return result;
    }
public:
    map<int, MyFloat> probs;
    MyFloat get(const int &v) {
        if(this->probs.find(v) == this->probs.end()){
            return MyFloat();
        }
        return this->probs[v];
    }

    void set_val(const int &v, const MyFloat &prob) {
        this->probs[v] = prob;
    }

    void add_val(const int &v, const MyFloat &val) {
        this->probs[v] = this->get(v) + val;
    }

    [[nodiscard]] MyFloat get_belief_reward(const unordered_map<int, MyFloat> &rewards, const string &opt_technique, const MyFloat &threshold) const {
        // returns expected reward
        MyFloat val;

        for(const auto & prob : this->probs) {
            MyFloat r = rewards.find(prob.first)->second;
            val = val + (r * prob.second);
        }
        if (opt_technique == "target") {
            if (val == threshold or val > threshold) {
                return MyFloat("1");
            } else{
                return MyFloat();
            }
        } else {
            return val;
        }
    }

    void check() {
        if (this->get_sum() != MyFloat("1")) {
            assert(false);
        }
    }
};

class Algorithm {
public:
    string action;
    vector<Algorithm*> children;
    int classical_state;
    int depth;
    bool is_measurement;
    unordered_map<int, double> children_probs;

    Algorithm(string action, int classical_state, int depth=-1){
        this->action = std::move(action);
        this->classical_state = classical_state;
        this->depth = depth;
        this->is_measurement = action.rfind("MEAS", 0) == 0;
        this->children_probs = unordered_map<int, double>();
    }

    Algorithm(json data){
        this->action = data["action"];
        this->depth = data["depth"];
        this->classical_state = data["classical_state"];
        if (data["children"] != "None"){
            for (int i = 0; i < data["children"].size(); i++) {
                if (data["children"][i] != "None") {
                    this->children.push_back(new Algorithm(data["children"][i]));
                }
                
            }
        }
    }

    vector<int> get_modified_bits() {
        vector<int> result;
        if (this->is_measurement) {
            vector<string> elements;
            split_str(this->action, '-', elements);

            for (int i = 1; i < elements.size(); i++) {
                int bit = stoi(elements[i]);
                result.push_back(bit);
            }
        }
        result;
    } 

    set<int> get_successor_classical_states(const int &current_classical_state) {
        set<int> result;
        vector<int> bits_to_modify = this->get_modified_bits();

        for (auto bit_index : bits_to_modify) {
            int new_classical_state = current_classical_state ^ (1 << bit_index);
            result.insert(new_classical_state);
        }

        result;
    }

    json serialize() const {
        if (this == nullptr) {
            return "None";
        }

        vector<json> children;

        for (int i = 0; i < this->children.size(); i++) {
            children.push_back(this->children[i]->serialize());
        }

        json result;
        result["action"] = this->action;
        result["classical_state"] = this->classical_state;
        result["children"] = children;
        result["depth"] = depth;
        result["children_probs"] = this->children_probs;
        return result;
    }

      

    bool exist_child_with_cstate(const int &cstate) {
        for (auto child : this->children) {
            if(child->classical_state == cstate) {
                return true;
            }
        }
        return false;
    }
};

Algorithm * deep_copy_algorithm(Algorithm *algorithm)  {
    string action = algorithm->action;
    int classical_state = algorithm-> classical_state;
    int depth = algorithm->depth;

    Algorithm * algorithm_copy = new Algorithm(action, classical_state, depth);

    for (auto child : algorithm->children) {
        algorithm_copy->children.push_back(deep_copy_algorithm(child));
    }

    return algorithm_copy;
}

void get_algorithm_end_nodes(Algorithm *algorithm, vector<Algorithm *> &end_nodes) {
    if (algorithm->children.size() == 0) {
        end_nodes.push_back(algorithm);
        return;
    }

   if (algorithm->is_measurement) {
        if (algorithm->children.size() < algorithm->get_successor_classical_states(algorithm->classical_state).size()) {
            end_nodes.push_back(algorithm);
        }
   }
    

    for (auto child : algorithm->children) {
        get_algorithm_end_nodes(child, end_nodes);
    }
}


void write_algorithm_file(Algorithm *algorithm, const string &output_path) {
    json serialized_algorithm = algorithm->serialize();
    ofstream f(output_path);
    f << serialized_algorithm.dump(4) << endl;
    f.close();
}


string get_project_path() {
    return "..";
}

vector<double> solve_lp_maximin(const unordered_map<int, unordered_map<int, double>> &maximin_matrix, const int &n_algorithms, const int &n_initial_states) {

    operations_research::MPSolver solver("max_v", operations_research::MPSolver::GLOP_LINEAR_PROGRAMMING);

    // Variables: x_i >= 0
    std::vector<operations_research::MPVariable*> x(n_algorithms);
    for (int i = 0; i < n_algorithms; ++i) {
        x[i] = solver.MakeNumVar(0.0, 1.0, "x_" + std::to_string(i));
    }

    // Variable: v
    operations_research::MPVariable* v = solver.MakeNumVar(0.0, 1.0, "v");

    // Constraint: sum_i x_i = 1
    operations_research::MPConstraint* prob_sum = solver.MakeRowConstraint(1.0, 1.0);
    for (int i = 0; i < n_algorithms; ++i) {
        prob_sum->SetCoefficient(x[i], 1.0);
    }

    // Constraints: sum_i x_i * M_ij >= v  for all j
    for (int j = 0; j < n_initial_states; ++j) {
        operations_research::MPConstraint* c = solver.MakeRowConstraint(0.0, solver.infinity());
        for (int i = 0; i < n_algorithms; ++i) {
            auto temp = *maximin_matrix.find(i);
            c->SetCoefficient(x[i], (*(temp.second.find(j))).second);
        }
        c->SetCoefficient(v, -1.0); // sum_i(...) - v >= 0  → sum_i(...) >= v
    }

    // Objective: maximize v
    operations_research::MPObjective* objective = solver.MutableObjective();
    objective->SetCoefficient(v, 1.0);
    objective->SetMaximization();

    // Solve
    auto result = solver.Solve();
    vector<double> mixed_algorithm;
    if (result == operations_research::MPSolver::OPTIMAL) {
        for (int i = 0; i < n_algorithms; ++i) {
            mixed_algorithm.push_back(x[i]->solution_value());
        }
    }

    return mixed_algorithm;
}

Algorithm *get_mixed_algorithm(const vector<double> &x, const unordered_map<int, Algorithm *> &mapping_index_algorithm) {
    Algorithm * new_head = new Algorithm("RANDOM", 0);

    for(int i = 0; i < x.size(); i++) {
        if(x[i] > 0) {
            new_head->children.push_back(mapping_index_algorithm.find(i)->second);
            assert(new_head->children_probs.find(i) == new_head->children_probs.end());
            new_head->children_probs.insert({i, x[i]});
        }
    }

    return new_head;
}

