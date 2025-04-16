#include <string>
#include <utility>
#include <vector>
#include <unordered_set>
#include <sstream>
#include <fstream>
#include <cassert>
#include <map>
#include "fp.cpp"
#include "json.hpp"

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

    Algorithm(string action, int classical_state, int depth=-1){
        this->action = std::move(action);
        this->classical_state = classical_state;
        this->depth = depth;
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
        return result;
    }
};


void write_algorithm_file(Algorithm *algorithm, const string &output_path) {
    json serialized_algorithm = algorithm->serialize();
    ofstream f(output_path);
    f << serialized_algorithm.dump(4) << endl;
    f.close();
}


string get_project_path() {
    return "..";
}


