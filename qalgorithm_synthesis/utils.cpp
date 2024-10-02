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

struct Instruction {
    int control{}; // only CX supported as a multi-qubit gate
    int target{};
    vector<string> params;
    string instruction;
};


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

    [[nodiscard]] MyFloat get_vertices_probs(const unordered_set<int> &target_vertices) const {
        MyFloat val;

        for(const auto & prob : this->probs) {
            if (target_vertices.find(prob.first) != target_vertices.end()) {
                val = val + prob.second;
            }
        }

        return val;
    }

    void check() {
        if (this->get_sum() != MyFloat("1")) {
            assert(false);
        }
    }
};

class Algorithm {
public:
    Instruction instruction;
    Algorithm *next_ins;
    Algorithm *case0; // We assume that single-qubit measurements are possible only, therefore only two cases
    Algorithm *case1;

    Algorithm(Instruction instruction, Algorithm *next_ins, Algorithm *case0, Algorithm *case1){
        this->instruction = std::move(instruction);
        this->next_ins = next_ins;
        this->case0 = case0;
        this->case1 = case1;
    }

    Algorithm(json data){
        auto instruction = Instruction();
        instruction.instruction = data["instruction"];
        instruction.control = data["control"];
        instruction.target = data["target"];
        this->instruction = instruction;

        if (data["next"] == -1){
            this->next_ins = nullptr;
        }else{
            this->next_ins = new Algorithm(data["next"]);
        }

        if (data["case0"] == -1) {
            this->case0 = nullptr;
        } else {
            this->case0 = new Algorithm(data["case0"]);
        }

        if (data["case1"] == -1) {
            this->case1 = nullptr;
        } else{
            this->case1 = new Algorithm(data["case1"]);
        }
    }

    json serialize() const {
        if (this == nullptr) {
            return "None";
        }
        json serialized_next_ins;

        if (this->next_ins == nullptr) {
            serialized_next_ins = "None";
        } else {
            serialized_next_ins = this->next_ins->serialize();
        }

        json serialized_case0;
        if(this->case0  == nullptr) {
            serialized_case0 = "None";
        } else {
            serialized_case0 = this->case0->serialize();
        }

        json serialized_case1;
        if(this->case1 == nullptr) {
            serialized_case1 = "None";
        } else {
            serialized_case1 = this->case1->serialize();
        }

        json control;
        if (this->instruction.control == -1) {
            control = "None";
        } else {
            control = this->instruction.control;
        }
        assert(this->instruction.target != -1);

        json result;
        result["instruction"] = this->instruction.instruction;
        result["target"] = this->instruction.target;
        result["control"] = control;
        result["params"] = this->instruction.params;
        result["next"] = serialized_next_ins;
        result["case0"] = serialized_case0;
        result["case1"] = serialized_case1; 
        return result;
    }
};


void write_algorithm_file(Algorithm *algorithm, const string &output_path) {
    json serialized_algorithm = algorithm->serialize();
    ofstream f(output_path);

    
    f << serialized_algorithm.dump(4) << endl;
    f.close();
}





