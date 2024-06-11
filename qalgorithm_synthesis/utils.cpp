#include <string>
#include <utility>
#include <vector>
#include <unordered_set>
#include <sstream>
#include <fstream>
#include <cassert>
#include <map>
#include "fp.cpp"

using namespace std;
string DIR_PREFIX = "./";
string EXP_INDEX = "1";

struct Instruction {
    int control{}; // only CX supported as a multi-qubit gate
    int target{};
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

    string serialize(const unordered_map<int, int> &inverse_mapping) const {
        if (this == nullptr) {
            return "None";
        }
        string serialized_next_ins;

        if (this->next_ins == nullptr) {
            serialized_next_ins = "None";
        } else {
            serialized_next_ins = this->next_ins->serialize(inverse_mapping);
        }

        string serialized_case0;
        if(this->case0  == nullptr) {
            serialized_case0 = "None";
        } else {
            serialized_case0 = this->case0->serialize(inverse_mapping);
        }

        string serialized_case1;
        if(this->case1 == nullptr) {
            serialized_case1 = "None";
        } else {
            serialized_case1 = this->case1->serialize(inverse_mapping);
        }

        string control;
        if (this->instruction.control == -1) {
            control = "None";
        } else {
            control = to_string(inverse_mapping.find(this->instruction.control)->second);
        }
        assert(this->instruction.target != -1);

        string result = "{";
        result+= "'instruction': " + this->instruction.instruction + ",";
        result+= "'target': " + to_string(inverse_mapping.find(this->instruction.target)->second) + ",";
        result+= "'control': " + control + ",";
        result+= "'params' : None,";
        result+= "'next' : " + serialized_next_ins + ",";
        result+= "'case0' : " + serialized_case0 + ",";
        result+= "'case1' : " + serialized_case1;
        result += "}\n";
        return result;

    }
};


void write_algorithm_file(Algorithm *algorithm, const string &output_path,
                          const unordered_map<int, int> &inverse_mapping) {
    auto serialized_algorithm = algorithm->serialize(inverse_mapping);
    ofstream f(output_path);

    f << "import os, sys" << endl;
    f << "sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))" << endl;
    f << "from utils import *" << endl << endl;
    f << "algorithms = [" << serialized_algorithm << "]" << endl;
    f.close();
}

unordered_map<int, int> get_inverse_mapping(const string &backend, const int &index) {
    string filename = DIR_PREFIX + "inverse_mappings"+EXP_INDEX+"/" + backend + "_" + to_string(index) + ".txt";
    cout << filename << endl;
    fstream f(filename);
    string line0, line1, line2;

    getline(f, line0); getline(f, line1); getline(f, line2);

    vector<string> v0, v1, v2;

    split_str(line0, ' ', v0); split_str(line1, ' ', v1);
    split_str(line2, ' ', v2);

    assert(v0.size() == 2); assert(v1.size() == 2); assert(v2.size() == 2);

    assert(stoi(v0[1]) == 0); assert(stoi(v1[1]) == 1); assert(stoi(v2[1]) == 2);

    unordered_map<int, int> result;
    result[stoi(v0[0])] = 0; result[stoi(v1[0])] = 1; result[stoi(v2[0])] = 2;

    return result;
}

unordered_map<string, int> backends_to_embeddings = {{"fake_athens",3},
{"fake_belem",3},
{"fake_tenerife",2},
{"fake_lima",3},
{"fake_rome",3},
{"fake_manila",3},
{"fake_santiago",3},
{"fake_bogota",3},
{"fake_ourense",3},
{"fake_yorktown",6},
{"fake_essex",3},
{"fake_vigo",3},
{"fake_burlington",3},
{"fake_quito",3},
{"fake_london",3},
{"fake_jakarta",5},
{"fake_oslo",5},
{"fake_perth",5},
{"fake_lagos",5},
{"fake_nairobi",5},
{"fake_casablanca",5},
{"fake_melbourne",8},
{"fake_guadalupe",4},
{"fake_tokyo",10},
{"fake_poughkeepsie",6},
{"fake_johannesburg",5},
{"fake_boeblingen",7},
{"fake_almaden",9},
{"fake_singapore",6},
{"fake_mumbai",8},
{"fake_paris",7},
{"fake_auckland",2},
{"fake_kolkata",7},
{"fake_toronto",9},
{"fake_montreal",4},
{"fake_sydney",6},
{"fake_cairo",10},
{"fake_hanoi",7},
{"fake_geneva",3},
{"fake_cambridge",5},
{"fake_rochester",5},
{"fake_brooklyn",6},
{"fake_manhattan",7},
{"fake_washington",7}};


