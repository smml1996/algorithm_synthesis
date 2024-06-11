//
// Created by Stefanie Muroya Lei on 28.01.24.
//
#include<vector>
#include <cassert>
#include <string>
#include <iostream>
#include <algorithm>

using namespace std;

class MyFloat {
    static const int precision = 80;
    static const int tolerance = 80;
    static void check_digit(const short &digit) {
        assert(digit > -1);
        assert(digit < 10);
    }

    static string remove_initial_zeros(const string &original) {
        string result;
        bool found_non_zero = false;
        bool dot_found = false;
        for(auto c : original) {

            if (dot_found or found_non_zero) {
                result += c;
            }else if (c == '.'){
                dot_found = true;
                if (result.empty()) {
                    result += '0';
                }
                result += c;
            } else if(c != '0') {
                found_non_zero = true;
                result += c;
                if(c == '.'){}
            }
        }
        if (result.empty()){
            result += '0';
        }
        return result;
    }
    static vector<short> integer_addition(const vector<short> &v1, const vector<short> &v2) {
        vector<short> result;
        assert(v1.size() == v2.size());

        int carry = 0;
        short val;
        for(int i = 0; i < v1.size(); i++) {
            val = (short) (v1[i] + v2[i] + carry);
            result.push_back((short) (val % 10));
            carry = val > 9 ? 1 : 0;
        }

        assert(carry == 0);
        return result;
    }
public:
    vector<short> exponent;
    vector<short> mantissa;
    friend ostream& operator<<(ostream& os, const MyFloat& myfloat) {

        for (int i = myfloat.exponent.size()-1 ; i >= 0; i--) {
            os << to_string(myfloat.exponent[i]);
        }
        os << ".";
        for (int i = myfloat.mantissa.size()-1 ; i >= 0; i--) {
            os << to_string(myfloat.mantissa[i]);
        }
        return os;
    }

    explicit MyFloat(const string& probability_ = "0", int custom_precision=-1){
        string probability = MyFloat::remove_initial_zeros((probability_));
        int actual_precision = custom_precision == -1 ? MyFloat::precision : custom_precision;
        assert(!probability.empty());
        bool dot_found = false;
        for (auto c : probability) {
            assert((c == '.') or (c >= '0' and c <= '9'));
            if(c == '.') {
                dot_found = true;
            }else if (!dot_found) {
                this->exponent.push_back(short(c - '0'));
            } else if (this->mantissa.size() < actual_precision){
                this->mantissa.push_back(short(c - '0'));
            }
        }
        assert(exponent.size() == 1);

        while (this->mantissa.size() < actual_precision) {
            this->mantissa.push_back(0);
        }
        reverse(this->mantissa.begin(), this->mantissa.end()); // operations need to start from the least significant decimal
    }

    MyFloat operator+(MyFloat const &other){
        MyFloat result;
        result.mantissa.clear();
        result.exponent.clear();
        short carry = 0;
        assert(this->mantissa.size() == other.mantissa.size());
        for(int i = 0; i < this->mantissa.size(); i++) {
            MyFloat::check_digit(this->mantissa[i]);
            MyFloat::check_digit(other.mantissa[i]);
            auto val = (short) (carry + this-> mantissa[i] + other.mantissa[i]);
            result.mantissa.push_back((short) (val % 10));
            if (val > 9) {
                carry = 1;
            } else {
                carry = 0;
            }
        }

        assert(other.exponent.size() == 1);
        assert(this->exponent.size() == 1);
        auto val = (short) (other.exponent[0] + this->exponent[0] + carry);
        assert(val < 2);
        result.exponent.push_back(val);
        return result;
    }

    MyFloat operator*(MyFloat const &other) {
        // We treat it as integer multiplication
        vector<short> n1;
        vector<short> n2;
        vector<short> result;

        assert(other.mantissa.size() == this->mantissa.size());
        assert(other.exponent.size() == this->exponent.size());

        for(int i = 0; i < other.mantissa.size(); i++){
            n1.push_back(this->mantissa[i]);
            n2.push_back(other.mantissa[i]);
        }


        for(int i = 0; i < other.exponent.size(); i++) {
            n1.push_back(this->exponent[i]);
            n2.push_back(other.exponent[i]);
        }




        while(result.size() < 2*n1.size()) {
            result.push_back(0);
        }

        // actual multiplication
        int carry;
        int val;
        for(int shift_digit = 0; shift_digit < n2.size(); shift_digit++) {
            // perform multiplication of n1 times the digit at position shift_digit in n2
            vector<short> n1_times_digit;

            n1_times_digit.reserve(shift_digit);
            for(int j = 0 ; j < shift_digit; j++) {
                n1_times_digit.push_back(0);
            }

            carry = 0;

            for (auto d : n1) {
                val = d*n2[shift_digit] + carry;
                n1_times_digit.push_back((short) (val%10));
                carry = val / 10;
            }
            n1_times_digit.push_back((short) carry);
            assert(n1_times_digit.size() <= 2*n1.size());

            while(n1_times_digit.size() < 2*n1.size()) {
                n1_times_digit.push_back(0);
            }
            result = MyFloat::integer_addition(n1_times_digit, result);
        }



        // convert result to string. Note: by creating a float it trims it to the actual precision
        string res;
        for(int i = 0; i < 2*this->mantissa.size(); i++) {
            res+= '0' + result[i];
        }
        res += ".";

        for(int i = 2*this->mantissa.size(); i < result.size(); i++) {
            res += '0' + result[i];
        }
        reverse(res.begin(), res.end());
        res = MyFloat::remove_initial_zeros(res);
        return MyFloat(res);
    }

    bool operator==(const MyFloat &rhs) {
        assert(this->exponent.size() == rhs.exponent.size());
        assert(this->mantissa.size() == rhs.mantissa.size());
        for (int i = 0; i < rhs.exponent.size(); i++) {
            if (rhs.exponent[i] != this->exponent[i]){
                return false;
            }
        }
        for(int i = 0; i < MyFloat::tolerance; i++) {
            
            if (this->mantissa[this->mantissa.size() - i-1] != rhs.mantissa[rhs.mantissa.size() - i-1]){
                return false;
            }
        }
        return true;
    }

    inline bool operator!=(const MyFloat &rhs) {
        return !(*this == rhs);
    }

    inline bool operator>(const MyFloat &other) const {
        // we have to check starting from the most significant digit
        for(int i = this->exponent.size()-1; i >= 0; i--) {
            if(this->exponent[i] > other.exponent[i]) {
                return true;
            }else if(this->exponent[i] < other.exponent[i]) {
                return false;
            }
        }

        for(int i = this->mantissa.size() -1; i >= 0; i--) {
            if(this->mantissa[i] > other.mantissa[i]) {
                return true;
            } else if(this->mantissa[i] < other.mantissa[i]) {
                return false;
            }
        }

        return false;
    }

    inline bool operator<(const MyFloat &other) const {
        return !(*this > other);
    }
};

MyFloat max(MyFloat const &a, MyFloat const &b) {
    if(b > a) {
        return b;
    }
    return a;
}


