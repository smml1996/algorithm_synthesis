#include <gtest/gtest.h>
#include "fp.cpp"


TEST(FPTest, FPInit) {
  EXPECT_EQ(MyFloat("-1").is_negative, true);
  EXPECT_EQ(MyFloat("-1.9126").is_negative, true);
  EXPECT_EQ(MyFloat().is_negative, false);
  EXPECT_EQ(MyFloat("3").is_negative, false);
}

// Demonstrate some basic assertions.
TEST(FPTest, Equality) {
  EXPECT_TRUE(MyFloat("1.00") == MyFloat("1"))<< "1.00 == 1";
  EXPECT_TRUE(MyFloat("1.00") == MyFloat("1.")) << "1.00 == 1.";
  EXPECT_TRUE(MyFloat("00") == MyFloat())<< "00 == ";
  EXPECT_TRUE(MyFloat() == MyFloat("0.00")) << " == 0.00";

  EXPECT_TRUE(MyFloat("-1.00") == MyFloat("-1"))<< " -1.00 == -1";
  EXPECT_TRUE(MyFloat("-1.53") == MyFloat("-1.530")) << "-1.53 == -1.530";
  EXPECT_TRUE(MyFloat("-100.53") == MyFloat("-100.530")) << "-100.53 == -100.530";

  EXPECT_FALSE(MyFloat("-100") == MyFloat("-1.918")) << "-100 != -1.918";
  EXPECT_FALSE(MyFloat() == MyFloat("-1")) << " != -1";
  EXPECT_FALSE(MyFloat("1") == MyFloat("-1")) << " 1 != -1";
}

TEST(FPTest, Greater) {
  EXPECT_FALSE(MyFloat("1.0") > MyFloat("1")) << "equal elements should not be greater"; 
  EXPECT_FALSE(MyFloat("-1.0") > MyFloat("1")) << "equal elements should not be greater"; 
  EXPECT_FALSE(MyFloat() > MyFloat()) << "equal elements should not be greater";

  EXPECT_FALSE(MyFloat("-1.0") > MyFloat("1")) << "equal elements should not be greater"; 
  EXPECT_FALSE(MyFloat("-1.59") > MyFloat("1")); 
  EXPECT_FALSE(MyFloat("-192.59") > MyFloat("1")); 
  EXPECT_FALSE(MyFloat("-192.59") > MyFloat("1.192351")); 
  EXPECT_FALSE(MyFloat("-192.59543213") > MyFloat("1.192351")); 
  EXPECT_FALSE(MyFloat("-192.59543213") > MyFloat("-192"));
  EXPECT_FALSE(MyFloat("-192.59543213") > MyFloat("-1")); 
  EXPECT_FALSE(MyFloat("-120") > MyFloat("100"));
  EXPECT_FALSE(MyFloat("-0.723") > MyFloat("-0.567")); 

  EXPECT_TRUE(MyFloat("-12") > MyFloat("-100"));
  EXPECT_TRUE(MyFloat("120") > MyFloat("-100"));
  EXPECT_TRUE(MyFloat("120") > MyFloat("100"));
  EXPECT_TRUE(MyFloat("12.123") > MyFloat("10.12"));
  EXPECT_TRUE(MyFloat("120.123") > MyFloat("120.12"));
  EXPECT_TRUE(MyFloat("-0.123") > MyFloat("-0.567")); 
  
}

TEST(FPTest, Least) {
  EXPECT_FALSE(MyFloat("1.0") < MyFloat("1")) << "equal elements should not be greater"; 
  EXPECT_FALSE(MyFloat() < MyFloat()) << "equal elements should not be greater";
  EXPECT_TRUE(MyFloat("-1.0") < MyFloat("1")); 
  

  EXPECT_TRUE(MyFloat("-1.0") < MyFloat("1")); 
  EXPECT_TRUE(MyFloat("-1.59") < MyFloat("1")); 
  EXPECT_TRUE(MyFloat("-192.59") < MyFloat("1")); 
  EXPECT_TRUE(MyFloat("-192.59") < MyFloat("1.192351")); 
  EXPECT_TRUE(MyFloat("-192.59543213") < MyFloat("1.192351")); 
  EXPECT_TRUE(MyFloat("-192.59543213") < MyFloat("-192"));
  EXPECT_TRUE(MyFloat("-192.59543213") < MyFloat("-1")); 
  EXPECT_TRUE(MyFloat("-120") < MyFloat("100"));
  EXPECT_TRUE(MyFloat("-0.723") < MyFloat("-0.567")); 

  EXPECT_FALSE(MyFloat("-12") < MyFloat("-100"));
  EXPECT_FALSE(MyFloat("120") < MyFloat("-100"));
  EXPECT_FALSE(MyFloat("120") < MyFloat("100"));
  EXPECT_FALSE(MyFloat("12.123") < MyFloat("10.12"));
  EXPECT_FALSE(MyFloat("120.123") < MyFloat("120.12"));
  EXPECT_FALSE(MyFloat("-0.123") < MyFloat("-0.567")); 
  
}

TEST(FPTest, Abs) {
  EXPECT_TRUE( MyFloat::abs(MyFloat()) == MyFloat()) << MyFloat::abs(MyFloat());
  EXPECT_TRUE( MyFloat::abs(MyFloat("-1")) == MyFloat("1")) << MyFloat::abs(MyFloat("-1"));
  EXPECT_TRUE( MyFloat::abs(MyFloat("1")) == MyFloat("1")) << MyFloat::abs(MyFloat("1"));
  EXPECT_TRUE( MyFloat::abs(MyFloat("-0.29341")) == MyFloat("0.29341")) << MyFloat::abs(MyFloat("-0.29341"));
  EXPECT_TRUE( MyFloat::abs(MyFloat("0.29341")) == MyFloat("0.29341")) << MyFloat::abs(MyFloat("0.29341"));
}

TEST(FPTest, DigitSubstraction) {
  EXPECT_EQ(MyFloat::digit_substraction(9,1,0), make_pair(8, 0));
  EXPECT_EQ(MyFloat::digit_substraction(9,1,1), make_pair(7, 0));
  EXPECT_EQ(MyFloat::digit_substraction(1,1,0), make_pair(0, 0));
  EXPECT_EQ(MyFloat::digit_substraction(2,5,0), make_pair(7, 1));
}

TEST(FPTest, Addition) {

  EXPECT_TRUE((MyFloat() + MyFloat()) == MyFloat());
  EXPECT_TRUE((MyFloat("123.123") + MyFloat("-123.123")) == MyFloat());
  EXPECT_TRUE((MyFloat("-123.123") + MyFloat("123.123")) == MyFloat());
  EXPECT_TRUE((MyFloat("-10") + MyFloat("-1")) == MyFloat("-11")) << (MyFloat("-10") + MyFloat("-1")) ;
  EXPECT_TRUE((MyFloat("1") + MyFloat("1")) == MyFloat("2")) << (MyFloat("1") + MyFloat("1"));
  EXPECT_TRUE((MyFloat("-0.1") + MyFloat("-0.1")) == MyFloat("-0.2")) << (MyFloat("-0.1") + MyFloat("-0.1"));
  EXPECT_TRUE((MyFloat("0.1") + MyFloat("-0.1")) == MyFloat()) << (MyFloat("0.1") + MyFloat("-0.1"));
  EXPECT_TRUE((MyFloat("0.1") + MyFloat("0.1")) == MyFloat("0.2")) << (MyFloat("0.1") + MyFloat("0.1"));

  EXPECT_TRUE((MyFloat("10") + MyFloat("-1")) == MyFloat("9")) << (MyFloat("10") + MyFloat("-1"));
  EXPECT_TRUE((MyFloat("-10") + MyFloat("1")) == MyFloat("-9")) << (MyFloat("-10") + MyFloat("1"));
  EXPECT_TRUE((MyFloat("-0.1") + MyFloat("1")) == MyFloat("0.9")) << (MyFloat("-0.1") + MyFloat("1"));
  
  
}


TEST(FPTest, Multiplication){
  EXPECT_TRUE((MyFloat() * MyFloat()) == MyFloat());

  EXPECT_TRUE((MyFloat("1") * MyFloat("-1")) == MyFloat("-1"));
  EXPECT_TRUE((MyFloat("-1") * MyFloat("-1")) == MyFloat("1"));
  EXPECT_TRUE((MyFloat("-1") * MyFloat("1")) == MyFloat("-1"));
  EXPECT_TRUE((MyFloat("1") * MyFloat("1")) == MyFloat("1"));

  EXPECT_TRUE((MyFloat("0.1") * MyFloat("1")) == MyFloat("0.1"));
  EXPECT_TRUE((MyFloat("0.1") * MyFloat("-1")) == MyFloat("-0.1"));

  EXPECT_TRUE((MyFloat("10.1") * MyFloat("-1")) == MyFloat("-10.1"));
}

TEST(FPTest, Max){
  EXPECT_TRUE(max(MyFloat("-1"), MyFloat("1")) == MyFloat("1"));
  EXPECT_TRUE(max(MyFloat("1"), MyFloat("1")) == MyFloat("1"));

  EXPECT_TRUE(max(MyFloat("-100"), MyFloat("-7")) == MyFloat("-7"));
  EXPECT_TRUE(max(MyFloat("100"), MyFloat("7")) == MyFloat("100"));
  EXPECT_TRUE(max(MyFloat("100"), MyFloat("-7")) == MyFloat("100"));

  EXPECT_TRUE(max(MyFloat("0.1"), MyFloat("-0.2")) == MyFloat("0.1"));
}

TEST(FPTest, Min) {
  EXPECT_TRUE(min(MyFloat("-1"), MyFloat("1")) == MyFloat("-1"));
  EXPECT_TRUE(min(MyFloat("1"), MyFloat("1")) == MyFloat("1"));

  EXPECT_TRUE(min(MyFloat("-100"), MyFloat("-7")) == MyFloat("-100"));
  EXPECT_TRUE(min(MyFloat("100"), MyFloat("7")) == MyFloat("7"));
  EXPECT_TRUE(min(MyFloat("100"), MyFloat("-7")) == MyFloat("-7"));

  EXPECT_TRUE(min(MyFloat("0.1"), MyFloat("-0.2")) == MyFloat("-0.2"));
}

