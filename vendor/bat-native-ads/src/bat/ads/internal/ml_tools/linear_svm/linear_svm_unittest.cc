/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include <cmath>

#include "bat/ads/internal/ml_tools/data_point/data_point.h"
#include "bat/ads/internal/ml_tools/linear_svm/linear_svm.h"

#include "bat/ads/internal/json_helper.h"
#include "bat/ads/internal/unittest_base.h"
#include "bat/ads/internal/unittest_util.h"

// npm run test -- brave_unit_tests --filter=BatAds*

namespace ads {
namespace ml_tools {

class BatAdsLinearSVMTest : public UnitTestBase {
 protected:
  BatAdsLinearSVMTest() = default;

  ~BatAdsLinearSVMTest() override = default;
};

TEST_F(BatAdsLinearSVMTest, ThreeClassesPredictionTest) {
  std::map<std::string, DataPoint> weights = {
    {"class_1", std::vector<double>{1.0, 0.0, 0.0}},
    {"class_2", std::vector<double>{0.0, 1.0, 0.0}},
    {"class_3", std::vector<double>{0.0, 0.0, 1.0}}
  };

  std::map<std::string,double> biases = {
    {"class_1", 0.0},
    {"class_2", 0.0},
    {"class_3", 0.0}
  };

  linear_svm::LinearSVM linear_svm(weights, biases);
  
  auto class1_data_point = DataPoint(std::vector<double>{1.0, 0.0, 0.0});
  auto res1 = linear_svm.Predict(class1_data_point);
  EXPECT_TRUE(res1["class_1"] > res1["class_2"]);
  EXPECT_TRUE(res1["class_1"] > res1["class_3"]);

  auto class2_data_point = DataPoint(std::vector<double>{0.0, 1.0, 0.0});
  auto res2 = linear_svm.Predict(class2_data_point);
  EXPECT_TRUE(res2["class_2"] > res2["class_1"]);
  EXPECT_TRUE(res2["class_2"] > res2["class_3"]);

  auto class3_data_point = DataPoint(std::vector<double>{0.0, 1.0, 2.0});
  auto res3 = linear_svm.Predict(class3_data_point);
  EXPECT_TRUE(res3["class_3"] > res3["class_1"]);
  EXPECT_TRUE(res3["class_3"] > res3["class_2"]);
}

TEST_F(BatAdsLinearSVMTest, BiasesPredictionTest) {
  std::map<std::string, DataPoint> weights = {
    {"class_1", std::vector<double>{1.0, 0.0, 0.0}},
    {"class_2", std::vector<double>{0.0, 1.0, 0.0}},
    {"class_3", std::vector<double>{0.0, 0.0, 1.0}}
  };

  std::map<std::string,double> biases = {
    {"class_1", 0.5},
    {"class_2", 0.25},
    {"class_3", 1.0}
  };

  linear_svm::LinearSVM biased_svm(weights,biases);

  auto avg_point = DataPoint(std::vector<double>{1.0, 1.0, 1.0});
  auto res = biased_svm.Predict(avg_point);
  EXPECT_TRUE(res["class_3"] > res["class_1"]);
  EXPECT_TRUE(res["class_3"] > res["class_2"]);
  EXPECT_TRUE(res["class_1"] > res["class_2"]);
}

TEST_F(BatAdsLinearSVMTest, SoftmaxTest) {
  const double kEps = 1e-8;

  std::map<std::string, double> group_1 = {
    {"c1", -1.0},
    {"c2", 2.0},
    {"c3", 3.0}
  };

  linear_svm::LinearSVM dummy_svm;
  auto sm = dummy_svm.Softmax(group_1);

  EXPECT_TRUE(sm["c3"] > sm["c1"]);
  EXPECT_TRUE(sm["c3"] > sm["c2"]);
  EXPECT_TRUE(sm["c2"] > sm["c1"]);
  EXPECT_TRUE(sm["c1"] > 0.0);
  EXPECT_TRUE(sm["c3"] < 1.0);

  double sum = 0.0;
  for (auto const& x : sm){
    sum += x.second;
  }

  EXPECT_TRUE(sum - 1.0 < kEps);
}

TEST_F(BatAdsLinearSVMTest, ExtendedSoftmaxTest) {
  const double kEps = 1e-8;

  std::map<std::string,double> group_1 = {
    {"c1", 0.0},
    {"c2", 1.0},
    {"c3", 2.0}
  };

  std::map<std::string,double> group_2 = {
    {"c1", 3.0},
    {"c2", 4.0},
    {"c3", 5.0}
  };

  linear_svm::LinearSVM dummy_svm;

  auto sm_1 = dummy_svm.Softmax(group_1);
  auto sm_2 = dummy_svm.Softmax(group_2);
  EXPECT_TRUE(std::fabs(sm_1["c1"] - sm_2["c1"]) < kEps);
  EXPECT_TRUE(std::fabs(sm_1["c2"] - sm_2["c2"]) < kEps);
  EXPECT_TRUE(std::fabs(sm_1["c3"] - sm_2["c3"]) < kEps);
  EXPECT_TRUE(std::fabs(sm_1["c1"] - 0.09003057) < kEps);
  EXPECT_TRUE(std::fabs(sm_1["c2"] - 0.24472847) < kEps);
  EXPECT_TRUE(std::fabs(sm_1["c3"] - 0.66524095) < kEps);
}

}  // namespace ml_tools
}  // namespace ads