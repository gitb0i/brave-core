/* Copyright (c) 2020 The Brave Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at http://mozilla.org/MPL/2.0/. */

#include <cmath>
#include <map>
#include <memory>
#include <string>

#include "bat/ads/internal/ml/ml_util.h"

#include "bat/ads/internal/unittest_base.h"
#include "bat/ads/internal/unittest_util.h"

// npm run test -- brave_unit_tests --filter=BatAds*

namespace ads {
namespace ml {

class BatAdsMLToolsUtilTest : public UnitTestBase {
 protected:
  BatAdsMLToolsUtilTest() = default;

  ~BatAdsMLToolsUtilTest() override = default;
};

TEST_F(BatAdsMLToolsUtilTest, SoftmaxTest) {
  // Arrange
  const double kTolerance = 1e-8;

  const std::map<std::string, double> group_1 = {
      {"c1", -1.0}, {"c2", 2.0}, {"c3", 3.0}};

  // Act
  const PredictionMap predictions = Softmax(group_1);

  double sum = 0.0;
  for (auto const& prediction : predictions) {
    sum += prediction.second;
  }

  // Assert
  ASSERT_GT(predictions.at("c3"), predictions.at("c1"));
  ASSERT_GT(predictions.at("c3"), predictions.at("c2"));
  ASSERT_GT(predictions.at("c2"), predictions.at("c1"));
  ASSERT_GT(predictions.at("c1"), 0.0);
  ASSERT_LT(predictions.at("c3"), 1.0);
  EXPECT_LT(sum - 1.0, kTolerance);
}

TEST_F(BatAdsMLToolsUtilTest, ExtendedSoftmaxTest) {
  // Arrange
  const double kTolerance = 1e-8;

  const std::map<std::string, double> group_1 = {
      {"c1", 0.0}, {"c2", 1.0}, {"c3", 2.0}};

  const std::map<std::string, double> group_2 = {
      {"c1", 3.0}, {"c2", 4.0}, {"c3", 5.0}};

  // Act
  const PredictionMap predictions_1 = Softmax(group_1);
  const PredictionMap predictions_2 = Softmax(group_2);

  // Assert
  ASSERT_LT(std::fabs(predictions_1.at("c1") - predictions_2.at("c1")),
            kTolerance);
  ASSERT_LT(std::fabs(predictions_1.at("c2") - predictions_2.at("c2")),
            kTolerance);
  ASSERT_LT(std::fabs(predictions_1.at("c3") - predictions_2.at("c3")),
            kTolerance);

  EXPECT_TRUE(std::fabs(predictions_1.at("c1") - 0.09003057) < kTolerance &&
              std::fabs(predictions_1.at("c2") - 0.24472847) < kTolerance &&
              std::fabs(predictions_1.at("c3") - 0.66524095) < kTolerance);
}

TEST_F(BatAdsMLToolsUtilTest, TransformationCopyTest) {
  // Arrange
  const NormalizationTransformation normalization;
  TransformationPtr transformation_ptr =
      std::make_unique<NormalizationTransformation>(normalization);

  // Act
  const TransformationPtr transformation_ptr_copy =
      GetTransformationCopy(transformation_ptr);

  // Assert
  EXPECT_EQ(transformation_ptr_copy->GetType(),
            TransformationType::NORMALIZATION);
}

TEST_F(BatAdsMLToolsUtilTest, TransformationVectorCopyTest) {
  // Arrange
  const size_t kVectorSize = 2;

  TransformationVector transformation_vector;
  const HashedNGramsTransformation hashed_ngrams;
  transformation_vector.push_back(
      std::make_unique<HashedNGramsTransformation>(hashed_ngrams));

  const NormalizationTransformation normalization;
  transformation_vector.push_back(
      std::make_unique<NormalizationTransformation>(normalization));

  // Act
  const TransformationVector transformation_vector_copy =
      GetTransformationVectorCopy(transformation_vector);

  // Assert
  ASSERT_EQ(kVectorSize, transformation_vector_copy.size());
  EXPECT_TRUE(transformation_vector_copy[0]->GetType() ==
                  TransformationType::HASHED_NGRAMS &&
              transformation_vector_copy[1]->GetType() ==
                  TransformationType::NORMALIZATION);
}

}  // namespace ml
}  // namespace ads
