/* Copyright (c) 2020 The Brave Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at http://mozilla.org/MPL/2.0/. */

#include <cmath>
#include <fstream>
#include <vector>

#include "bat/ads/internal/ml/data/data.h"
#include "bat/ads/internal/ml/data/text_data.h"
#include "bat/ads/internal/ml/data/vector_data.h"
#include "bat/ads/internal/ml/model/linear/linear.h"
#include "bat/ads/internal/ml/pipeline/pipeline_info.h"
#include "bat/ads/internal/ml/pipeline/text_processing/text_processing.h"
#include "bat/ads/internal/ml/transformation/hashed_ngrams_transformation.h"
#include "bat/ads/internal/ml/transformation/lowercase_transformation.h"
#include "bat/ads/internal/ml/transformation/transformation.h"

#include "bat/ads/internal/unittest_base.h"
#include "bat/ads/internal/unittest_util.h"

// npm run test -- brave_unit_tests --filter=BatAds*

namespace ads {
namespace ml {

namespace {

const char kValidSegmentClassificationPipeline[] =
    "ml/pipeline/text_processing/valid_segment_classification_min.json";

const char kInvalidSpamClassificationPipeline[] =
    "ml/pipeline/text_processing/invalid_spam_classification.json";

const char kValidSpamClassificationPipeline[] =
    "ml/pipeline/text_processing/valid_spam_classification.json";

const char kTextCMCCrash[] = "ml/pipeline/text_processing/text_cmc_crash.txt";

}  // namespace

class BatAdsTextProcessingPipelineTest : public UnitTestBase {
 protected:
  BatAdsTextProcessingPipelineTest() = default;

  ~BatAdsTextProcessingPipelineTest() override = default;
};

TEST_F(BatAdsTextProcessingPipelineTest, BuildSimplePipeline) {
  // Arrange
  const double kTolerance = 1e-6;
  const unsigned kExpectedLen = 3;
  const std::string kTestString = "Test String";

  TransformationVector transformations;
  LowercaseTransformation lowercase;
  transformations.push_back(
      std::make_unique<LowercaseTransformation>(lowercase));
  HashedNGramsTransformation hashed_ngrams(3, std::vector<int>{1, 2, 3});
  transformations.push_back(
      std::make_unique<HashedNGramsTransformation>(hashed_ngrams));

  const std::map<std::string, VectorData> weights = {
      {"class_1", VectorData(std::vector<double>{1.0, 2.0, 3.0})},
      {"class_2", VectorData(std::vector<double>{3.0, 2.0, 1.0})},
      {"class_3", VectorData(std::vector<double>{2.0, 2.0, 2.0})}};

  const std::map<std::string, double> biases = {
      {"class_1", 0.0}, {"class_2", 0.0}, {"class_3", 0.0}};

  const model::Linear linear_model(weights, biases);
  const pipeline::TextProcessing pipeline =
      pipeline::TextProcessing(transformations, linear_model);

  const VectorData data_point_3(std::vector<double>{1.0, 0.0, 0.0});

  // Act
  const PredictionMap data_point_3_res = linear_model.Predict(data_point_3);
  const PredictionMap res = pipeline.GetTopPredictions(kTestString);

  // Assert
  ASSERT_EQ(kExpectedLen, data_point_3_res.size());
  ASSERT_TRUE(res.size() && res.size() <= kExpectedLen);
  for (const auto& pred : res) {
    EXPECT_TRUE(pred.second > -kTolerance && pred.second < 1.0 + kTolerance);
  }
}

TEST_F(BatAdsTextProcessingPipelineTest, TestLoadFromJson) {
  // Arrange
  const std::vector<std::string> train_texts = {
      "This is a spam email.", "Another spam trying to sell you viagra",
      "Message from mom with no real subject",
      "Another messase from mom with no real subject", "Yadayada"};
  const std::vector<std::string> train_labels = {"spam", "spam", "ham", "ham",
                                                 "junk"};

  const base::Optional<std::string> opt_value =
      ReadFileFromTestPathToString(kValidSpamClassificationPipeline);
  pipeline::TextProcessing pipeline;

  // Act
  ASSERT_TRUE(opt_value.has_value());
  const std::string json = opt_value.value();
  bool load_success = pipeline.FromJson(json);
  ASSERT_TRUE(load_success);

  std::vector<PredictionMap> prediction_maps(train_texts.size());
  for (size_t i = 0; i < train_texts.size(); i++) {
    const std::unique_ptr<Data> text_data =
        std::make_unique<TextData>(TextData(train_texts[i]));
    const PredictionMap prediction_map = pipeline.Apply(text_data);
    prediction_maps[i] = prediction_map;
  }

  // Assert
  for (size_t i = 0; i < prediction_maps.size(); i++) {
    const PredictionMap& prediction_map = prediction_maps[i];
    for (const auto& pred : prediction_map) {
      const double other_prediction = pred.second;
      EXPECT_TRUE(prediction_map.at(train_labels[i]) >= other_prediction);
    }
  }
}

TEST_F(BatAdsTextProcessingPipelineTest, InitValidModelTest) {
  // Arrange
  pipeline::TextProcessing text_proc_pipeline;
  const base::Optional<std::string> opt_value =
      ReadFileFromTestPathToString(kValidSegmentClassificationPipeline);

  // Act
  ASSERT_TRUE(opt_value.has_value());
  const std::string model_json = opt_value.value();
  bool loaded_successfully = text_proc_pipeline.FromJson(model_json);

  // Assert
  EXPECT_TRUE(loaded_successfully);
}

TEST_F(BatAdsTextProcessingPipelineTest, InvalidModelTest) {
  // Arrange
  pipeline::TextProcessing text_proc_pipeline;
  const base::Optional<std::string> opt_value =
      ReadFileFromTestPathToString(kInvalidSpamClassificationPipeline);

  // Act
  ASSERT_TRUE(opt_value.has_value());
  const std::string model_json = opt_value.value();
  bool loaded_successfully = text_proc_pipeline.FromJson(model_json);

  // Assert
  EXPECT_FALSE(loaded_successfully);
}

TEST_F(BatAdsTextProcessingPipelineTest, EmptyModelTest) {
  // Arrange
  pipeline::TextProcessing text_proc_pipeline;
  const std::string empty_model_json = "{}";

  // Act
  bool loaded_successfully = text_proc_pipeline.FromJson(empty_model_json);

  // Assert
  EXPECT_FALSE(loaded_successfully);
}

TEST_F(BatAdsTextProcessingPipelineTest, MissingModelTest) {
  // Arrange
  pipeline::TextProcessing text_proc_pipeline;

  // Act
  const std::string missing_model_json = "";
  bool loaded_successfully = text_proc_pipeline.FromJson(missing_model_json);

  // Assert
  EXPECT_FALSE(loaded_successfully);
}

TEST_F(BatAdsTextProcessingPipelineTest, TopPredUnitTest) {
  // Arrange
  const size_t kMaxPredictionsSize = 100;
  const std::string kTestPage = "ethereum bitcoin bat zcash crypto tokens!";
  pipeline::TextProcessing text_proc_pipeline;
  const base::Optional<std::string> opt_value =
      ReadFileFromTestPathToString(kValidSegmentClassificationPipeline);

  // Act
  ASSERT_TRUE(opt_value.has_value());
  const std::string model_json = opt_value.value();
  ASSERT_TRUE(text_proc_pipeline.FromJson(model_json));
  const PredictionMap preds = text_proc_pipeline.ClassifyPage(kTestPage);

  // Assert
  ASSERT_TRUE(preds.size());
  ASSERT_LT(preds.size(), kMaxPredictionsSize);
  ASSERT_TRUE(preds.count("crypto-crypto"));
  for (auto const& pred : preds) {
    EXPECT_TRUE(pred.second <= preds.at("crypto-crypto"));
  }
}

TEST_F(BatAdsTextProcessingPipelineTest, TextCMCCrashTest) {
  // Arrange
  const size_t kMinPredictionsSize = 2;
  const size_t kMaxPredictionsSize = 100;
  pipeline::TextProcessing text_proc_pipeline;

  const base::Optional<std::string> opt_value =
      ReadFileFromTestPathToString(kValidSegmentClassificationPipeline);
  ASSERT_TRUE(opt_value.has_value());

  const std::string model_json = opt_value.value();
  ASSERT_TRUE(text_proc_pipeline.FromJson(model_json));

  const base::Optional<std::string> opt_text_value =
      ReadFileFromTestPathToString(kTextCMCCrash);

  // Act
  ASSERT_TRUE(opt_text_value.has_value());
  const std::string bad_text = opt_text_value.value();
  const PredictionMap preds = text_proc_pipeline.ClassifyPage(bad_text);

  // Assert
  ASSERT_GT(preds.size(), kMinPredictionsSize);
  ASSERT_LT(preds.size(), kMaxPredictionsSize);
  ASSERT_TRUE(preds.count("personal finance-personal finance"));
  for (const auto& pred : preds) {
    EXPECT_TRUE(pred.second <= preds.at("personal finance-personal finance"));
  }
}

}  // namespace ml
}  // namespace ads
