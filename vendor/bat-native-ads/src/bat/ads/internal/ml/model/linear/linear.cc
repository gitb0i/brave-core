/* Copyright (c) 2020 The Brave Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at http://mozilla.org/MPL/2.0/. */

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "bat/ads/internal/ml/data/vector_data.h"
#include "bat/ads/internal/ml/ml_prediction_util.h"
#include "bat/ads/internal/ml/model/linear/linear.h"

namespace ads {
namespace ml {
namespace model {

Linear::Linear() {}

Linear::Linear(const std::map<std::string, VectorData>& weights,
               const std::map<std::string, double>& biases) {
  weights_ = weights;
  biases_ = biases;
}

Linear::Linear(const Linear& linear_model) = default;

Linear::~Linear() = default;

PredictionMap Linear::Predict(const VectorData& x) const {
  PredictionMap predictions;
  for (const auto& kv : weights_) {
    double prediction = kv.second * x;
    const auto iter = biases_.find(kv.first);
    if (iter != biases_.end()) {
      prediction += iter->second;
    }
    predictions[kv.first] = prediction;
  }
  return predictions;
}

PredictionMap Linear::GetTopPredictions(const VectorData& x,
                                        const int top_count) const {
  PredictionMap prediction_map = Predict(x);
  PredictionMap prediction_map_softmax = Softmax(prediction_map);
  std::vector<std::pair<double, std::string>> prediction_order;
  prediction_order.reserve(prediction_map_softmax.size());
  for (auto iter = prediction_map_softmax.begin();
       iter != prediction_map_softmax.end(); iter++) {
    prediction_order.push_back(std::make_pair(iter->second, iter->first));
  }
  std::sort(prediction_order.rbegin(), prediction_order.rend());
  PredictionMap top_predictions;
  if (top_count > 0) {
    prediction_order.resize(top_count);
  }
  for (size_t i = 0; i < prediction_order.size(); ++i) {
    top_predictions[prediction_order[i].second] = prediction_order[i].first;
  }
  return top_predictions;
}

}  // namespace model
}  // namespace ml
}  // namespace ads
