/* Copyright (c) 2020 The Brave Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at http://mozilla.org/MPL/2.0/. */

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>

#include "base/notreached.h"
#include "bat/ads/internal/ml/ml_util.h"

namespace ads {
namespace ml {

PredictionMap Softmax(const PredictionMap& predictions) {
  double maximum = -std::numeric_limits<double>::infinity();
  for (const auto& prediction : predictions) {
    maximum = std::max(maximum, prediction.second);
  }
  std::map<std::string, double> softmax_predictions;
  double sum_exp = 0.0;
  for (const auto& prediction : predictions) {
    const double val = std::exp(prediction.second - maximum);
    softmax_predictions[prediction.first] = val;
    sum_exp += val;
  }
  for (auto& prediction : softmax_predictions) {
    prediction.second /= sum_exp;
  }
  return softmax_predictions;
}

TransformationPtr GetTransformationCopy(
    const TransformationPtr& transformation_ptr) {
  if (transformation_ptr->GetType() ==
      transformation::TransformationType::LOWERCASE) {
    transformation::Lowercase* lowercase_ptr =
        static_cast<transformation::Lowercase*>(transformation_ptr.get());
    transformation::Lowercase lowercase_copy = *lowercase_ptr;
    return std::make_unique<transformation::Lowercase>(lowercase_copy);
  }

  if (transformation_ptr->GetType() ==
      transformation::TransformationType::HASHED_NGRAMS) {
    transformation::HashedNGrams* hashed_n_grams_ptr =
        static_cast<transformation::HashedNGrams*>(transformation_ptr.get());
    transformation::HashedNGrams hashed_n_grams_ptr_copy = *hashed_n_grams_ptr;
    return std::make_unique<transformation::HashedNGrams>(
        hashed_n_grams_ptr_copy);
  }

  if (transformation_ptr->GetType() ==
      transformation::TransformationType::NORMALIZATION) {
    transformation::Normalization* normalization_ptr =
        static_cast<transformation::Normalization*>(transformation_ptr.get());
    transformation::Normalization normalization_copy = *normalization_ptr;
    return std::make_unique<transformation::Normalization>(normalization_copy);
  }

  NOTREACHED();
  return TransformationPtr(nullptr);
}

TransformationVector GetTransformationVectorCopy(
    const TransformationVector& transformation_vector) {
  TransformationVector transformation_vector_copy;
  const size_t transformation_count = transformation_vector.size();
  for (size_t i = 0; i < transformation_count; ++i) {
    transformation_vector_copy.push_back(
        GetTransformationCopy(transformation_vector[i]));
  }
  return transformation_vector_copy;
}

}  // namespace ml
}  // namespace ads