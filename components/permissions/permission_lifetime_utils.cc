/* Copyright (c) 2021 The Brave Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "brave/components/permissions/permission_lifetime_utils.h"

#include <utility>

#include "base/i18n/time_formatting.h"
#include "base/stl_util.h"
#include "base/strings/stringprintf.h"
#include "base/strings/utf_string_conversions.h"
#include "components/grit/brave_components_strings.h"
#include "components/permissions/features.h"
#include "ui/base/l10n/l10n_util.h"

namespace permissions {

namespace {

base::FeatureParam<int> kPermissionLifetimeTestSecondsParam{
    &features::kPermissionLifetime, "test_seconds", 0};

}  // namespace

PermissionLifetimeOption::PermissionLifetimeOption(
    base::string16 label,
    base::Optional<base::TimeDelta> lifetime)
    : label(std::move(label)), lifetime(std::move(lifetime)) {}

PermissionLifetimeOption::PermissionLifetimeOption(
    const PermissionLifetimeOption&) = default;
PermissionLifetimeOption& PermissionLifetimeOption::operator=(
    const PermissionLifetimeOption&) = default;
PermissionLifetimeOption::PermissionLifetimeOption(
    PermissionLifetimeOption&&) noexcept = default;
PermissionLifetimeOption& PermissionLifetimeOption::operator=(
    PermissionLifetimeOption&&) noexcept = default;
PermissionLifetimeOption::~PermissionLifetimeOption() = default;

std::vector<PermissionLifetimeOption> CreatePermissionLifetimeOptions() {
  std::vector<PermissionLifetimeOption> options;
  const size_t kOptionsCount = 4;
  options.reserve(kOptionsCount);

  options.emplace_back(PermissionLifetimeOption(
      l10n_util::GetStringUTF16(
          IDS_PERMISSIONS_BUBBLE_UNTIL_PAGE_CLOSE_LIFETIME_OPTION),
      base::TimeDelta()));
  options.emplace_back(PermissionLifetimeOption(
      l10n_util::GetStringUTF16(
          IDS_PERMISSIONS_BUBBLE_24_HOURS_LIFETIME_OPTION),
      base::TimeDelta::FromHours(24)));
  options.emplace_back(PermissionLifetimeOption(
      l10n_util::GetStringUTF16(IDS_PERMISSIONS_BUBBLE_1_WEEK_LIFETIME_OPTION),
      base::TimeDelta::FromDays(7)));
  options.emplace_back(PermissionLifetimeOption(
      l10n_util::GetStringUTF16(IDS_PERMISSIONS_BUBBLE_FOREVER_LIFETIME_OPTION),
      base::nullopt));
  DCHECK_EQ(options.size(), kOptionsCount);

  // This is strictly for manual testing.
  const int test_seconds = kPermissionLifetimeTestSecondsParam.Get();
  if (test_seconds) {
    options.insert(
        options.begin(),
        PermissionLifetimeOption(
            base::UTF8ToUTF16(base::StringPrintf("%d seconds", test_seconds)),
            base::TimeDelta::FromSeconds(test_seconds)));
  }

  return options;
}

}  // namespace permissions
