/* Copyright (c) 2019 The Brave Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "brave/components/brave_shields/browser/ad_block_subscription_service_manager.h"

#include <memory>
#include <utility>
#include <vector>

#include "base/strings/string_util.h"
#include "base/task/post_task.h"
#include "base/values.h"
#include "brave/browser/brave_browser_process_impl.h"
#include "brave/common/pref_names.h"
#include "brave/components/brave_shields/browser/ad_block_subscription_service.h"
#include "brave/components/brave_shields/browser/ad_block_service.h"
#include "brave/components/brave_shields/browser/ad_block_service_helper.h"
#include "brave/vendor/adblock_rust_ffi/src/wrapper.h"
#include "components/prefs/pref_service.h"
#include "components/prefs/scoped_user_pref_update.h"
#include "content/public/browser/browser_task_traits.h"
#include "content/public/browser/browser_thread.h"

namespace brave_shields {

AdBlockSubscriptionServiceManager::AdBlockSubscriptionServiceManager(
    brave_component_updater::BraveComponent::Delegate* delegate)
    : delegate_(delegate),
      initialized_(false) {
}

AdBlockSubscriptionServiceManager::~AdBlockSubscriptionServiceManager() {
}

bool AdBlockSubscriptionServiceManager::Init() {
  DCHECK(!initialized_);
  base::PostTask(
      FROM_HERE, {content::BrowserThread::UI},
      base::BindOnce(&AdBlockSubscriptionServiceManager::StartSubscriptionServices,
                     base::Unretained(this)));
  return true;
}

void AdBlockSubscriptionServiceManager::CreateSubscription(const GURL list_url) {
  auto subscription_service = AdBlockSubscriptionServiceFactory(list_url, delegate_);
  subscription_service->Start();
  subscription_services_.insert(std::make_pair(list_url, std::move(subscription_service)));
}

std::vector<FilterListSubscriptionInfo> AdBlockSubscriptionServiceManager::GetSubscriptions() const {
  auto infos = std::vector<FilterListSubscriptionInfo>();

  for (const auto& subscription_service : subscription_services_) {
    infos.push_back(subscription_service.second->GetInfo());
  }

  return infos;
}

void AdBlockSubscriptionServiceManager::EnableSubscription(const SubscriptionIdentifier& id, bool enabled) {
  auto it = subscription_services_.find(id);
  DCHECK(it != subscription_services_.end());
  it->second->SetEnabled(enabled);
}

void AdBlockSubscriptionServiceManager::DeleteSubscription(const SubscriptionIdentifier& id) {
  auto it = subscription_services_.find(id);
  DCHECK(it != subscription_services_.end());
  it->second->Unregister();
  subscription_services_.erase(it);
}

void AdBlockSubscriptionServiceManager::RefreshSubscription(const SubscriptionIdentifier& id) {
  auto it = subscription_services_.find(id);
  DCHECK(it != subscription_services_.end());
  it->second->RefreshSubscription();
}

void AdBlockSubscriptionServiceManager::RefreshAllSubscriptions() {
  for (const auto& subscription_service : subscription_services_) {
    subscription_service.second->RefreshSubscription();
  }
}

void AdBlockSubscriptionServiceManager::StartSubscriptionServices() {
  DCHECK_CURRENTLY_ON(content::BrowserThread::UI);
  PrefService* local_state = g_browser_process->local_state();
  if (!local_state)
    return;

  /*if (regional_catalog_.size() == 0) {
    return;
  }

  // Enable the default regional list, but only do this once so that
  // user can override this setting in the future
  bool checked_default_region =
      local_state->GetBoolean(kAdBlockCheckedDefaultRegion);
  if (!checked_default_region) {
    local_state->SetBoolean(kAdBlockCheckedDefaultRegion, true);
    auto it = brave_shields::FindAdBlockFilterListByLocale(
        regional_catalog_, g_brave_browser_process->GetApplicationLocale());
    if (it == regional_catalog_.end())
      return;
    EnableFilterList(it->uuid, true);
  }

  // Start all regional services associated with enabled filter lists
  base::AutoLock lock(regional_services_lock_);
  const base::DictionaryValue* regional_filters_dict =
      local_state->GetDictionary(kAdBlockSubscriptionFilters);
  for (base::DictionaryValue::Iterator it(*regional_filters_dict);
       !it.IsAtEnd(); it.Advance()) {
    const std::string uuid = it.key();
    bool enabled = false;
    const base::DictionaryValue* regional_filter_dict = nullptr;
    regional_filters_dict->GetDictionary(uuid, &regional_filter_dict);
    if (regional_filter_dict)
      regional_filter_dict->GetBoolean("enabled", &enabled);
    if (enabled) {
      auto catalog_entry = brave_shields::FindAdBlockFilterListByUUID(
          regional_catalog_, uuid);
      if (catalog_entry != regional_catalog_.end()) {
        auto regional_service = AdBlockSubscriptionServiceFactory(
            *catalog_entry, delegate_);
        regional_service->Start();
        regional_services_.insert(
            std::make_pair(uuid, std::move(regional_service)));
      }
    }
  }*/

  initialized_ = true;
}

void AdBlockSubscriptionServiceManager::UpdateFilterListPrefs(
    const std::string& uuid,
    bool enabled) {
  DCHECK_CURRENTLY_ON(content::BrowserThread::UI);
  /*PrefService* local_state = g_browser_process->local_state();
  if (!local_state)
    return;
  DictionaryPrefUpdate update(local_state, kAdBlockSubscriptionFilters);
  base::DictionaryValue* regional_filters_dict = update.Get();
  auto regional_filter_dict = std::make_unique<base::DictionaryValue>();
  regional_filter_dict->SetBoolean("enabled", enabled);
  regional_filters_dict->Set(uuid, std::move(regional_filter_dict));*/
}

bool AdBlockSubscriptionServiceManager::IsInitialized() const {
  return initialized_;
}

bool AdBlockSubscriptionServiceManager::Start() {
  //base::AutoLock lock(regional_services_lock_);
  for (const auto& subscription_service : subscription_services_) {
    subscription_service.second->Start();
  }

  return true;
}

void AdBlockSubscriptionServiceManager::ShouldStartRequest(
    const GURL& url,
    blink::mojom::ResourceType resource_type,
    const std::string& tab_host,
    bool* did_match_rule,
    bool* did_match_exception,
    bool* did_match_important,
    std::string* mock_data_url) {
  //base::AutoLock lock(regional_services_lock_);
  for (const auto& subscription_service : subscription_services_) {
    if (subscription_service.second->GetInfo().enabled) {
      subscription_service.second->ShouldStartRequest(
          url, resource_type, tab_host, did_match_rule, did_match_exception,
          did_match_important, mock_data_url);
      if (did_match_important && *did_match_important) {
        return;
      }
    }
  }
}

void AdBlockSubscriptionServiceManager::EnableTag(const std::string& tag,
                                              bool enabled) {
  //base::AutoLock lock(regional_services_lock_);
  for (const auto& subscription_service : subscription_services_) {
    subscription_service.second->EnableTag(tag, enabled);
  }
}

void AdBlockSubscriptionServiceManager::AddResources(
    const std::string& resources) {
  //base::AutoLock lock(regional_services_lock_);
  for (const auto& subscription_service : subscription_services_) {
    subscription_service.second->AddResources(resources);
  }
}

void AdBlockSubscriptionServiceManager::EnableFilterList(
    const std::string& uuid, bool enabled) {
  DCHECK(!uuid.empty());
  /*auto catalog_entry = brave_shields::FindAdBlockFilterListByUUID(
      regional_catalog_, uuid);

  // Enable or disable the specified filter list
  if (initialized_) {
    base::AutoLock lock(regional_services_lock_);
    DCHECK(catalog_entry != regional_catalog_.end());
    auto it = regional_services_.find(uuid);
    if (enabled) {
      DCHECK(it == regional_services_.end());
      auto regional_service = AdBlockSubscriptionServiceFactory(
          *catalog_entry, delegate_);
      regional_service->Start();
      regional_services_.insert(
          std::make_pair(uuid, std::move(regional_service)));
    } else {
      DCHECK(it != regional_services_.end());
      it->second->Unregister();
      regional_services_.erase(it);
    }
  }

  // Update preferences to reflect enabled/disabled state of specified
  // filter list
  base::PostTask(
      FROM_HERE, {content::BrowserThread::UI},
      base::BindOnce(&AdBlockSubscriptionServiceManager::UpdateFilterListPrefs,
                     base::Unretained(this), uuid, enabled));*/
}

base::Optional<base::Value>
AdBlockSubscriptionServiceManager::UrlCosmeticResources(
        const std::string& url) {
  //base::AutoLock lock(regional_services_lock_);
  auto it = subscription_services_.begin();
  if (it == subscription_services_.end()) {
    return base::Optional<base::Value>();
  }
  base::Optional<base::Value> first_value =
      it->second->UrlCosmeticResources(url);

  for ( ; it != subscription_services_.end(); it++) {
    base::Optional<base::Value> next_value =
        it->second->UrlCosmeticResources(url);
    if (first_value) {
      if (next_value) {
        MergeResourcesInto(std::move(*next_value), &*first_value, false);
      }
    } else {
      first_value = std::move(next_value);
    }
  }

  return first_value;
}

base::Optional<base::Value>
AdBlockSubscriptionServiceManager::HiddenClassIdSelectors(
        const std::vector<std::string>& classes,
        const std::vector<std::string>& ids,
        const std::vector<std::string>& exceptions) {
  //base::AutoLock lock(regional_services_lock_);
  auto it = subscription_services_.begin();
  if (it == subscription_services_.end()) {
    return base::Optional<base::Value>();
  }
  base::Optional<base::Value> first_value =
      it->second->HiddenClassIdSelectors(classes, ids, exceptions);

  for ( ; it != subscription_services_.end(); it++) {
    base::Optional<base::Value> next_value =
        it->second->HiddenClassIdSelectors(classes, ids, exceptions);
    if (first_value && first_value->is_list()) {
      if (next_value && next_value->is_list()) {
        for (auto i = next_value->GetList().begin();
                i < next_value->GetList().end();
                i++) {
          first_value->Append(std::move(*i));
        }
      }
    } else {
      first_value = std::move(next_value);
    }
  }

  return first_value;
}

///////////////////////////////////////////////////////////////////////////////

std::unique_ptr<AdBlockSubscriptionServiceManager>
AdBlockSubscriptionServiceManagerFactory(BraveComponent::Delegate* delegate) {
  return std::make_unique<AdBlockSubscriptionServiceManager>(delegate);
}

}  // namespace brave_shields
