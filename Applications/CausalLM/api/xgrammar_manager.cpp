// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   xgrammar_manager.cpp
 * @date   14 April 2026
 * @brief  Implementation of XGrammarManager for grammar-guided generation
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "xgrammar_manager.h"
#include "xgrammar_wrapper.h"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <tokenizers_cpp.h>
#include <utility>

#include "json.hpp"

using json = nlohmann::json;

namespace causallm {
namespace {

constexpr int CACHE_FORMAT_VERSION = 2;
constexpr std::uint64_t FNV_OFFSET_BASIS = UINT64_C(14695981039346656037);
constexpr std::uint64_t FNV_PRIME = UINT64_C(1099511628211);
std::mutex toolset_cache_mutex;

void hashBytes(std::uint64_t &hash, const char *data, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    hash ^= static_cast<unsigned char>(data[i]);
    hash *= FNV_PRIME;
  }
}

void hashSize(std::uint64_t &hash, size_t size) {
  const auto fixed_size = static_cast<std::uint64_t>(size);
  for (size_t i = 0; i < sizeof(fixed_size); ++i) {
    const auto byte =
      static_cast<unsigned char>((fixed_size >> (i * 8)) & 0xffU);
    hash ^= byte;
    hash *= FNV_PRIME;
  }
}

std::string formatFingerprint(std::uint64_t hash) {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0') << std::setw(16) << hash;
  return stream.str();
}

std::string fingerprintText(const std::string &text) {
  std::uint64_t hash = FNV_OFFSET_BASIS;
  hashBytes(hash, text.data(), text.size());
  return formatFingerprint(hash);
}

std::string
fingerprintTokenizer(const std::vector<std::string> &encoded_vocab,
                     const std::string &canonical_tokenizer_metadata) {
  std::uint64_t hash = FNV_OFFSET_BASIS;
  hashSize(hash, encoded_vocab.size());
  for (const auto &token : encoded_vocab) {
    hashSize(hash, token.size());
    hashBytes(hash, token.data(), token.size());
  }
  hashSize(hash, canonical_tokenizer_metadata.size());
  hashBytes(hash, canonical_tokenizer_metadata.data(),
            canonical_tokenizer_metadata.size());
  return formatFingerprint(hash);
}

struct TokenizerMetadata {
  xgrammar::VocabType vocab_type;
  bool add_prefix_space;
  std::optional<std::vector<int32_t>> stop_token_ids;
};

struct GrammarContextDeleter {
  std::shared_ptr<xgrammar::TokenizerInfo> tokenizer_info;
  std::shared_ptr<xgrammar::GrammarCompiler> grammar_compiler;

  void operator()(XGrammar *grammar) const { delete grammar; }
};

XGrammarManager::GrammarPtr makeGrammarWithContext(
  const std::shared_ptr<xgrammar::TokenizerInfo> &tokenizer_info,
  const std::shared_ptr<xgrammar::GrammarCompiler> &grammar_compiler) {
  return XGrammarManager::GrammarPtr(
    new XGrammar(), GrammarContextDeleter{tokenizer_info, grammar_compiler});
}

bool parseTokenizerMetadata(const std::string &tokenizer_metadata,
                            unsigned int vocab_size,
                            TokenizerMetadata &parsed_metadata) {
  try {
    json metadata = json::parse(tokenizer_metadata);
    if (!metadata.is_object() || !metadata.contains("vocab_type") ||
        !metadata["vocab_type"].is_number_integer() ||
        !metadata.contains("add_prefix_space") ||
        !metadata["add_prefix_space"].is_boolean()) {
      std::cerr << "[XGrammarManager] Error: Invalid tokenizer metadata"
                << std::endl;
      return false;
    }

    const int vocab_type_value = metadata["vocab_type"].get<int>();
    if (vocab_type_value < static_cast<int>(xgrammar::VocabType::RAW) ||
        vocab_type_value > static_cast<int>(xgrammar::VocabType::BYTE_LEVEL)) {
      std::cerr << "[XGrammarManager] Error: Invalid tokenizer vocab type"
                << std::endl;
      return false;
    }

    if (metadata.contains("vocab_size")) {
      if (!metadata["vocab_size"].is_number_integer()) {
        std::cerr << "[XGrammarManager] Error: Invalid tokenizer metadata "
                     "vocabulary size"
                  << std::endl;
        return false;
      }
      const auto metadata_vocab_size = metadata["vocab_size"].get<int64_t>();
      if (metadata_vocab_size < 0 ||
          static_cast<uint64_t>(metadata_vocab_size) != vocab_size) {
        std::cerr << "[XGrammarManager] Error: Tokenizer metadata vocabulary "
                     "size does not match"
                  << std::endl;
        return false;
      }
    }

    std::optional<std::vector<int32_t>> stop_token_ids;
    if (metadata.contains("stop_token_ids") &&
        !metadata["stop_token_ids"].is_array()) {
      std::cerr << "[XGrammarManager] Error: Invalid tokenizer stop token IDs"
                << std::endl;
      return false;
    }
    if (metadata.contains("stop_token_ids")) {
      stop_token_ids.emplace();
      stop_token_ids->reserve(metadata["stop_token_ids"].size());
      for (const auto &token_id_json : metadata["stop_token_ids"]) {
        if (!token_id_json.is_number_integer()) {
          std::cerr
            << "[XGrammarManager] Error: Invalid tokenizer stop token ID"
            << std::endl;
          return false;
        }
        const auto token_id = token_id_json.get<int64_t>();
        if (token_id < 0 || static_cast<uint64_t>(token_id) >= vocab_size ||
            token_id > std::numeric_limits<int32_t>::max()) {
          std::cerr
            << "[XGrammarManager] Error: Tokenizer stop token ID is out "
               "of range"
            << std::endl;
          return false;
        }
        stop_token_ids->push_back(static_cast<int32_t>(token_id));
      }
    }

    parsed_metadata = {static_cast<xgrammar::VocabType>(vocab_type_value),
                       metadata["add_prefix_space"].get<bool>(),
                       std::move(stop_token_ids)};
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[XGrammarManager] Error: Failed to parse tokenizer metadata: "
              << e.what() << std::endl;
    return false;
  }
}

} // namespace

XGrammarManager::XGrammarManager() = default;

XGrammarManager::~XGrammarManager() = default;

XGrammarManager &XGrammarManager::Instance() {
  static XGrammarManager instance;
  return instance;
}

bool XGrammarManager::initialize(tokenizers::Tokenizer *tokenizer,
                                 unsigned int vocab_size) {
  const json legacy_metadata = {
    {"vocab_type", static_cast<int>(xgrammar::VocabType::BYTE_LEVEL)},
    {"add_prefix_space", false},
    {"vocab_size", vocab_size},
  };
  return initialize(tokenizer, vocab_size, legacy_metadata.dump());
}

bool XGrammarManager::initialize(tokenizers::Tokenizer *tokenizer,
                                 unsigned int vocab_size,
                                 const std::string &tokenizer_metadata) {
  std::lock_guard<std::mutex> lock(mutex_);
  try {
    return initializeUnlocked(tokenizer, vocab_size, tokenizer_metadata);
  } catch (const std::exception &e) {
    std::cerr << "[XGrammarManager] Error: Initialization failed: " << e.what()
              << std::endl;
    return false;
  } catch (...) {
    std::cerr << "[XGrammarManager] Error: Initialization failed" << std::endl;
    return false;
  }
}

bool XGrammarManager::initializeUnlocked(
  tokenizers::Tokenizer *tokenizer, unsigned int vocab_size,
  const std::string &tokenizer_metadata) {
  if (tokenizer == nullptr) {
    std::cerr << "[XGrammarManager] Error: Tokenizer is null" << std::endl;
    return false;
  }
  if (vocab_size == 0 || vocab_size > static_cast<unsigned int>(
                                        std::numeric_limits<int32_t>::max())) {
    std::cerr << "[XGrammarManager] Error: Vocabulary size is invalid"
              << std::endl;
    return false;
  }

  TokenizerMetadata parsed_metadata;
  if (!parseTokenizerMetadata(tokenizer_metadata, vocab_size,
                              parsed_metadata)) {
    return false;
  }

  std::cout << "[XGrammarManager] Extracting vocabulary from tokenizer..."
            << std::endl;
  std::vector<std::string> encoded_vocab;
  encoded_vocab.reserve(vocab_size);
  for (size_t i = 0; i < vocab_size; ++i) {
    encoded_vocab.push_back(tokenizer->IdToToken(static_cast<int32_t>(i)));
  }
  std::cout << "[XGrammarManager] Vocabulary size: " << vocab_size << std::endl;

  // Build replacement state before invalidating current grammars. Failed
  // tokenizer or xgrammar construction therefore leaves this manager usable.
  std::cout << "[XGrammarManager] Creating TokenizerInfo..." << std::endl;
  std::shared_ptr<xgrammar::TokenizerInfo> tokenizer_info;
  try {
    tokenizer_info = std::make_shared<xgrammar::TokenizerInfo>(
      encoded_vocab, parsed_metadata.vocab_type, encoded_vocab.size(),
      std::move(parsed_metadata.stop_token_ids),
      parsed_metadata.add_prefix_space);
  } catch (const std::exception &e) {
    std::cerr << "[XGrammarManager] Error: Failed to create TokenizerInfo: "
              << e.what() << std::endl;
    return false;
  }

  std::cout << "[XGrammarManager] Creating GrammarCompiler..." << std::endl;
  std::shared_ptr<xgrammar::GrammarCompiler> grammar_compiler;
  std::string canonical_metadata;
  try {
    grammar_compiler =
      std::make_shared<xgrammar::GrammarCompiler>(*tokenizer_info);
    canonical_metadata = tokenizer_info->DumpMetadata();
  } catch (const std::exception &e) {
    std::cerr << "[XGrammarManager] Error: Failed to create GrammarCompiler: "
              << e.what() << std::endl;
    return false;
  } catch (...) {
    std::cerr << "[XGrammarManager] Error: Failed to create GrammarCompiler"
              << std::endl;
    return false;
  }

  clearUnlocked();
  tokenizer_info_ = std::move(tokenizer_info);
  grammar_compiler_ = std::move(grammar_compiler);
  tokenizer_ = tokenizer;
  vocab_size_ = vocab_size;
  tokenizer_fingerprint_ =
    fingerprintTokenizer(encoded_vocab, canonical_metadata);
  tokenizer_metadata_fingerprint_ = fingerprintText(canonical_metadata);
  initialized_ = true;
  return true;
}

bool XGrammarManager::loadToolset(const std::string &toolset_path,
                                  tokenizers::Tokenizer *tokenizer,
                                  unsigned int vocab_size) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (tokenizer == nullptr) {
    std::cerr << "[XGrammarManager] Error: Tokenizer is null" << std::endl;
    return false;
  }
  if (vocab_size == 0) {
    std::cerr << "[XGrammarManager] Error: Vocabulary is empty" << std::endl;
    return false;
  }
  if (!initialized_ || tokenizer_ != tokenizer || vocab_size_ != vocab_size) {
    std::cerr << "[XGrammarManager] Error: Manager is not initialized for "
                 "this tokenizer. Call initialize() with tokenizer metadata "
                 "first."
              << std::endl;
    return false;
  }

  std::cout << "[XGrammarManager] Loading toolset from: " << toolset_path
            << std::endl;

  // Load the source first. A cache is valid only when generated from this
  // exact canonical toolset and tokenizer vocabulary.
  std::ifstream file(toolset_path);
  if (!file.is_open()) {
    std::cerr << "[XGrammarManager] Error: Failed to open toolset file: "
              << toolset_path << std::endl;
    return false;
  }

  json toolset;
  try {
    file >> toolset;
  } catch (const json::exception &e) {
    std::cerr << "[XGrammarManager] Error: Failed to parse toolset JSON: "
              << e.what() << std::endl;
    return false;
  }
  if (!toolset.is_object()) {
    std::cerr << "[XGrammarManager] Error: Toolset must be a JSON object"
              << std::endl;
    return false;
  }

  const std::string toolset_fingerprint = fingerprintText(toolset.dump());
  const std::string cache_path = toolset_path + ".cache";
  // Different handles have different manager mutexes but may share a toolset
  // path. Serialize cache read/compile/write so one handle cannot observe a
  // partially written cache from another.
  std::lock_guard<std::mutex> cache_lock(toolset_cache_mutex);

  std::ifstream cache_file(cache_path);
  if (cache_file.is_open()) {
    std::cout << "[XGrammarManager] Found cache file: " << cache_path
              << std::endl;

    json cache_data;
    try {
      cache_file >> cache_data;
      cache_file.close();

      const bool cache_matches =
        cache_data.is_object() &&
        cache_data.value("format_version", 0) == CACHE_FORMAT_VERSION &&
        cache_data.value("tokenizer_fingerprint", std::string()) ==
          tokenizer_fingerprint_ &&
        cache_data.value("tokenizer_metadata_fingerprint", std::string()) ==
          tokenizer_metadata_fingerprint_ &&
        cache_data.value("toolset_fingerprint", std::string()) ==
          toolset_fingerprint &&
        cache_data.contains("grammars") && cache_data["grammars"].is_object() &&
        cache_data["grammars"].size() == toolset.size();

      std::unordered_map<std::string, GrammarPtr> cached_grammars;
      bool all_loaded = cache_matches;
      if (cache_matches) {
        const json &serialized_grammars = cache_data["grammars"];
        for (auto it = toolset.begin(); it != toolset.end(); ++it) {
          const std::string &tool_name = it.key();
          auto cached = serialized_grammars.find(tool_name);
          if (cached == serialized_grammars.end() || !cached->is_string()) {
            all_loaded = false;
            break;
          }

          auto grammar =
            makeGrammarWithContext(tokenizer_info_, grammar_compiler_);
          if (!grammar->loadFromCache(cached->get<std::string>(),
                                      tokenizer_info_.get(), vocab_size)) {
            std::cerr << "[XGrammarManager] Warning: Failed to load cached "
                         "grammar for: "
                      << tool_name << std::endl;
            all_loaded = false;
            break;
          }
          cached_grammars[tool_name] = std::move(grammar);
        }
      }

      if (all_loaded) {
        compiled_grammars_ = std::move(cached_grammars);
        current_toolset_path_ = toolset_path;
        std::cout << "[XGrammarManager] Successfully loaded "
                  << compiled_grammars_.size() << " tool grammars from cache"
                  << std::endl;
        return true;
      }
      std::cout << "[XGrammarManager] Cache is stale or incomplete; "
                   "falling back to compilation..."
                << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "[XGrammarManager] Warning: Failed to load cache file: "
                << e.what() << std::endl;
      cache_file.close();
    }
  }

  std::cout << "[XGrammarManager] Pre-compiling all tool grammars..."
            << std::endl;
  json serialized_grammars = json::object();
  std::unordered_map<std::string, GrammarPtr> new_grammars;

  try {
    for (auto it = toolset.begin(); it != toolset.end(); ++it) {
      const std::string &tool_name = it.key();
      const std::string json_schema = it.value().dump();
      std::cout << "[XGrammarManager] Compiling grammar for tool: " << tool_name
                << std::endl;

      auto grammar = makeGrammarWithContext(tokenizer_info_, grammar_compiler_);
      grammar->initializeGrammar("json", json_schema, grammar_compiler_.get(),
                                 vocab_size);
      serialized_grammars[tool_name] = grammar->serialize();
      new_grammars[tool_name] = std::move(grammar);
    }
  } catch (const std::exception &e) {
    std::cerr << "[XGrammarManager] Error: Failed to compile toolset: "
              << e.what() << std::endl;
    return false;
  }

  compiled_grammars_ = std::move(new_grammars);
  const json cache_data = {
    {"format_version", CACHE_FORMAT_VERSION},
    {"tokenizer_fingerprint", tokenizer_fingerprint_},
    {"tokenizer_metadata_fingerprint", tokenizer_metadata_fingerprint_},
    {"toolset_fingerprint", toolset_fingerprint},
    {"grammars", std::move(serialized_grammars)},
  };

  std::ofstream out_cache(cache_path);
  if (out_cache.is_open()) {
    out_cache << cache_data.dump();
    out_cache.close();
    std::cout << "[XGrammarManager] Saved grammar cache to: " << cache_path
              << std::endl;
  } else {
    std::cerr << "[XGrammarManager] Warning: Failed to save cache file to: "
              << cache_path << std::endl;
  }

  current_toolset_path_ = toolset_path;
  std::cout << "[XGrammarManager] Successfully compiled "
            << compiled_grammars_.size() << " tool grammars" << std::endl;
  return true;
}

XGrammarManager::GrammarPtr
XGrammarManager::getGrammarShared(const std::string &tool_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = compiled_grammars_.find(tool_name);
  if (it != compiled_grammars_.end())
    return it->second;

  std::cerr << "[XGrammarManager] Warning: Tool not found: " << tool_name
            << std::endl;
  return {};
}

XGrammar *XGrammarManager::getGrammar(const std::string &tool_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = compiled_grammars_.find(tool_name);
  if (it != compiled_grammars_.end())
    return it->second.get();

  std::cerr << "[XGrammarManager] Warning: Tool not found: " << tool_name
            << std::endl;
  return nullptr;
}

void XGrammarManager::resetGrammar(const std::string &tool_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = compiled_grammars_.find(tool_name);
  if (it != compiled_grammars_.end())
    it->second->resetGrammar();
}

bool XGrammarManager::registerTool(const std::string &tool_name,
                                   const std::string &json_schema) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!tokenizer_info_ || !grammar_compiler_) {
    std::cerr << "[XGrammarManager] Error: Manager not initialized. Call "
                 "initialize() first."
              << std::endl;
    return false;
  }
  if (tool_name.empty()) {
    std::cerr << "[XGrammarManager] Error: Tool name must not be empty"
              << std::endl;
    return false;
  }

  const unsigned int vocab_size = tokenizer_info_->GetVocabSize();
  std::cout << "[XGrammarManager] Registering tool: " << tool_name << std::endl;

  GrammarPtr grammar;
  try {
    grammar = makeGrammarWithContext(tokenizer_info_, grammar_compiler_);
    grammar->initializeGrammar("json", json_schema, grammar_compiler_.get(),
                               vocab_size);
  } catch (const std::exception &e) {
    std::cerr << "[XGrammarManager] Error: Failed to compile tool '"
              << tool_name << "': " << e.what() << std::endl;
    return false;
  } catch (...) {
    std::cerr << "[XGrammarManager] Error: Failed to compile tool '"
              << tool_name << "'" << std::endl;
    return false;
  }

  compiled_grammars_[tool_name] = std::move(grammar);
  std::cout << "[XGrammarManager] Successfully registered tool: " << tool_name
            << std::endl;
  return true;
}

void XGrammarManager::unregisterTool(const std::string &tool_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  compiled_grammars_.erase(tool_name);
}

bool XGrammarManager::hasTool(const std::string &tool_name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return compiled_grammars_.find(tool_name) != compiled_grammars_.end();
}

std::vector<std::string> XGrammarManager::getToolNames() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> names;
  names.reserve(compiled_grammars_.size());
  for (const auto &pair : compiled_grammars_)
    names.push_back(pair.first);
  return names;
}

bool XGrammarManager::isInitialized() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return initialized_;
}

void XGrammarManager::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  clearUnlocked();
  std::cout << "[XGrammarManager] Cleared all compiled grammars" << std::endl;
}

void XGrammarManager::clearUnlocked() {
  compiled_grammars_.clear();
  tokenizer_info_.reset();
  grammar_compiler_.reset();
  current_toolset_path_.clear();
  tokenizer_ = nullptr;
  vocab_size_ = 0;
  tokenizer_fingerprint_.clear();
  tokenizer_metadata_fingerprint_.clear();
  initialized_ = false;
}

} // namespace causallm
