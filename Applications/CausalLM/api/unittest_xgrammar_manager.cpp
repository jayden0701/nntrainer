// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_xgrammar_manager.cpp
 * @brief  Unit tests for XGrammarManager lifecycle and cache validation
 * @author Quick.AI Team
 * @bug    No known bugs except for NYI items
 */

#include "xgrammar_manager.h"
#include "xgrammar_wrapper.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <tokenizers_cpp.h>

#include "json.hpp"

namespace {

using json = nlohmann::json;

class FakeTokenizer final : public tokenizers::Tokenizer {
public:
  explicit FakeTokenizer(std::vector<std::string> vocabulary) :
    vocabulary_(std::move(vocabulary)) {}

  std::vector<int32_t> Encode(const std::string &) override { return {}; }

  std::vector<int32_t> Encode(const std::string &, bool) override { return {}; }

  std::string Decode(const std::vector<int32_t> &) override { return {}; }

  size_t GetVocabSize() override { return vocabulary_.size(); }

  std::string IdToToken(int32_t token_id) override {
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocabulary_.size())
      return {};
    return vocabulary_[static_cast<size_t>(token_id)];
  }

  int32_t TokenToId(const std::string &token) override {
    for (size_t i = 0; i < vocabulary_.size(); ++i) {
      if (vocabulary_[i] == token)
        return static_cast<int32_t>(i);
    }
    return -1;
  }

private:
  std::vector<std::string> vocabulary_;
};

std::vector<std::string> makeVocabulary() {
  std::vector<std::string> vocabulary;
  for (char token = ' '; token <= '~'; ++token)
    vocabulary.emplace_back(1, token);
  vocabulary.emplace_back("\n");
  vocabulary.emplace_back("\t");
  return vocabulary;
}

class ScopedToolset {
public:
  ScopedToolset() {
    const auto *test_info =
      ::testing::UnitTest::GetInstance()->current_test_info();
    path_ = ::testing::TempDir() + "nntrainer_" + test_info->test_suite_name() +
            "_" + test_info->name() + ".json";
    std::remove(path_.c_str());
    std::remove(cachePath().c_str());
  }

  ~ScopedToolset() {
    std::remove(path_.c_str());
    std::remove(cachePath().c_str());
  }

  const std::string &path() const { return path_; }

  std::string cachePath() const { return path_ + ".cache"; }

  void write(const json &toolset) const {
    std::ofstream output(path_, std::ios::trunc);
    ASSERT_TRUE(output.is_open());
    output << toolset.dump();
    ASSERT_TRUE(output.good());
  }

  json readCache() const {
    std::ifstream input(cachePath());
    EXPECT_TRUE(input.is_open());
    json cache;
    input >> cache;
    return cache;
  }

private:
  std::string path_;
};

json makeToolset(const std::string &property_name) {
  return {
    {"test_tool",
     {{"type", "object"},
      {"properties", {{property_name, {{"type", "integer"}}}}},
      {"required", {property_name}},
      {"additionalProperties", false}}},
  };
}

std::string makeTokenizerMetadata(bool add_prefix_space = false) {
  return json({{"vocab_type", 0}, {"add_prefix_space", add_prefix_space}})
    .dump();
}

TEST(XGrammarWrapper, CompilesCallerProvidedEbnfAndRegexPayloads) {
  FakeTokenizer tokenizer(makeVocabulary());
  causallm::XGrammar ebnf;
  causallm::XGrammar regex;

  EXPECT_NO_THROW(ebnf.initializeGrammar("ebnf",
                                         R"(root ::= "caller supplied ebnf")",
                                         &tokenizer, tokenizer.GetVocabSize()));
  EXPECT_TRUE(ebnf.isGrammarEnabled());

  EXPECT_NO_THROW(regex.initializeGrammar(
    "regex", "caller supplied regex", &tokenizer, tokenizer.GetVocabSize()));
  EXPECT_TRUE(regex.isGrammarEnabled());
}

TEST(XGrammarWrapper, RejectsMissingOrMalformedGrammarPayloads) {
  FakeTokenizer tokenizer(makeVocabulary());
  causallm::XGrammar grammar;

  EXPECT_THROW(
    grammar.initializeGrammar("ebnf", "", &tokenizer, tokenizer.GetVocabSize()),
    std::invalid_argument);
  EXPECT_THROW(grammar.initializeGrammar("regex", "", &tokenizer,
                                         tokenizer.GetVocabSize()),
               std::invalid_argument);
  EXPECT_ANY_THROW(grammar.initializeGrammar(
    "ebnf", R"(root ::= ")", &tokenizer, tokenizer.GetVocabSize()));
  EXPECT_ANY_THROW(grammar.initializeGrammar("regex", "(", &tokenizer,
                                             tokenizer.GetVocabSize()));
  EXPECT_FALSE(grammar.isGrammarEnabled());
}

TEST(XGrammarWrapper, ExplicitlyRejectsQuantizedLogitMasking) {
  causallm::XGrammar grammar;
  uint16_t logits[] = {17U, 23U};

  EXPECT_THROW(grammar.applyGrammarMask(logits, 2, 0.125F, 7),
               std::logic_error);
  EXPECT_EQ(logits[0], 17U);
  EXPECT_EQ(logits[1], 23U);
}

TEST(XGrammarManager, ConstructedInstancesHaveIndependentLifecycle) {
  FakeTokenizer tokenizer(makeVocabulary());
  causallm::XGrammarManager first;
  causallm::XGrammarManager second;

  EXPECT_FALSE(first.isInitialized());
  EXPECT_FALSE(second.isInitialized());
  EXPECT_FALSE(first.initialize(nullptr, tokenizer.GetVocabSize(),
                                makeTokenizerMetadata()));
  EXPECT_FALSE(first.initialize(&tokenizer, 0, makeTokenizerMetadata()));
  EXPECT_FALSE(first.initialize(&tokenizer, tokenizer.GetVocabSize(), "{}"));

  ASSERT_TRUE(first.initialize(&tokenizer, tokenizer.GetVocabSize(),
                               makeTokenizerMetadata()));
  ASSERT_TRUE(first.registerTool("test_tool",
                                 makeToolset("value").at("test_tool").dump()));

  EXPECT_TRUE(first.isInitialized());
  EXPECT_TRUE(first.hasTool("test_tool"));
  EXPECT_EQ(first.getToolNames().size(), 1U);
  EXPECT_FALSE(second.isInitialized());
  EXPECT_FALSE(second.hasTool("test_tool"));

  auto grammar_lease = first.getGrammarShared("test_tool");
  ASSERT_NE(grammar_lease, nullptr);
  first.unregisterTool("test_tool");
  EXPECT_FALSE(first.hasTool("test_tool"));
  EXPECT_NE(grammar_lease, nullptr);

  first.clear();
  EXPECT_FALSE(first.isInitialized());
  EXPECT_FALSE(first.hasTool("test_tool"));
  EXPECT_TRUE(grammar_lease->isGrammarEnabled());
}

TEST(XGrammarManager, LegacyInitializeOverloadRemainsAvailable) {
  FakeTokenizer tokenizer(makeVocabulary());
  causallm::XGrammarManager manager;

  EXPECT_TRUE(manager.initialize(&tokenizer, tokenizer.GetVocabSize()));
  EXPECT_TRUE(manager.isInitialized());
}

TEST(XGrammarManager, ReinitializeClearsRegisteredTools) {
  FakeTokenizer tokenizer(makeVocabulary());
  causallm::XGrammarManager manager;

  ASSERT_TRUE(manager.initialize(&tokenizer, tokenizer.GetVocabSize(),
                                 makeTokenizerMetadata()));
  ASSERT_TRUE(manager.registerTool(
    "test_tool", makeToolset("value").at("test_tool").dump()));
  ASSERT_TRUE(manager.hasTool("test_tool"));

  ASSERT_TRUE(manager.initialize(&tokenizer, tokenizer.GetVocabSize(),
                                 makeTokenizerMetadata()));
  EXPECT_TRUE(manager.isInitialized());
  EXPECT_FALSE(manager.hasTool("test_tool"));
  EXPECT_TRUE(manager.getToolNames().empty());
}

TEST(XGrammarManager, CacheIsBoundToToolsetAndTokenizerVocabulary) {
  ScopedToolset files;
  FakeTokenizer tokenizer(makeVocabulary());
  causallm::XGrammarManager manager;

  files.write(makeToolset("first"));
  ASSERT_TRUE(manager.initialize(&tokenizer, tokenizer.GetVocabSize(),
                                 makeTokenizerMetadata()));
  ASSERT_TRUE(
    manager.loadToolset(files.path(), &tokenizer, tokenizer.GetVocabSize()));
  ASSERT_TRUE(manager.hasTool("test_tool"));

  const json first_cache = files.readCache();
  EXPECT_EQ(first_cache.at("format_version"), 2);
  ASSERT_TRUE(first_cache.at("grammars").contains("test_tool"));
  const std::string first_toolset_fingerprint =
    first_cache.at("toolset_fingerprint");
  const std::string first_tokenizer_fingerprint =
    first_cache.at("tokenizer_fingerprint");
  const std::string first_metadata_fingerprint =
    first_cache.at("tokenizer_metadata_fingerprint");

  // A separately constructed manager can consume the serialized cache.
  causallm::XGrammarManager cached_manager;
  ASSERT_TRUE(cached_manager.initialize(&tokenizer, tokenizer.GetVocabSize(),
                                        makeTokenizerMetadata()));
  ASSERT_TRUE(cached_manager.loadToolset(files.path(), &tokenizer,
                                         tokenizer.GetVocabSize()));
  EXPECT_TRUE(cached_manager.hasTool("test_tool"));

  files.write(makeToolset("second"));
  ASSERT_TRUE(
    manager.loadToolset(files.path(), &tokenizer, tokenizer.GetVocabSize()));
  const json changed_toolset_cache = files.readCache();
  EXPECT_NE(changed_toolset_cache.at("toolset_fingerprint"),
            first_toolset_fingerprint);
  EXPECT_EQ(changed_toolset_cache.at("tokenizer_fingerprint"),
            first_tokenizer_fingerprint);

  auto changed_vocabulary = makeVocabulary();
  changed_vocabulary.emplace_back("integer");
  FakeTokenizer changed_tokenizer(std::move(changed_vocabulary));
  ASSERT_TRUE(manager.initialize(&changed_tokenizer,
                                 changed_tokenizer.GetVocabSize(),
                                 makeTokenizerMetadata()));
  ASSERT_TRUE(manager.loadToolset(files.path(), &changed_tokenizer,
                                  changed_tokenizer.GetVocabSize()));
  const json changed_tokenizer_cache = files.readCache();
  EXPECT_NE(changed_tokenizer_cache.at("tokenizer_fingerprint"),
            first_tokenizer_fingerprint);

  ASSERT_TRUE(manager.initialize(&changed_tokenizer,
                                 changed_tokenizer.GetVocabSize(),
                                 makeTokenizerMetadata(true)));
  ASSERT_TRUE(manager.loadToolset(files.path(), &changed_tokenizer,
                                  changed_tokenizer.GetVocabSize()));
  const json changed_metadata_cache = files.readCache();
  EXPECT_NE(changed_metadata_cache.at("tokenizer_fingerprint"),
            changed_tokenizer_cache.at("tokenizer_fingerprint"));
  EXPECT_NE(changed_metadata_cache.at("tokenizer_metadata_fingerprint"),
            first_metadata_fingerprint);
}

} // namespace
