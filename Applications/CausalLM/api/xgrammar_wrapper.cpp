// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   xgrammar_wrapper.cpp
 * @date   14 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file implements XGrammar class for grammar-guided generation
 */

#include "xgrammar_wrapper.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <variant>

#include <dlpack/dlpack.h>
#include <tokenizers_cpp.h>

namespace causallm {

XGrammar::XGrammar() {
  // Store configurations if needed
  grammar_enabled = false;
}

void XGrammar::initializeGrammar(const std::string &grammar_type,
                                 const std::string &grammar_payload,
                                 tokenizers::Tokenizer *tokenizer,
                                 unsigned int vocab_size) {
  if (tokenizer == nullptr) {
    throw std::runtime_error("Tokenizer is null. Cannot initialize grammar.");
  }
  if (vocab_size == 0 || vocab_size > static_cast<unsigned int>(
                                        std::numeric_limits<int32_t>::max())) {
    throw std::invalid_argument("Vocabulary size is invalid.");
  }
  // std::cout << "[xgrammar] Initializing grammar constraints...\n";

  // Step 1: Extract vocabulary from tokenizer
  // std::cout << "[xgrammar] Extracting vocabulary from tokenizer...\n";
  std::vector<std::string> encoded_vocab;
  encoded_vocab.reserve(vocab_size);

  for (size_t i = 0; i < vocab_size; ++i) {
    std::string token = tokenizer->IdToToken(static_cast<int32_t>(i));
    encoded_vocab.push_back(token);
  }
  // std::cout << "[xgrammar] Vocabulary size: " << vocab_size << "\n";

  // Step 2: Create TokenizerInfo from encoded vocabulary
  // std::cout << "[xgrammar] Creating TokenizerInfo...\n";
  auto tokenizer_info = std::make_unique<xgrammar::TokenizerInfo>(
    encoded_vocab, xgrammar::VocabType::BYTE_LEVEL, encoded_vocab.size());

  // Step 3: Create GrammarCompiler
  // std::cout << "[xgrammar] Creating GrammarCompiler...\n";
  auto grammar_compiler =
    std::make_unique<xgrammar::GrammarCompiler>(*tokenizer_info);

  // Step 4: Compile grammar using the shared helper
  compileGrammar(grammar_type, grammar_payload, grammar_compiler.get(),
                 vocab_size);
  tokenizer_info_ = std::move(tokenizer_info);
  grammar_compiler_ = std::move(grammar_compiler);
}

void XGrammar::initializeGrammar(const std::string &grammar_type,
                                 const std::string &grammar_payload,
                                 xgrammar::GrammarCompiler *grammar_compiler,
                                 unsigned int vocab_size) {
  if (grammar_compiler == nullptr) {
    throw std::runtime_error(
      "GrammarCompiler is null. Cannot initialize grammar.");
  }
  if (vocab_size == 0 || vocab_size > static_cast<unsigned int>(
                                        std::numeric_limits<int32_t>::max())) {
    throw std::invalid_argument("Vocabulary size is invalid.");
  }
  // Skip encoded vocab extraction - reuse shared GrammarCompiler
  // Compile grammar using the shared helper
  compileGrammar(grammar_type, grammar_payload, grammar_compiler, vocab_size);
}

void XGrammar::compileGrammar(const std::string &grammar_type,
                              const std::string &grammar_payload,
                              xgrammar::GrammarCompiler *grammar_compiler,
                              unsigned int vocab_size) {
  std::unique_ptr<xgrammar::CompiledGrammar> compiled_grammar;
  if (grammar_type == "json") {
    if (!grammar_payload.empty()) {
      // Compile from JSON schema
      compiled_grammar = std::make_unique<xgrammar::CompiledGrammar>(
        grammar_compiler->CompileJSONSchema(grammar_payload));
    } else {
      // Compile built-in JSON grammar
      compiled_grammar = std::make_unique<xgrammar::CompiledGrammar>(
        grammar_compiler->CompileBuiltinJSONGrammar());
      std::cout << "[xgrammar] Built-in JSON grammar compiled\n";
    }
  } else if (grammar_type == "ebnf") {
    if (grammar_payload.empty()) {
      throw std::invalid_argument("EBNF grammar payload is empty.");
    }
    compiled_grammar = std::make_unique<xgrammar::CompiledGrammar>(
      grammar_compiler->CompileGrammar(grammar_payload, "root"));
    std::cout << "[xgrammar] EBNF grammar compiled\n";
  } else if (grammar_type == "regex") {
    if (grammar_payload.empty()) {
      throw std::invalid_argument("Regex grammar payload is empty.");
    }
    compiled_grammar = std::make_unique<xgrammar::CompiledGrammar>(
      grammar_compiler->CompileRegex(grammar_payload));
    std::cout << "[xgrammar] Regex pattern compiled\n";
  } else {
    throw std::invalid_argument("Unsupported grammar type: " + grammar_type);
  }
  // std::cout << "[xgrammar] Grammar memory usage: "
  //           << compiled_grammar_->MemorySizeBytes() << " bytes\n";

  // Create GrammarMatcher
  // std::cout << "[xgrammar] Creating GrammarMatcher...\n";
  auto grammar_matcher =
    std::make_unique<xgrammar::GrammarMatcher>(*compiled_grammar);

  // std::cout << "[xgrammar] Grammar constraints initialized
  // successfully!\n\n"; Allocate bitmask storage (optimized - allocated once)
  int64_t bitmask_size = (static_cast<int64_t>(vocab_size) + 31) / 32;
  std::vector<int32_t> bitmask_data(static_cast<size_t>(bitmask_size));
  DLTensor bitmask_tensor{};
  bitmask_tensor.data = bitmask_data.data();
  bitmask_tensor.device = {kDLCPU, 0};
  bitmask_tensor.ndim = 1;
  bitmask_tensor.shape = &bitmask_size;
  bitmask_tensor.dtype = {kDLInt, 32, 1};
  bitmask_tensor.strides = nullptr;
  bitmask_tensor.byte_offset = 0;
  grammar_matcher->FillNextTokenBitmask(&bitmask_tensor);

  compiled_grammar_ = std::move(compiled_grammar);
  grammar_matcher_ = std::move(grammar_matcher);
  bitmask_data_ = std::move(bitmask_data);
  bitmask_size_ = bitmask_size;
  bitmask_tensor_ = bitmask_tensor;
  bitmask_tensor_.data = bitmask_data_.data();
  bitmask_tensor_.shape = &bitmask_size_;

  grammar_enabled = true;
}

void XGrammar::resetGrammar() {
  if (grammar_matcher_ != nullptr) {
    grammar_matcher_->Reset();
    // Update bitmask for initial token constraint after reset
    grammar_matcher_->FillNextTokenBitmask(&bitmask_tensor_);
  }
}

bool XGrammar::loadFromCache(const std::string &serialized_json,
                             xgrammar::TokenizerInfo *tokenizer_info,
                             unsigned int vocab_size) {
  if (tokenizer_info == nullptr) {
    std::cerr << "[xgrammar] Error: TokenizerInfo is null for cache loading\n";
    return false;
  }
  if (vocab_size == 0 || vocab_size > static_cast<unsigned int>(
                                        std::numeric_limits<int32_t>::max())) {
    std::cerr << "[xgrammar] Error: Vocabulary size is invalid\n";
    return false;
  }

  auto result = xgrammar::CompiledGrammar::DeserializeJSON(serialized_json,
                                                           *tokenizer_info);
  if (!std::holds_alternative<xgrammar::CompiledGrammar>(result)) {
    std::cerr << "[xgrammar] Error: Failed to deserialize compiled grammar "
                 "from cache\n";
    return false;
  }

  auto compiled_grammar = std::make_unique<xgrammar::CompiledGrammar>(
    std::get<xgrammar::CompiledGrammar>(std::move(result)));

  // Create GrammarMatcher
  auto grammar_matcher =
    std::make_unique<xgrammar::GrammarMatcher>(*compiled_grammar);

  // Allocate bitmask storage
  int64_t bitmask_size = (static_cast<int64_t>(vocab_size) + 31) / 32;
  std::vector<int32_t> bitmask_data(static_cast<size_t>(bitmask_size));
  DLTensor bitmask_tensor{};
  bitmask_tensor.data = bitmask_data.data();
  bitmask_tensor.device = {kDLCPU, 0};
  bitmask_tensor.ndim = 1;
  bitmask_tensor.shape = &bitmask_size;
  bitmask_tensor.dtype = {kDLInt, 32, 1};
  bitmask_tensor.strides = nullptr;
  bitmask_tensor.byte_offset = 0;
  grammar_matcher->FillNextTokenBitmask(&bitmask_tensor);

  compiled_grammar_ = std::move(compiled_grammar);
  grammar_matcher_ = std::move(grammar_matcher);
  bitmask_data_ = std::move(bitmask_data);
  bitmask_size_ = bitmask_size;
  bitmask_tensor_ = bitmask_tensor;
  bitmask_tensor_.data = bitmask_data_.data();
  bitmask_tensor_.shape = &bitmask_size_;

  grammar_enabled = true;
  return true;
}

std::string XGrammar::serialize() const {
  if (compiled_grammar_ == nullptr) {
    return "";
  }
  return compiled_grammar_->SerializeJSON();
}

std::vector<int32_t> &XGrammar::getBitmaskData() { return bitmask_data_; }

DLTensor &XGrammar::getBitmaskTensor() { return bitmask_tensor_; }

int64_t XGrammar::getBitmaskSize() { return bitmask_size_; }

xgrammar::GrammarMatcher *XGrammar::getGrammarMatcher() {
  return grammar_matcher_.get();
}

void XGrammar::applyGrammarMask(float *logits, int vocab_size) {
  if (logits == nullptr)
    throw std::invalid_argument("Logit buffer is null.");
  if (vocab_size < 0 ||
      bitmask_data_.size() < (static_cast<size_t>(vocab_size) + 31U) / 32U) {
    throw std::invalid_argument("Grammar mask does not match vocabulary size.");
  }
  for (int i = 0; i < vocab_size; ++i) {
    const auto block =
      static_cast<uint32_t>(bitmask_data_[static_cast<size_t>(i) / 32U]);
    bool is_accepted = (block >> (i % 32)) & 1;

    if (!is_accepted) {
      logits[i] = -std::numeric_limits<float>::infinity();
    }
  }
}

void XGrammar::applyGrammarMask(uint16_t *logits, int vocab_size,
                                float logit_scale, int logit_offset) {
  static_cast<void>(logits);
  static_cast<void>(vocab_size);
  static_cast<void>(logit_scale);
  static_cast<void>(logit_offset);
  throw std::logic_error(
    "Quantized grammar masking is unsupported; apply the grammar bitmask "
    "during sampling instead.");
}

} // namespace causallm
