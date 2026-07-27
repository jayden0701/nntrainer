// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   xgrammar_wrapper.h
 * @date   14 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @brief  The header for the support of grammar-guided generation.
 * @bug    No known bugs except for NYI items
 */

#ifndef NNTRAINER_CAUSALLM_XGRAMMAR_WRAPPER_H_
#define NNTRAINER_CAUSALLM_XGRAMMAR_WRAPPER_H_

#pragma once
#ifndef WIN_EXPORT
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>
#include <xgrammar/compiler.h>
#include <xgrammar/config.h>
#include <xgrammar/exception.h>
#include <xgrammar/grammar.h>
#include <xgrammar/matcher.h>
#include <xgrammar/tokenizer_info.h>

// Forward declarations
namespace tokenizers {
class Tokenizer;
}

namespace causallm {
/**
 * @brief Grammar-guided generation helper wrapping xgrammar's compiler and
 * matcher.
 */
class WIN_EXPORT XGrammar {

public:
  XGrammar();

  /**
   * @brief Destroy the grammar helper
   */
  virtual ~XGrammar() = default;

  /**
   * @brief Initialize xgrammar for grammar-guided generation
   * @param grammar_type Type of grammar ("json", "ebnf", "regex")
   * @param grammar_payload Optional JSON schema for "json", or the required

   * * EBNF grammar/regular expression for "ebnf"/"regex"
   * @param tokenizer
   * Tokenizer instance to extract vocabulary
   * @param vocab_size Size of the vocabulary
   */
  void initializeGrammar(const std::string &grammar_type = "json",
                         const std::string &grammar_payload = "",
                         tokenizers::Tokenizer *tokenizer = nullptr,
                         unsigned int vocab_size = 0);

  /**
   * @brief Initialize xgrammar with pre-created TokenizerInfo and
   * GrammarCompiler This is optimized for cases where multiple grammars share
   * the same tokenizer.
   * @param grammar_type Type of grammar ("json", "ebnf", "regex")
   * @param grammar_payload Optional JSON schema for "json", or the required

   * * EBNF grammar/regular expression for "ebnf"/"regex"
   * @param
   * grammar_compiler Pre-created GrammarCompiler (shared, not owned)
   * @param vocab_size Size of the vocabulary (for bitmask allocation)
   */
  void initializeGrammar(const std::string &grammar_type,
                         const std::string &grammar_payload,
                         xgrammar::GrammarCompiler *grammar_compiler,
                         unsigned int vocab_size);

  /**
   * @brief Reset grammar matcher state
   */
  void resetGrammar();

  /**
   * @brief Check if grammar constraints are enabled
   */
  bool isGrammarEnabled() const { return grammar_enabled; }

  /**
   * @brief Load grammar from serialized JSON cache
   * @param serialized_json The serialized compiled grammar JSON string
   * @param tokenizer_info TokenizerInfo needed for deserialization
   * @param vocab_size Size of the vocabulary (for bitmask allocation)
   * @return true on success, false on failure
   */
  bool loadFromCache(const std::string &serialized_json,
                     xgrammar::TokenizerInfo *tokenizer_info,
                     unsigned int vocab_size);

  /**
   * @brief Get serialized JSON of the compiled grammar
   * @return JSON string representation of the compiled grammar
   */
  std::string serialize() const;

  /**
   * @brief Get bitmask data
   * @return Mutable grammar bitmask storage

   */
  std::vector<int32_t> &getBitmaskData();

  /**
   * @brief Get bitmask tensor
   * @return DLPack view over the grammar
   * bitmask
   */
  DLTensor &getBitmaskTensor();

  /**
   * @brief Get bitmask size
   * @return Number of 32-bit bitmask
   * elements
   */
  int64_t getBitmaskSize();

  /**
   * @brief Get grammar matcher pointer
   * @return Non-owning matcher
   * pointer
   */
  xgrammar::GrammarMatcher *getGrammarMatcher();

  /**
   * @brief Apply the current grammar mask to floating-point logits
   *
   * @param logits Mutable logit buffer
   * @param vocab_size Number of logits
   * in the buffer
   */
  void applyGrammarMask(float *logits, int vocab_size);

  /**
   * @brief Report that in-place masking of quantized logits is
   * unsupported
   * @param logits Mutable quantized logit buffer
   * @param
   * vocab_size Number of logits in the buffer
   * @param logit_scale
   * Quantization scale
   * @param logit_offset Quantization offset
   *
   * @throws std::logic_error Quantized masking cannot represent negative
   *
   * infinity and is unsupported; use the bitmask directly instead
   */
  void applyGrammarMask(uint16_t *logits, int vocab_size, float logit_scale,
                        int logit_offset);

private:
  /**
   * @brief Internal helper to compile grammar (shared between overloaded
   * methods)
   */
  void compileGrammar(const std::string &grammar_type,
                      const std::string &grammar_payload,
                      xgrammar::GrammarCompiler *grammar_compiler,
                      unsigned int vocab_size);

protected:
  // xgrammar components for grammar-guided generation
  std::unique_ptr<xgrammar::TokenizerInfo> tokenizer_info_;
  std::unique_ptr<xgrammar::GrammarCompiler> grammar_compiler_;
  std::unique_ptr<xgrammar::CompiledGrammar> compiled_grammar_;
  std::unique_ptr<xgrammar::GrammarMatcher> grammar_matcher_;

  // Optimized bitmask storage (reused across generations)
  DLTensor bitmask_tensor_{};
  std::vector<int32_t> bitmask_data_;
  int64_t bitmask_size_ = 0;

  bool grammar_enabled = false;
};

} // namespace causallm

#endif // NNTRAINER_CAUSALLM_XGRAMMAR_WRAPPER_H_
