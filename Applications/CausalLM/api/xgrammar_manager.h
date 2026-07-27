// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   xgrammar_manager.h
 * @date   14 April 2026
 * @brief  XGrammar manager for grammar-constrained generation
 * @see    https://github.com/nntrainer/nntrainer
 * @author Quick.AI Team
 * @bug    No known bugs except for NYI items
 *
 * @note   This manager pre-compiles tool grammars from Toolset.json and
 *         provides grammar instances for structured decoding. Use one manager
 *         per model handle and reinitialize it when the model changes.
 */

#ifndef NNTRAINER_CAUSALLM_XGRAMMAR_MANAGER_H_
#define NNTRAINER_CAUSALLM_XGRAMMAR_MANAGER_H_

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <xgrammar/compiler.h>
#include <xgrammar/grammar.h>
#include <xgrammar/matcher.h>
#include <xgrammar/tokenizer_info.h>

// Forward declarations
namespace tokenizers {
class Tokenizer;
}

namespace causallm {

class XGrammar;

/**
 * @brief Manages compiled grammars for one model and tokenizer
 *
 * Handle-based callers should construct one manager per handle. Instance()
 * remains available for legacy process-wide callers.
 */
class XGrammarManager {
public:
  /**
   * @brief Shared grammar reference that keeps a grammar alive
   */
  using GrammarPtr = std::shared_ptr<XGrammar>;

  /**
   * @brief Construct an independent grammar manager
   */
  XGrammarManager();

  /**
   * @brief Destroy the grammar manager
   */
  ~XGrammarManager();

  XGrammarManager(const XGrammarManager &) = delete;
  XGrammarManager &operator=(const XGrammarManager &) = delete;

  /**
   * @brief Get the legacy process-wide singleton instance
   * @return Singleton grammar manager
   */
  static XGrammarManager &Instance();

  /**
   * @brief Load toolset from JSON file and pre-compile all grammars
   * @param toolset_path Path to Toolset.json file
   * @param tokenizer Pointer to the tokenizer instance
   * @param vocab_size Size of the vocabulary
   * @return true on success, false on failure
   * @note initialize() must first be called for the same tokenizer.
   */
  bool loadToolset(const std::string &toolset_path,
                   tokenizers::Tokenizer *tokenizer, unsigned int vocab_size);

  /**
   * @brief Acquire a compiled grammar while retaining its lifetime
   * @param tool_name Name of the tool
   * @return Shared grammar reference, or an empty reference if not found
   */
  GrammarPtr getGrammarShared(const std::string &tool_name);

  /**
   * @brief Get a legacy non-owning grammar pointer for a specific tool
   * @param tool_name Name of the tool (e.g., "alarm", "send_email", "memo")
   * @return Pointer to XGrammar, or nullptr if not found
   * @warning The returned pointer can become invalid after clear(),
   *          initialize(), unregisterTool(), or loadToolset(). New callers
   *          should retain getGrammarShared() instead.
   */
  XGrammar *getGrammar(const std::string &tool_name);

  /**
   * @brief Reset grammar matcher state for a specific tool
   * @param tool_name Name of the tool
   */
  void resetGrammar(const std::string &tool_name);

  /**
   * @brief Check if a tool exists in the loaded toolset
   * @param tool_name Name of the tool
   * @return true if tool exists, false otherwise
   */
  bool hasTool(const std::string &tool_name) const;

  /**
   * @brief Get list of all available tool names
   * @return Vector of tool names
   */
  std::vector<std::string> getToolNames() const;

  /**
   * @brief Initialize using the legacy byte-level tokenizer assumptions
   * @param tokenizer Tokenizer instance used by the model
   * @param vocab_size Size of the tokenizer vocabulary
   * @return true on success, false on failure
   * @note This compatibility overload assumes BYTE_LEVEL vocabulary with no
   *       prefix space. New callers should pass detected tokenizer metadata.
   */
  bool initialize(tokenizers::Tokenizer *tokenizer, unsigned int vocab_size);

  /**
   * @brief Initialize using xgrammar tokenizer metadata
   * @param tokenizer Tokenizer instance used by the model
   * @param vocab_size Size of the tokenizer vocabulary
   * @param tokenizer_metadata Metadata returned by
   *        xgrammar::TokenizerInfo::DetectMetadataFromHF()
   * @return true on success, false on failure
   */
  bool initialize(tokenizers::Tokenizer *tokenizer, unsigned int vocab_size,
                  const std::string &tokenizer_metadata);

  /**
   * @brief Check if manager is initialized
   * @return true if initialized, false otherwise
   */
  bool isInitialized() const;

  /**
   * @brief Register a single tool with its JSON schema dynamically
   * @param tool_name Name of the tool
   * @param json_schema JSON schema string for the tool
   * @return true on success, false on failure
   * @note Requires initialize() or loadToolset() first.
   */
  bool registerTool(const std::string &tool_name,
                    const std::string &json_schema);

  /**
   * @brief Remove one dynamically registered grammar
   * @param tool_name Name of the tool to remove
   */
  void unregisterTool(const std::string &tool_name);

  /**
   * @brief Clear all compiled grammars and tokenizer state
   */
  void clear();

private:
  /**
   * @brief Initialize while mutex_ is already held
   */
  bool initializeUnlocked(tokenizers::Tokenizer *tokenizer,
                          unsigned int vocab_size,
                          const std::string &tokenizer_metadata);

  /**
   * @brief Clear all state while mutex_ is already held
   */
  void clearUnlocked();

  // Pre-compiled grammars: tool_name -> XGrammar
  std::unordered_map<std::string, GrammarPtr> compiled_grammars_;

  // Shared tokenizer info (created once per model)
  std::shared_ptr<xgrammar::TokenizerInfo> tokenizer_info_;
  std::shared_ptr<xgrammar::GrammarCompiler> grammar_compiler_;

  // Current toolset path (to detect if reload needed)
  std::string current_toolset_path_;

  // Tokenizer identity used to validate serialized grammar caches.
  tokenizers::Tokenizer *tokenizer_ = nullptr;
  unsigned int vocab_size_ = 0;
  std::string tokenizer_fingerprint_;
  std::string tokenizer_metadata_fingerprint_;

  // Initialization flag
  bool initialized_ = false;

  // Thread safety
  mutable std::mutex mutex_;
};

} // namespace causallm

#endif // NNTRAINER_CAUSALLM_XGRAMMAR_MANAGER_H_
