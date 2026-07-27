// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   openai_request.h
 * @brief  Validation and normalization for OpenAI-compatible chat requests
 * @author Quick.AI Team
 * @bug    No known bugs except for NYI items
 */

#ifndef NNTRAINER_CAUSALLM_OPENAI_REQUEST_H_
#define NNTRAINER_CAUSALLM_OPENAI_REQUEST_H_

#include "json.hpp"

#include <optional>
#include <string>
#include <vector>

namespace causallm {
namespace openai {

/**
 * @brief Placeholder inserted into rendered text for each image content part
 */
constexpr const char *DEFAULT_IMAGE_PLACEHOLDER = "<|image|>";

/**
 * @brief Supported OpenAI message content part types
 */
enum class ContentPartType { TEXT, IMAGE_URL };

/**
 * @brief A validated message content part whose array order is preserved
 */
struct ContentPart {
  ContentPartType type = ContentPartType::TEXT;
  std::string text;
  std::string image_url;
  std::string image_detail;
};

/**
 * @brief A validated OpenAI chat message
 */
struct Message {
  std::string role;
  std::vector<ContentPart> content;
};

/**
 * @brief A normalized function tool from either tools[] or legacy functions[]
 */
struct FunctionTool {
  std::string name;
  std::string description;
  nlohmann::json parameters = nlohmann::json::object();
};

/**
 * @brief The decoding constraint selected from the request
 */
enum class GrammarKind {
  NONE,
  JSON_OBJECT,
  JSON_SCHEMA,
  TOOL_CALL,
};

/**
 * @brief Grammar metadata for a generation caller
 *
 * For JSON_SCHEMA and TOOL_CALL, @c schema is ready to serialize and pass to
 * an xgrammar JSON-schema compiler. Tool constraints describe the complete
 * normalized {"name": ..., "arguments": {...}} output envelope, not just the
 * function arguments. An omitted OpenAI response-format schema is normalized
 * to an empty JSON Schema object. A named choice sets @c forced_tool_name.
 */
struct GrammarSelection {
  GrammarKind kind = GrammarKind::NONE;
  nlohmann::json schema;
  std::vector<FunctionTool> tools;
  std::optional<std::string> forced_tool_name;
};

/**
 * @brief Fully validated OpenAI request data needed by the generation API
 */
struct Request {
  nlohmann::json original;
  std::vector<Message> messages;
  std::vector<FunctionTool> tools;
  std::vector<std::string> image_sources;
  /** Top-level OpenAI fields whose behavior this runtime cannot honor. */
  std::vector<std::string> unsupported_fields;
  GrammarSelection grammar;
};

/**
 * @brief Parse a content string or an ordered array of text/image_url parts
 * @param content JSON message content
 * @param path Request path used in validation errors
 * @return Validated content parts in their original order
 * @throws std::invalid_argument when the content is malformed
 */
std::vector<ContentPart> parseContentParts(const nlohmann::json &content,
                                           const std::string &path = "content");

/**
 * @brief Render content parts while preserving text/image ordering
 * @param parts Validated content parts
 * @param image_placeholder Text inserted for every image part
 * @return Rendered text with image placeholders
 */
std::string renderContentWithImagePlaceholders(
  const std::vector<ContentPart> &parts,
  const std::string &image_placeholder = DEFAULT_IMAGE_PLACEHOLDER);

/**
 * @brief Parse and validate an already-decoded JSON request
 * @param request Decoded OpenAI-compatible request
 * @return Validated and normalized request
 * @throws std::invalid_argument when the request is malformed
 */
Request parseRequest(const nlohmann::json &request);

/**
 * @brief Parse and validate a UTF-8 JSON request string
 * @param request_json UTF-8 encoded request JSON
 * @return Validated and normalized request
 * @throws std::invalid_argument when the request is malformed
 */
Request parseRequest(const std::string &request_json);

} // namespace openai
} // namespace causallm

#endif // NNTRAINER_CAUSALLM_OPENAI_REQUEST_H_
