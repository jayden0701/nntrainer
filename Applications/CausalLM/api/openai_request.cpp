// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   openai_request.cpp
 * @brief  OpenAI-compatible chat request parser and validator
 * @author Quick.AI Team
 * @bug    No known bugs except for NYI items
 */

#include "openai_request.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace causallm {
namespace openai {
namespace {

[[noreturn]] void fail(const std::string &path, const std::string &message) {
  throw std::invalid_argument(path + ": " + message);
}

std::string requireString(const nlohmann::json &value, const std::string &path,
                          bool allow_empty = false) {
  if (!value.is_string())
    fail(path, "must be a string");

  std::string result = value.get<std::string>();
  if (!allow_empty && result.empty())
    fail(path, "must not be empty");
  return result;
}

bool isSupportedRole(const std::string &role) {
  return role == "system" || role == "developer" || role == "user" ||
         role == "assistant" || role == "tool" || role == "function";
}

nlohmann::json defaultParametersSchema() {
  return {{"type", "object"},
          {"properties", nlohmann::json::object()},
          {"additionalProperties", false}};
}

bool isValidFunctionName(const std::string &name) {
  if (name.empty() || name.size() > 64)
    return false;
  return std::all_of(name.begin(), name.end(), [](unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_' || c == '-';
  });
}

void validateFunctionCallHistory(const nlohmann::json &function,
                                 const std::string &path) {
  if (!function.is_object())
    fail(path, "must be an object");
  if (!function.contains("name"))
    fail(path + ".name", "is required");
  const std::string name = requireString(function["name"], path + ".name");
  if (!isValidFunctionName(name))
    fail(path + ".name", "is not a valid function name");
  if (!function.contains("arguments"))
    fail(path + ".arguments", "is required");
  requireString(function["arguments"], path + ".arguments", true);
}

void validateAssistantToolHistory(const nlohmann::json &message,
                                  const std::string &path) {
  if (message.contains("tool_calls") && message.contains("function_call")) {
    fail(path, "tool_calls and legacy function_call cannot both be present");
  }

  if (message.contains("tool_calls")) {
    const auto &calls = message["tool_calls"];
    if (!calls.is_array() || calls.empty())
      fail(path + ".tool_calls", "must be a non-empty array");
    std::unordered_set<std::string> ids;
    for (size_t i = 0; i < calls.size(); ++i) {
      const auto &call = calls[i];
      const std::string call_path =
        path + ".tool_calls[" + std::to_string(i) + "]";
      if (!call.is_object())
        fail(call_path, "must be an object");
      if (!call.contains("id"))
        fail(call_path + ".id", "is required");
      const std::string id = requireString(call["id"], call_path + ".id");
      if (!ids.insert(id).second)
        fail(call_path + ".id", "must be unique within the message");
      if (!call.contains("type"))
        fail(call_path + ".type", "is required");
      if (requireString(call["type"], call_path + ".type") != "function") {
        fail(call_path + ".type", "only function tool calls are supported");
      }
      if (!call.contains("function"))
        fail(call_path + ".function", "is required");
      validateFunctionCallHistory(call["function"], call_path + ".function");
    }
  }

  if (message.contains("function_call")) {
    validateFunctionCallHistory(message["function_call"],
                                path + ".function_call");
  }
}

FunctionTool parseFunction(const nlohmann::json &function,
                           const std::string &path) {
  if (!function.is_object())
    fail(path, "must be an object");
  if (!function.contains("name"))
    fail(path + ".name", "is required");

  FunctionTool result;
  result.name = requireString(function["name"], path + ".name");
  if (!isValidFunctionName(result.name)) {
    fail(path + ".name",
         "must contain only letters, digits, underscores, or hyphens and be "
         "at most 64 characters");
  }
  if (function.contains("description")) {
    result.description =
      requireString(function["description"], path + ".description", true);
  }

  if (function.contains("parameters")) {
    if (!function["parameters"].is_object() &&
        !function["parameters"].is_boolean()) {
      fail(path + ".parameters", "must be a JSON schema object or boolean");
    }
    result.parameters = function["parameters"];
  } else {
    result.parameters = defaultParametersSchema();
  }
  if (function.contains("strict") && !function["strict"].is_boolean())
    fail(path + ".strict", "must be a boolean");
  return result;
}

std::vector<FunctionTool> parseTools(const nlohmann::json &request,
                                     bool &uses_legacy_functions) {
  const bool has_tools = request.contains("tools");
  const bool has_functions = request.contains("functions");
  if (has_tools && has_functions)
    fail("request", "tools and legacy functions cannot both be present");
  if (has_tools && request.contains("function_call")) {
    fail("request.function_call",
         "cannot be combined with tools; use tool_choice");
  }
  if (has_functions && request.contains("tool_choice")) {
    fail("request.tool_choice",
         "cannot be combined with legacy functions; use function_call");
  }
  if (!has_functions && request.contains("function_call"))
    fail("request.function_call", "requires legacy functions");
  if (!has_tools && request.contains("tool_choice"))
    fail("request.tool_choice", "requires tools");

  uses_legacy_functions = has_functions;
  if (!has_tools && !has_functions)
    return {};

  const char *field = has_tools ? "tools" : "functions";
  const auto &entries = request[field];
  if (!entries.is_array())
    fail(std::string("request.") + field, "must be an array");

  std::vector<FunctionTool> result;
  result.reserve(entries.size());
  std::unordered_set<std::string> names;
  for (size_t i = 0; i < entries.size(); ++i) {
    const auto &entry = entries[i];
    const std::string path =
      "request." + std::string(field) + "[" + std::to_string(i) + "]";
    if (!entry.is_object())
      fail(path, "must be an object");

    FunctionTool tool;
    if (has_tools) {
      if (!entry.contains("type"))
        fail(path + ".type", "is required");
      if (requireString(entry["type"], path + ".type") != "function") {
        fail(path + ".type", "only function tools are supported");
      }
      if (!entry.contains("function"))
        fail(path + ".function", "is required");
      tool = parseFunction(entry["function"], path + ".function");
    } else {
      tool = parseFunction(entry, path);
    }

    if (!names.insert(tool.name).second)
      fail(path, "duplicate function name: " + tool.name);
    result.push_back(std::move(tool));
  }
  return result;
}

enum class ToolChoiceKind { AUTO, NONE, REQUIRED, NAMED };

struct ToolChoice {
  ToolChoiceKind kind = ToolChoiceKind::AUTO;
  std::string name;
};

ToolChoice parseToolChoice(const nlohmann::json &request,
                           bool uses_legacy_functions) {
  const char *field = uses_legacy_functions ? "function_call" : "tool_choice";
  if (!request.contains(field))
    return {};

  const auto &choice = request[field];
  const std::string path = "request." + std::string(field);
  if (choice.is_string()) {
    const std::string value = choice.get<std::string>();
    if (value == "auto")
      return {ToolChoiceKind::AUTO, {}};
    if (value == "none")
      return {ToolChoiceKind::NONE, {}};
    if (!uses_legacy_functions && value == "required")
      return {ToolChoiceKind::REQUIRED, {}};
    fail(path, uses_legacy_functions
                 ? "must be auto, none, or a named function object"
                 : "must be auto, none, required, or a named function object");
  }

  if (!choice.is_object())
    fail(path, "must be a string or object");

  if (uses_legacy_functions) {
    if (!choice.contains("name"))
      fail(path + ".name", "is required");
    return {ToolChoiceKind::NAMED,
            requireString(choice["name"], path + ".name")};
  }

  if (!choice.contains("type"))
    fail(path + ".type", "is required");
  if (requireString(choice["type"], path + ".type") != "function") {
    fail(path + ".type", "only function tool choices are supported");
  }
  if (!choice.contains("function") || !choice["function"].is_object())
    fail(path + ".function", "must be an object");
  if (!choice["function"].contains("name"))
    fail(path + ".function.name", "is required");
  return {ToolChoiceKind::NAMED,
          requireString(choice["function"]["name"], path + ".function.name")};
}

GrammarSelection parseResponseFormat(const nlohmann::json &request) {
  GrammarSelection result;
  if (!request.contains("response_format"))
    return result;

  const auto &format = request["response_format"];
  if (!format.is_object())
    fail("request.response_format", "must be an object");
  if (!format.contains("type"))
    fail("request.response_format.type", "is required");

  const std::string type =
    requireString(format["type"], "request.response_format.type");
  if (type == "text")
    return result;
  if (type == "json_object") {
    result.kind = GrammarKind::JSON_OBJECT;
    return result;
  }
  if (type != "json_schema")
    fail("request.response_format.type",
         "unsupported response format: " + type);

  if (!format.contains("json_schema") || !format["json_schema"].is_object()) {
    fail("request.response_format.json_schema", "must be an object");
  }
  const auto &wrapper = format["json_schema"];
  if (!wrapper.contains("name"))
    fail("request.response_format.json_schema.name", "is required");
  const std::string schema_name =
    requireString(wrapper["name"], "request.response_format.json_schema.name");
  if (!isValidFunctionName(schema_name)) {
    fail("request.response_format.json_schema.name",
         "must contain only letters, digits, underscores, or hyphens and be "
         "at most 64 characters");
  }
  if (wrapper.contains("description")) {
    requireString(wrapper["description"],
                  "request.response_format.json_schema.description", true);
  }
  if (wrapper.contains("strict") && !wrapper["strict"].is_boolean())
    fail("request.response_format.json_schema.strict", "must be a boolean");
  if (wrapper.contains("schema") && !wrapper["schema"].is_object()) {
    fail("request.response_format.json_schema.schema",
         "must be a JSON schema object");
  }

  result.kind = GrammarKind::JSON_SCHEMA;
  result.schema = wrapper.value("schema", nlohmann::json::object());
  return result;
}

GrammarSelection selectToolGrammar(const std::vector<FunctionTool> &tools,
                                   const ToolChoice &choice) {
  GrammarSelection result;
  if (choice.kind == ToolChoiceKind::AUTO ||
      choice.kind == ToolChoiceKind::NONE) {
    return result;
  }
  if (tools.empty())
    fail("request.tool_choice", "requires at least one function tool");

  result.kind = GrammarKind::TOOL_CALL;
  auto tool_call_schema = [](const FunctionTool &tool) {
    return nlohmann::json{
      {"type", "object"},
      {"properties",
       {{"name", {{"type", "string"}, {"const", tool.name}}},
        {"arguments", tool.parameters}}},
      {"required", {"name", "arguments"}},
      {"additionalProperties", false},
    };
  };
  if (choice.kind == ToolChoiceKind::NAMED) {
    const auto it = std::find_if(
      tools.begin(), tools.end(),
      [&choice](const FunctionTool &tool) { return tool.name == choice.name; });
    if (it == tools.end())
      fail("request.tool_choice", "names an unknown function: " + choice.name);
    result.tools.push_back(*it);
    result.schema = tool_call_schema(*it);
    result.forced_tool_name = choice.name;
    return result;
  }

  result.tools = tools;
  if (tools.size() == 1) {
    result.schema = tool_call_schema(tools.front());
  } else {
    result.schema = {{"oneOf", nlohmann::json::array()}};
    for (const auto &tool : tools)
      result.schema["oneOf"].push_back(tool_call_schema(tool));
  }
  return result;
}

} // namespace

std::vector<ContentPart> parseContentParts(const nlohmann::json &content,
                                           const std::string &path) {
  if (content.is_string()) {
    ContentPart part;
    part.type = ContentPartType::TEXT;
    part.text = content.get<std::string>();
    return {std::move(part)};
  }
  if (!content.is_array())
    fail(path, "must be a string or an array of content parts");
  if (content.empty())
    fail(path, "content parts must not be empty");

  std::vector<ContentPart> result;
  result.reserve(content.size());
  for (size_t i = 0; i < content.size(); ++i) {
    const auto &entry = content[i];
    const std::string part_path = path + "[" + std::to_string(i) + "]";
    if (!entry.is_object())
      fail(part_path, "must be an object");
    if (!entry.contains("type"))
      fail(part_path + ".type", "is required");
    const std::string type = requireString(entry["type"], part_path + ".type");

    ContentPart part;
    if (type == "text") {
      if (!entry.contains("text"))
        fail(part_path + ".text", "is required");
      part.type = ContentPartType::TEXT;
      part.text = requireString(entry["text"], part_path + ".text", true);
    } else if (type == "image_url") {
      if (!entry.contains("image_url"))
        fail(part_path + ".image_url", "is required");
      part.type = ContentPartType::IMAGE_URL;
      const auto &image = entry["image_url"];
      if (image.is_string()) {
        part.image_url = requireString(image, part_path + ".image_url");
      } else if (image.is_object()) {
        if (!image.contains("url"))
          fail(part_path + ".image_url.url", "is required");
        part.image_url =
          requireString(image["url"], part_path + ".image_url.url");
        if (image.contains("detail")) {
          part.image_detail =
            requireString(image["detail"], part_path + ".image_url.detail");
          if (part.image_detail != "auto" && part.image_detail != "low" &&
              part.image_detail != "high") {
            fail(part_path + ".image_url.detail",
                 "must be one of auto, low, or high");
          }
        }
      } else {
        fail(part_path + ".image_url", "must be a string or object");
      }
    } else {
      fail(part_path + ".type", "unsupported content part: " + type);
    }
    result.push_back(std::move(part));
  }
  return result;
}

std::string
renderContentWithImagePlaceholders(const std::vector<ContentPart> &parts,
                                   const std::string &image_placeholder) {
  std::string result;
  for (const auto &part : parts) {
    if (part.type == ContentPartType::TEXT)
      result += part.text;
    else
      result += image_placeholder;
  }
  return result;
}

Request parseRequest(const nlohmann::json &request) {
  if (!request.is_object())
    fail("request", "must be a JSON object");
  if (!request.contains("messages") || !request["messages"].is_array())
    fail("request.messages", "must be an array");
  if (request["messages"].empty())
    fail("request.messages", "must not be empty");

  Request result;
  result.original = request;
  static const std::unordered_set<std::string> supported_fields = {
    "messages",
    "model",
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "response_format",
    "stream",
    "user",
    "metadata",
    "store",
    "add_generation_prompt",
    "parallel_tool_calls",
  };
  for (auto it = request.begin(); it != request.end(); ++it) {
    if (supported_fields.count(it.key()) == 0)
      result.unsupported_fields.push_back(it.key());
  }

  if (request.contains("model"))
    requireString(request["model"], "request.model");
  if (request.contains("stream") && !request["stream"].is_boolean())
    fail("request.stream", "must be a boolean");
  if (request.contains("user"))
    requireString(request["user"], "request.user", true);
  if (request.contains("metadata") && !request["metadata"].is_object() &&
      !request["metadata"].is_null()) {
    fail("request.metadata", "must be an object or null");
  }
  if (request.contains("store")) {
    if (!request["store"].is_boolean())
      fail("request.store", "must be a boolean");
    if (request["store"].get<bool>())
      result.unsupported_fields.push_back("store");
  }
  if (request.contains("add_generation_prompt") &&
      !request["add_generation_prompt"].is_boolean()) {
    fail("request.add_generation_prompt", "must be a boolean");
  }
  if (request.contains("parallel_tool_calls")) {
    if (!request["parallel_tool_calls"].is_boolean())
      fail("request.parallel_tool_calls", "must be a boolean");
    if (request["parallel_tool_calls"].get<bool>())
      result.unsupported_fields.push_back("parallel_tool_calls");
  }

  const auto &messages = request["messages"];
  result.messages.reserve(messages.size());
  struct PendingToolCall {
    std::string function_name;
    bool legacy = false;
  };
  std::unordered_map<std::string, PendingToolCall> pending_tool_calls;
  for (size_t i = 0; i < messages.size(); ++i) {
    const auto &entry = messages[i];
    const std::string path = "request.messages[" + std::to_string(i) + "]";
    if (!entry.is_object())
      fail(path, "must be an object");
    if (!entry.contains("role"))
      fail(path + ".role", "is required");

    Message message;
    const std::string input_role = requireString(entry["role"], path + ".role");
    message.role = input_role;
    if (!isSupportedRole(message.role))
      fail(path + ".role", "unsupported role: " + message.role);
    const bool legacy_function_result = input_role == "function";

    if (!pending_tool_calls.empty()) {
      const bool expects_legacy = pending_tool_calls.begin()->second.legacy;
      if ((expects_legacy && !legacy_function_result) ||
          (!expects_legacy && input_role != "tool")) {
        fail(path + ".role",
             expects_legacy
               ? "must resolve the preceding function_call with role function"
               : "must resolve all preceding tool_calls with role tool");
      }
    }

    if (legacy_function_result) {
      message.role = "tool";
      result.original["messages"][i]["role"] = "tool";
      result.original["messages"][i]["tool_call_id"] = "function_call";
    }

    const bool has_tool_history =
      entry.contains("tool_calls") || entry.contains("function_call");
    if (has_tool_history && message.role != "assistant") {
      fail(path, "tool_calls and function_call require role assistant");
    }
    if (message.role == "assistant") {
      validateAssistantToolHistory(entry, path);
      if (entry.contains("tool_calls")) {
        for (const auto &call : entry["tool_calls"]) {
          const std::string id = call["id"].get<std::string>();
          const std::string function_name =
            call["function"]["name"].get<std::string>();
          if (!pending_tool_calls
                 .emplace(id, PendingToolCall{function_name, false})
                 .second) {
            fail(path + ".tool_calls", "reuses an unresolved tool call id");
          }
        }
      } else if (entry.contains("function_call")) {
        const std::string function_name =
          entry["function_call"]["name"].get<std::string>();
        if (!pending_tool_calls
               .emplace("function_call", PendingToolCall{function_name, true})
               .second) {
          fail(path + ".function_call",
               "another legacy function call is still unresolved");
        }
      }
    }
    if (message.role == "tool") {
      std::string tool_call_id;
      std::string legacy_function_name;
      if (legacy_function_result) {
        tool_call_id = "function_call";
        if (!entry.contains("name"))
          fail(path + ".name", "is required for legacy function results");
        legacy_function_name = requireString(entry["name"], path + ".name");
      } else if (!entry.contains("tool_call_id")) {
        fail(path + ".tool_call_id", "is required for role tool");
      } else {
        tool_call_id =
          requireString(entry["tool_call_id"], path + ".tool_call_id");
      }
      const auto pending = pending_tool_calls.find(tool_call_id);
      if (pending == pending_tool_calls.end()) {
        fail(path + ".tool_call_id",
             "does not reference an earlier unresolved assistant tool call");
      }
      if (pending->second.legacy != legacy_function_result) {
        fail(path + ".role",
             pending->second.legacy
               ? "must use role function for a legacy function_call result"
               : "must use role tool for a tool_calls result");
      }
      if (legacy_function_result &&
          legacy_function_name != pending->second.function_name) {
        fail(path + ".name", "does not match the preceding function_call name");
      }
      pending_tool_calls.erase(pending);
    }
    if (entry.contains("name")) {
      const std::string name = requireString(entry["name"], path + ".name");
      if (!isValidFunctionName(name))
        fail(path + ".name", "is not a valid name");
    }

    if (!entry.contains("content") || entry["content"].is_null()) {
      if (message.role != "assistant" || !has_tool_history)
        fail(path + ".content", "is required");
    } else {
      message.content = parseContentParts(entry["content"], path + ".content");
      for (const auto &part : message.content) {
        if (part.type == ContentPartType::IMAGE_URL) {
          if (message.role != "user")
            fail(path + ".content", "image_url parts require role user");
          result.image_sources.push_back(part.image_url);
        }
      }
    }
    result.messages.push_back(std::move(message));
  }
  if (!pending_tool_calls.empty()) {
    fail("request.messages",
         "must resolve every assistant tool call before generation");
  }

  bool uses_legacy_functions = false;
  result.tools = parseTools(request, uses_legacy_functions);
  const ToolChoice choice = parseToolChoice(request, uses_legacy_functions);
  if (choice.kind == ToolChoiceKind::NONE) {
    result.tools.clear();
    result.original.erase("tools");
    result.original.erase("tool_choice");
    result.original.erase("functions");
    result.original.erase("function_call");
  }
  GrammarSelection tool_grammar = selectToolGrammar(result.tools, choice);
  GrammarSelection response_grammar = parseResponseFormat(request);

  // A forced/required tool response is the effective response shape. Auto
  // tools deliberately do not force a grammar, allowing response_format (if
  // present) or unconstrained text to govern the assistant response instead.
  result.grammar = tool_grammar.kind != GrammarKind::NONE
                     ? std::move(tool_grammar)
                     : std::move(response_grammar);
  std::sort(result.unsupported_fields.begin(), result.unsupported_fields.end());
  result.unsupported_fields.erase(std::unique(result.unsupported_fields.begin(),
                                              result.unsupported_fields.end()),
                                  result.unsupported_fields.end());
  return result;
}

Request parseRequest(const std::string &request_json) {
  try {
    return parseRequest(nlohmann::json::parse(request_json));
  } catch (const nlohmann::json::exception &e) {
    throw std::invalid_argument(std::string("request: invalid JSON: ") +
                                e.what());
  }
}

} // namespace openai
} // namespace causallm
