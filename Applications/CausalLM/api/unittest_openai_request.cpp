// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_openai_request.cpp
 * @brief  Unit tests for OpenAI request parsing and chat content rendering
 * @author Quick.AI Team
 * @bug    No known bugs except for NYI items
 */

#include "api/openai_request.h"
#include "chat_template.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

using causallm::openai::GrammarKind;
using causallm::openai::parseRequest;
using json = nlohmann::json;

json basicRequest() {
  return {{"messages", {{{"role", "user"}, {"content", "hello"}}}}};
}

TEST(OpenAIRequest, RequiresObjectAndMessages) {
  EXPECT_THROW(parseRequest(json::array()), std::invalid_argument);
  EXPECT_THROW(parseRequest(json::object()), std::invalid_argument);

  json request = {{"messages", json::array()}};
  EXPECT_THROW(parseRequest(request), std::invalid_argument);
}

TEST(OpenAIRequest, ReportsUnsupportedGenerationControls) {
  json request = basicRequest();
  request["temperature"] = 0.5;
  request["max_completion_tokens"] = 32;

  const auto parsed = parseRequest(request);
  EXPECT_EQ(parsed.unsupported_fields,
            (std::vector<std::string>{"max_completion_tokens", "temperature"}));
}

TEST(OpenAIRequest, RejectsParallelToolCallsExplicitly) {
  json request = basicRequest();
  request["parallel_tool_calls"] = true;
  EXPECT_EQ(parseRequest(request).unsupported_fields,
            (std::vector<std::string>{"parallel_tool_calls"}));

  request["parallel_tool_calls"] = false;
  EXPECT_TRUE(parseRequest(request).unsupported_fields.empty());
}

TEST(OpenAIRequest, ValidatesAcceptedMetadataFields) {
  json request = basicRequest();
  request["model"] = "local-model";
  request["stream"] = true;
  request["metadata"] = {{"trace", "abc"}};
  request["store"] = false;
  EXPECT_TRUE(parseRequest(request).unsupported_fields.empty());

  request["stream"] = "yes";
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request = basicRequest();
  request["store"] = true;
  EXPECT_EQ(parseRequest(request).unsupported_fields,
            (std::vector<std::string>{"store"}));
}

TEST(OpenAIRequest, PreservesOrderedTextAndImageParts) {
  json request = json::parse(R"json(
    {
      "messages": [{
        "role": "user",
        "content": [
          {"type": "text", "text": "before:"},
          {"type": "image_url", "image_url": {"url": "media://one", "detail": "high"}},
          {"type": "text", "text": ":between:"},
          {"type": "image_url", "image_url": "media://two"},
          {"type": "text", "text": ":after"}
        ]
      }]
    }
  )json");

  auto parsed = parseRequest(request);
  ASSERT_EQ(parsed.messages.size(), 1u);
  ASSERT_EQ(parsed.messages[0].content.size(), 5u);
  EXPECT_EQ(parsed.messages[0].content[1].image_url, "media://one");
  EXPECT_EQ(parsed.messages[0].content[1].image_detail, "high");
  EXPECT_EQ(parsed.image_sources,
            (std::vector<std::string>{"media://one", "media://two"}));
  EXPECT_EQ(causallm::openai::renderContentWithImagePlaceholders(
              parsed.messages[0].content),
            "before:<|image|>:between:<|image|>:after");
}

TEST(OpenAIRequest, RejectsMalformedContentParts) {
  json request = basicRequest();
  request["messages"][0]["content"] =
    json::array({{{"type", "image_url"}, {"image_url", json::object()}}});
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request["messages"][0]["content"] =
    json::array({{{"type", "audio"}, {"audio", "media://audio"}}});
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request["messages"][0]["content"] = json::array(
    {{{"type", "image_url"},
      {"image_url", {{"url", "media://image"}, {"detail", "full"}}}}});
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request = basicRequest();
  request["messages"][0]["role"] = "assistant";
  request["messages"][0]["content"] =
    json::array({{{"type", "image_url"}, {"image_url", "media://image"}}});
  EXPECT_THROW(parseRequest(request), std::invalid_argument);
}

TEST(OpenAIRequest, RejectsMixedModernAndLegacyToolControls) {
  json request = basicRequest();
  request["tools"] =
    json::array({{{"type", "function"}, {"function", {{"name", "lookup"}}}}});
  request["function_call"] = "auto";
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request = basicRequest();
  request["functions"] = json::array({{{"name", "lookup"}}});
  request["tool_choice"] = "auto";
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request = basicRequest();
  request["function_call"] = "none";
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request = basicRequest();
  request["tool_choice"] = "auto";
  EXPECT_THROW(parseRequest(request), std::invalid_argument);
}

TEST(OpenAIRequest, RequiresModernFunctionDiscriminators) {
  json request = basicRequest();
  request["tools"] = json::array({{{"function", {{"name", "lookup"}}}}});
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request["tools"][0]["type"] = "function";
  request["tool_choice"] = {{"function", {{"name", "lookup"}}}};
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request["tool_choice"]["type"] = "function";
  EXPECT_NO_THROW(parseRequest(request));
}

TEST(OpenAIRequest, ValidatesFunctionDefinitionFields) {
  json request = basicRequest();
  request["tools"] =
    json::array({{{"type", "function"},
                  {"function", {{"name", "not valid"}, {"strict", true}}}}});
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request["tools"][0]["function"]["name"] = "valid-name_1";
  request["tools"][0]["function"]["strict"] = "yes";
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request["tools"][0]["function"]["strict"] = true;
  EXPECT_NO_THROW(parseRequest(request));
}

TEST(OpenAIRequest, ValidatesAssistantToolHistory) {
  json request = basicRequest();
  request["messages"].push_back(
    {{"role", "assistant"},
     {"content", nullptr},
     {"tool_calls",
      json::array(
        {{{"id", "call_1"},
          {"type", "function"},
          {"function", {{"name", "lookup"}, {"arguments", "{\"q\":1}"}}}}})}});
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request["messages"].push_back(
    {{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "result"}});
  EXPECT_NO_THROW(parseRequest(request));

  request["messages"][1]["tool_calls"][0]["function"].erase("arguments");
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request = basicRequest();
  request["messages"].push_back(
    {{"role", "assistant"},
     {"content", nullptr},
     {"tool_calls",
      json::array(
        {{{"id", "call_1"},
          {"function", {{"name", "lookup"}, {"arguments", "{}"}}}}})}});
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request = basicRequest();
  request["messages"].push_back({{"role", "tool"}, {"content", "result"}});
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request = basicRequest();
  request["messages"].push_back(
    {{"role", "tool"}, {"tool_call_id", "missing"}, {"content", "result"}});
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request = basicRequest();
  request["messages"].push_back(
    {{"role", "assistant"},
     {"content", nullptr},
     {"tool_calls",
      json::array(
        {{{"id", "call_1"},
          {"type", "function"},
          {"function", {{"name", "lookup"}, {"arguments", "{}"}}}}})}});
  request["messages"].push_back({{"role", "user"}, {"content", "skip"}});
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request = basicRequest();
  request["messages"].push_back(
    {{"role", "assistant"},
     {"content", nullptr},
     {"tool_calls",
      json::array(
        {{{"id", "call_1"},
          {"type", "function"},
          {"function", {{"name", "first"}, {"arguments", "{}"}}}},
         {{"id", "call_2"},
          {"type", "function"},
          {"function", {{"name", "second"}, {"arguments", "{}"}}}}})}});
  request["messages"].push_back({{"role", "tool"},
                                 {"tool_call_id", "call_2"},
                                 {"content", "second result"}});
  request["messages"].push_back({{"role", "tool"},
                                 {"tool_call_id", "call_1"},
                                 {"content", "first result"}});
  request["messages"].push_back({{"role", "user"}, {"content", "continue"}});

  EXPECT_NO_THROW(parseRequest(request));
}

TEST(OpenAIRequest, NormalizesLegacyFunctionResultHistory) {
  json request = basicRequest();
  request["messages"].push_back(
    {{"role", "assistant"},
     {"content", nullptr},
     {"function_call", {{"name", "lookup"}, {"arguments", "{}"}}}});
  request["messages"].push_back(
    {{"role", "function"}, {"name", "lookup"}, {"content", "result"}});

  const auto parsed = parseRequest(request);
  EXPECT_EQ(parsed.messages.back().role, "tool");
  EXPECT_EQ(parsed.original["messages"].back()["role"], "tool");
  EXPECT_EQ(parsed.original["messages"].back()["tool_call_id"],
            "function_call");

  request["messages"].back()["name"] = "different";
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request["messages"].back()["name"] = "lookup";
  request["messages"].back()["role"] = "tool";
  request["messages"].back()["tool_call_id"] = "function_call";
  EXPECT_THROW(parseRequest(request), std::invalid_argument);
}

TEST(OpenAIRequest, ExtractsJsonSchemaResponseFormat) {
  json request = basicRequest();
  request["response_format"] = json::parse(R"json(
    {
      "type": "json_schema",
      "json_schema": {
        "name": "answer",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {"answer": {"type": "string"}},
          "required": ["answer"]
        }
      }
    }
  )json");

  auto parsed = parseRequest(request);
  EXPECT_EQ(parsed.grammar.kind, GrammarKind::JSON_SCHEMA);
  EXPECT_EQ(parsed.grammar.schema["type"], "object");
  EXPECT_EQ(parsed.grammar.schema["required"], json::array({"answer"}));
}

TEST(OpenAIRequest, ValidatesJsonSchemaWrapper) {
  json request = basicRequest();
  request["response_format"] = {
    {"type", "json_schema"},
    {"json_schema", {{"schema", {{"type", "object"}}}}}};
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request["response_format"]["json_schema"]["name"] = "answer";
  request["response_format"]["json_schema"]["strict"] = "yes";
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request["response_format"]["json_schema"]["strict"] = true;
  request["response_format"]["json_schema"]["description"] = 42;
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request["response_format"]["json_schema"]["description"] = "answer data";
  request["response_format"]["json_schema"]["name"] = "not valid";
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request["response_format"]["json_schema"]["name"] = "answer";
  request["response_format"]["json_schema"]["schema"] = true;
  EXPECT_THROW(parseRequest(request), std::invalid_argument);

  request["response_format"]["json_schema"].erase("schema");
  const auto parsed = parseRequest(request);
  EXPECT_EQ(parsed.grammar.kind, GrammarKind::JSON_SCHEMA);
  EXPECT_EQ(parsed.grammar.schema, json::object());
}

TEST(OpenAIRequest, SelectsBuiltinJsonObjectGrammar) {
  json request = basicRequest();
  request["response_format"] = {{"type", "json_object"}};

  EXPECT_EQ(parseRequest(request).grammar.kind, GrammarKind::JSON_OBJECT);
}

TEST(OpenAIRequest, AutoToolsDoNotForceGrammar) {
  json request = basicRequest();
  request["tools"] = json::array(
    {{{"type", "function"},
      {"function",
       {{"name", "weather"}, {"parameters", {{"type", "object"}}}}}}});
  request["tool_choice"] = "auto";

  auto parsed = parseRequest(request);
  ASSERT_EQ(parsed.tools.size(), 1u);
  EXPECT_EQ(parsed.grammar.kind, GrammarKind::NONE);
}

TEST(OpenAIRequest, NoneToolChoiceRemovesToolsFromRenderedRequest) {
  json request = basicRequest();
  request["tools"] = json::array(
    {{{"type", "function"},
      {"function",
       {{"name", "weather"}, {"parameters", {{"type", "object"}}}}}}});
  request["tool_choice"] = "none";

  const auto parsed = parseRequest(request);
  EXPECT_TRUE(parsed.tools.empty());
  EXPECT_EQ(parsed.grammar.kind, GrammarKind::NONE);
  EXPECT_FALSE(parsed.original.contains("tools"));
  EXPECT_FALSE(parsed.original.contains("tool_choice"));
}

TEST(OpenAIRequest, AutoToolsAllowResponseFormatGrammar) {
  json request = basicRequest();
  request["tools"] = json::array(
    {{{"type", "function"},
      {"function",
       {{"name", "weather"}, {"parameters", {{"type", "object"}}}}}}});
  request["tool_choice"] = "auto";
  request["response_format"] = {{"type", "json_object"}};

  EXPECT_EQ(parseRequest(request).grammar.kind, GrammarKind::JSON_OBJECT);
}

TEST(OpenAIRequest, ForcedToolSelectsItsParameterSchema) {
  json request = basicRequest();
  request["tools"] = json::parse(R"json(
    [
      {
        "type": "function",
        "function": {
          "name": "weather",
          "parameters": {
            "type": "object",
            "required": ["city"],
            "properties": {"city": {"type": "string"}}
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "clock",
          "parameters": {
            "type": "object",
            "required": ["timezone"]
          }
        }
      }
    ]
  )json");
  request["tool_choice"] = {{"type", "function"},
                            {"function", {{"name", "clock"}}}};

  auto parsed = parseRequest(request);
  EXPECT_EQ(parsed.grammar.kind, GrammarKind::TOOL_CALL);
  ASSERT_TRUE(parsed.grammar.forced_tool_name.has_value());
  EXPECT_EQ(*parsed.grammar.forced_tool_name, "clock");
  ASSERT_EQ(parsed.grammar.tools.size(), 1u);
  EXPECT_EQ(parsed.grammar.schema["properties"]["name"]["const"], "clock");
  EXPECT_EQ(parsed.grammar.schema["properties"]["arguments"]["required"],
            json::array({"timezone"}));
}

TEST(OpenAIRequest, ToolWithoutParametersAcceptsOnlyEmptyArguments) {
  json request = basicRequest();
  request["tools"] =
    json::array({{{"type", "function"}, {"function", {{"name", "ping"}}}}});
  request["tool_choice"] = {{"type", "function"},
                            {"function", {{"name", "ping"}}}};

  const auto parsed = parseRequest(request);
  const auto &arguments = parsed.grammar.schema["properties"]["arguments"];
  EXPECT_EQ(arguments["type"], "object");
  EXPECT_EQ(arguments["properties"], json::object());
  EXPECT_EQ(arguments["additionalProperties"], false);
}

TEST(OpenAIRequest, RequiredToolsProduceOneOfSchema) {
  json request = basicRequest();
  request["tools"] = json::parse(R"json(
    [
      {
        "type": "function",
        "function": {
          "name": "first",
          "parameters": {"type": "object", "required": ["a"]}
        }
      },
      {
        "type": "function",
        "function": {
          "name": "second",
          "parameters": {"type": "object", "required": ["b"]}
        }
      }
    ]
  )json");
  request["tool_choice"] = "required";

  auto parsed = parseRequest(request);
  EXPECT_EQ(parsed.grammar.kind, GrammarKind::TOOL_CALL);
  EXPECT_FALSE(parsed.grammar.forced_tool_name.has_value());
  ASSERT_EQ(parsed.grammar.tools.size(), 2u);
  ASSERT_TRUE(parsed.grammar.schema.contains("oneOf"));
  EXPECT_EQ(parsed.grammar.schema["oneOf"].size(), 2u);
  EXPECT_EQ(parsed.grammar.schema["oneOf"][0]["properties"]["name"]["const"],
            "first");
  EXPECT_EQ(parsed.grammar.schema["oneOf"][1]["properties"]["name"]["const"],
            "second");
}

TEST(OpenAIRequest, LegacyForcedFunctionSelectsSchema) {
  json request = basicRequest();
  request["functions"] = json::array(
    {{{"name", "lookup"},
      {"parameters", {{"type", "object"}, {"required", {"query"}}}}}});
  request["function_call"] = {{"name", "lookup"}};

  auto parsed = parseRequest(request);
  EXPECT_EQ(parsed.grammar.kind, GrammarKind::TOOL_CALL);
  ASSERT_TRUE(parsed.grammar.forced_tool_name.has_value());
  EXPECT_EQ(*parsed.grammar.forced_tool_name, "lookup");
  EXPECT_EQ(parsed.grammar.schema["properties"]["name"]["const"], "lookup");
  EXPECT_EQ(parsed.grammar.schema["properties"]["arguments"]["required"],
            json::array({"query"}));
}

class TemporaryChatTemplate {
public:
  TemporaryChatTemplate() {
    const auto suffix =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("nntrainer_openai_request_" + std::to_string(suffix));
    std::filesystem::create_directories(path_);
    std::ofstream file(path_ / "chat_template.jinja");
    file << "{% for message in messages %}{{ message['role'] }}="
            "{{ message['content'] }}{% endfor %}";
  }

  ~TemporaryChatTemplate() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  std::string path() const { return path_.string(); }

private:
  std::filesystem::path path_;
};

TEST(ChatTemplate, RendersOpenAIImagePlaceholdersInContentOrder) {
  TemporaryChatTemplate fixture;
  auto chat_template = causallm::ChatTemplate::Load(fixture.path());
  json request = json::parse(R"json(
    {
      "messages": [{
        "role": "user",
        "content": [
          {"type": "text", "text": "A"},
          {"type": "image_url", "image_url": {"url": "media://one"}},
          {"type": "text", "text": "B"},
          {"type": "image_url", "image_url": {"url": "media://two"}},
          {"type": "text", "text": "C"}
        ]
      }],
      "add_generation_prompt": false
    }
  )json");

  EXPECT_EQ(chat_template.apply(request), "user=A<|image|>B<|image|>C");
}

} // namespace
