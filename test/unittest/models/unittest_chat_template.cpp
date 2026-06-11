// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_chat_template.cpp
 * @date   11 Jun 2026
 * @brief  Unit tests for Hugging Face chat template loading
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 */

#include <chat_template.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

namespace {

std::string sanitizePathComponent(std::string value) {
  std::replace_if(
    value.begin(), value.end(),
    [](unsigned char ch) { return !std::isalnum(ch) && ch != '_'; }, '_');
  return value;
}

class ChatTemplateTest : public ::testing::Test {
protected:
  void SetUp() override {
    const auto test_info =
      ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string suite_name =
      test_info != nullptr ? test_info->test_suite_name() : "unknown_suite";
    const std::string test_name =
      test_info != nullptr ? test_info->name() : "unknown_test";
    const auto unique_suffix =
      std::chrono::steady_clock::now().time_since_epoch().count();
    test_dir = std::filesystem::temp_directory_path() /
               ("nntrainer_chat_template_" +
                sanitizePathComponent(suite_name) + "_" +
                sanitizePathComponent(test_name) + "_" +
                std::to_string(unique_suffix));
    std::filesystem::remove_all(test_dir);
    std::filesystem::create_directories(test_dir);
  }

  void TearDown() override { std::filesystem::remove_all(test_dir); }

  void writeTokenizerConfig(const nlohmann::json &config) const {
    std::ofstream out(test_dir / "tokenizer_config.json");
    out << config.dump(2);
  }

  std::filesystem::path test_dir;
};

std::string templateWithToolCallProbe(const std::string &literal) {
  return "{% for message in messages %}"
         "{% if message.tool_calls %}"
         "{% for tool_call in message.tool_calls %}"
         "{{ tool_call.function.name }}"
         "{{ tool_call.function.arguments }}"
         "{% endfor %}"
         "{% endif %}"
         "{% endfor %}" +
         literal;
}

TEST_F(ChatTemplateTest, array_chat_template_default_renders) {
  writeTokenizerConfig({
    {"chat_template",
     {{{"name", "default"},
       {"template", templateWithToolCallProbe("ARRAY_DEFAULT")}}}},
  });

  causallm::ChatTemplate chat_template =
    causallm::ChatTemplate::Load(test_dir.string());

  const std::string rendered = chat_template.apply(
    {{"messages", {{{"role", "user"}, {"content", "hi"}}}}});

  EXPECT_EQ(rendered, "ARRAY_DEFAULT");
}

TEST_F(ChatTemplateTest, array_chat_template_selects_tool_use_for_tools) {
  writeTokenizerConfig({
    {"chat_template",
     {{{"name", "default"},
       {"template", templateWithToolCallProbe("ARRAY_DEFAULT")}},
      {{"name", "tool_use"},
       {"template", templateWithToolCallProbe("ARRAY_TOOL")}}}},
  });

  causallm::ChatTemplate chat_template =
    causallm::ChatTemplate::Load(test_dir.string());

  const std::string rendered = chat_template.apply(
    {{"messages", {{{"role", "user"}, {"content", "hi"}}}},
     {"tools", {{{"type", "function"},
                  {"function", {{"name", "lookup"}}}}}}});

  EXPECT_EQ(rendered, "ARRAY_TOOL");
}

TEST_F(ChatTemplateTest, array_chat_template_selects_explicit_template_name) {
  writeTokenizerConfig({
    {"chat_template",
     {{{"name", "default"},
       {"template", templateWithToolCallProbe("ARRAY_DEFAULT")}},
      {{"name", "custom"},
       {"template", templateWithToolCallProbe("ARRAY_CUSTOM")}}}},
  });

  causallm::ChatTemplate chat_template =
    causallm::ChatTemplate::Load(test_dir.string());

  causallm::ChatTemplate::Options options;
  options.template_name = "custom";
  const std::string rendered =
    chat_template.apply({{"messages",
                          {{{"role", "user"}, {"content", "hi"}}}}},
                        options);

  EXPECT_EQ(rendered, "ARRAY_CUSTOM");
}

TEST_F(ChatTemplateTest, string_chat_template_still_renders) {
  writeTokenizerConfig({
    {"chat_template", templateWithToolCallProbe("STRING_TEMPLATE")},
  });

  causallm::ChatTemplate chat_template =
    causallm::ChatTemplate::Load(test_dir.string());

  const std::string rendered = chat_template.apply(
    {{"messages", {{{"role", "user"}, {"content", "hi"}}}}});

  EXPECT_EQ(rendered, "STRING_TEMPLATE");
}

TEST_F(ChatTemplateTest, object_chat_template_still_renders) {
  writeTokenizerConfig({
    {"chat_template",
     {{"default", templateWithToolCallProbe("OBJECT_DEFAULT")}}},
  });

  causallm::ChatTemplate chat_template =
    causallm::ChatTemplate::Load(test_dir.string());

  const std::string rendered = chat_template.apply(
    {{"messages", {{{"role", "user"}, {"content", "hi"}}}}});

  EXPECT_EQ(rendered, "OBJECT_DEFAULT");
}

TEST_F(ChatTemplateTest, array_chat_template_rejects_missing_template) {
  writeTokenizerConfig({
    {"chat_template", {{{"name", "default"}}}},
  });

  try {
    static_cast<void>(causallm::ChatTemplate::Load(test_dir.string()));
    FAIL() << "Expected ChatTemplate::Load to reject malformed array entry";
  } catch (const std::runtime_error &e) {
    const std::string message = e.what();
    EXPECT_NE(message.find(
                "tokenizer_config.chat_template array entries must include "
                "string name and template"),
              std::string::npos)
      << message;
  }
}

TEST_F(ChatTemplateTest, array_chat_template_rejects_non_object_entry) {
  writeTokenizerConfig({
    {"chat_template", {"not-an-object"}},
  });

  EXPECT_THROW(static_cast<void>(causallm::ChatTemplate::Load(test_dir.string())),
               std::runtime_error);
}

TEST_F(ChatTemplateTest, array_chat_template_rejects_non_string_name) {
  writeTokenizerConfig({
    {"chat_template", {{{"name", 7}, {"template", "ignored"}}}},
  });

  EXPECT_THROW(static_cast<void>(causallm::ChatTemplate::Load(test_dir.string())),
               std::runtime_error);
}

TEST_F(ChatTemplateTest, array_chat_template_rejects_non_string_template) {
  writeTokenizerConfig({
    {"chat_template", {{{"name", "default"}, {"template", 7}}}},
  });

  EXPECT_THROW(static_cast<void>(causallm::ChatTemplate::Load(test_dir.string())),
               std::runtime_error);
}

} // namespace
