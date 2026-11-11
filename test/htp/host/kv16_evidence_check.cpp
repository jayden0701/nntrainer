// SPDX-License-Identifier: Apache-2.0
/** @file kv16_evidence_check.cpp
 * @brief Collect KV16 full-scope device evidence: DSP resident INT8 KV state
 *        (scales + decode state, standard checkpoint framing) and full logits
 *        rows, with SHA256 identity fields for compare_device_evidence.py.
 * Usage: kv16_evidence_check MODEL_DIR FIXTURE_JSON TRACE_JSONL OUTPUT_JSON
 *                            DUMP_DIR MODE
 * MODE is "baseline" (codec OFF) or "codec" (codec ON+VERIFY). The checkpoint
 * trace is written to TRACE_JSONL; the codec accounting for MODE=codec is read
 * back from it. Observation mapping (identical for both modes): the resident
 * session dump is the standard saveDeviceKVCache framing, whose INT8 K/V
 * payload, scale arrays and decode state are serialized together, so
 * k/v/seed_kv/seed_scales/decode_state hashes reference that one file.
 */
#include <algorithm>
#include <causal_lm.h>
#include <cstdint>
#include <cstring>
#include <factory.h>
#include <filesystem>
#include <fstream>
#include <htp_model_config.h>
#include <htp_runtime_config.h>
#include <iomanip>
#include <iostream>
#include <model_registry.h>
#include <sstream>
#include <sys/system_properties.h>

using json = causallm::json;
using Clock = std::chrono::steady_clock;

namespace {

/** Compact SHA256 (public-domain style, no external dependency). */
class Sha256 {
public:
  Sha256() { reset(); }
  void reset() {
    h_[0] = 0x6a09e667u, h_[1] = 0xbb67ae85u, h_[2] = 0x3c6ef372u,
    h_[3] = 0xa54ff53au, h_[4] = 0x510e527fu, h_[5] = 0x9b05688cu,
    h_[6] = 0x1f83d9abu, h_[7] = 0x5be0cd19u;
    len_ = 0;
    buf_used_ = 0;
  }
  void update(const void *data, size_t bytes) {
    const auto *p = static_cast<const unsigned char *>(data);
    len_ += bytes;
    while (bytes) {
      const size_t take = std::min(bytes, sizeof(buf_) - buf_used_);
      std::memcpy(buf_ + buf_used_, p, take);
      buf_used_ += take;
      p += take;
      bytes -= take;
      if (buf_used_ == sizeof(buf_))
        compress();
    }
  }
  std::string hex() {
    unsigned char digest[32];
    const uint64_t bits = len_ * 8u;
    const unsigned char one = 0x80u;
    update(&one, 1);
    const unsigned char zero = 0u;
    while (buf_used_ != 56)
      update(&zero, 1);
    unsigned char tail[8];
    for (int i = 0; i < 8; ++i)
      tail[i] = static_cast<unsigned char>(bits >> (56 - 8 * i));
    update(tail, 8);
    for (int i = 0; i < 8; ++i) {
      digest[i * 4] = h_[i] >> 24;
      digest[i * 4 + 1] = h_[i] >> 16;
      digest[i * 4 + 2] = h_[i] >> 8;
      digest[i * 4 + 3] = h_[i];
    }
    std::ostringstream text;
    text << std::hex << std::setfill('0');
    for (unsigned char c : digest)
      text << std::setw(2) << static_cast<unsigned>(c);
    return text.str();
  }

private:
  static uint32_t rotr(uint32_t v, int s) { return (v >> s) | (v << (32 - s)); }
  void compress() {
    static const uint32_t k[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
      0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
      0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
      0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
      0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
      0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
      0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
      0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t)buf_[i * 4] << 24 | (uint32_t)buf_[i * 4 + 1] << 16 |
             (uint32_t)buf_[i * 4 + 2] << 8 | buf_[i * 4 + 3];
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 =
        rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 =
        rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5],
             g = h_[6], hh = h_[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = hh + s1 + ch + k[i] + w[i];
      const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + maj;
      hh = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
    }
    h_[0] += a, h_[1] += b, h_[2] += c, h_[3] += d;
    h_[4] += e, h_[5] += f, h_[6] += g, h_[7] += hh;
    buf_used_ = 0;
  }
  uint32_t h_[8];
  uint64_t len_;
  size_t buf_used_;
  unsigned char buf_[64];
};

std::string sha256File(const std::filesystem::path &path) {
  Sha256 sha;
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Cannot read " + path.string());
  char buf[1 << 20];
  while (file.read(buf, sizeof(buf)) || file.gcount())
    sha.update(buf, static_cast<size_t>(file.gcount()));
  return sha.hex();
}

std::string sha256Text(const std::string &text) {
  Sha256 sha;
  sha.update(text.data(), text.size());
  return sha.hex();
}

/** Hash a resident-session checkpoint dump while skipping the two fields
 *  that are intentionally run-unique: the snapshot cookie (header words 2-3,
 *  byte range [8,16)) and the trailing FNV trailer (last 8 bytes), which
 *  covers the cookie. Everything else in the framing is state-derived. */
std::string sha256KvDump(const std::filesystem::path &path) {
  Sha256 sha;
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("Cannot read " + path.string());
  const std::streamoff size = file.tellg();
  if (size <= 24)
    throw std::runtime_error("KV dump too small: " + path.string());
  file.seekg(0);
  char buf[1 << 20];
  std::streamoff done = 0;
  while (done < size) {
    std::streamoff take = std::min<size_t>(sizeof(buf), size - done);
    // never straddle a skipped range; the skipped ranges are at the edges
    if (done < 16 && done + take > 16)
      take = 16 - done; // header prefix [0,16) contains the cookie at [8,16)
    file.read(buf, take);
    const std::streamoff got = file.gcount();
    const std::streamoff begin = done, end = done + got;
    // hash [begin, end) minus [8,16) and minus the trailing 8 bytes
    std::streamoff lo = begin, hi = end;
    std::streamoff skipLo = 8, skipHi = 16;
    auto emit = [&](std::streamoff a, std::streamoff b) {
      if (a >= b)
        return;
      sha.update(buf + (a - begin), static_cast<size_t>(b - a));
    };
    // order the pieces: [lo, min(hi,skipLo)), (skip), [max(lo,skipHi), hi),
    // then the file tail skip is handled by clamping hi to size-8 below.
    std::streamoff tail = size - 8;
    std::streamoff pieceEnd = std::min(hi, tail);
    emit(lo, std::min(pieceEnd, skipLo));
    emit(std::max(lo, skipHi), pieceEnd);
    done += got;
  }
  return sha.hex();
}

/** Full-logits capture: the DSP returns the whole FP32 row whenever a
 *  LogitsProcessor is installed; we dump the FIRST decode step of each run()
 *  call. Baseline and codec must capture the same (run, step) rows. */
class EvidenceLogitsProcessor : public causallm::LogitsProcessor {
public:
  void beginRun(unsigned index, const std::filesystem::path &dir) {
    run_ = index;
    dir_ = dir;
    captured_ = false;
  }
  void process(float *logits, unsigned int vocab_size, unsigned int) override {
    if (captured_)
      return;
    captured_ = true;
    const auto path = dir_ / ("logits-" + std::to_string(run_) + ".f32");
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.write(reinterpret_cast<const char *>(logits),
               static_cast<std::streamsize>(vocab_size) * sizeof(float));
    vocab_ = vocab_size;
  }
  void acceptToken(unsigned int, unsigned int) override {}
  bool captured() const { return captured_; }
  unsigned int vocab() const { return vocab_; }

private:
  unsigned run_ = 0;
  unsigned vocab_ = 0;
  bool captured_ = false;
  std::filesystem::path dir_;
};

json read(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("Cannot read " + path.string());
  json value;
  file >> value;
  return value;
}

void require(bool value, const std::string &message) {
  if (!value)
    throw std::runtime_error(message);
}

/** Hash the FP32 calibration bit arrays of every checkpoint-trace event that
 *  appeared since the last call, in file order. The trace serializes the
 *  calibration as exact u32 bit words (never decimal floats), so the hash is
 *  a bit-exact observation of the quantization state. */
std::string hashNewCalibration(const std::filesystem::path &trace,
                               std::streamoff &offset) {
  Sha256 sha;
  std::error_code ec;
  if (!std::filesystem::exists(trace, ec))
    return {}; // no capture events yet: no observation
  std::ifstream file(trace, std::ios::binary);
  if (!file)
    throw std::runtime_error("Cannot read trace " + trace.string());
  file.seekg(offset);
  std::string line;
  bool any = false;
  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    const auto record = json::parse(line);
    if (record.value("schema", "") != "nntrainer.kv16.checkpoint-trace.v1")
      continue;
    any = true;
    offset = offset + static_cast<std::streamoff>(line.size()) + 1;
    const auto &calibration = record.at("calibration");
    const uint32_t count =
      static_cast<uint32_t>(calibration.is_array() ? calibration.size() : 0u);
    sha.update(&count, sizeof(count));
    for (const auto &state : calibration) {
      const unsigned char calibrated =
        state.at("calibrated").get<bool>() ? 1 : 0;
      sha.update(&calibrated, 1);
      for (const char *key : {"scale_k", "scale_v", "q_enc", "out_enc", "kmax",
                              "vmax", "floor_k", "floor_v"}) {
        const auto &words = state.at(key);
        for (const auto &word : words) {
          const uint32_t bits = word.get<uint32_t>();
          sha.update(&bits, sizeof(bits));
        }
      }
    }
  }
  return any ? sha.hex() : std::string{};
}

std::string prop(const char *name) {
  char value[92] = {};
  __system_property_get(name, value);
  return value;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 7)
    return 2;
  const std::filesystem::path dir = argv[1];
  const std::filesystem::path fixturePath = argv[2];
  const std::filesystem::path tracePath = argv[3];
  const std::filesystem::path outputPath = argv[4];
  const std::filesystem::path dumpDir = argv[5];
  const std::string mode = argv[6];
  if (mode != "baseline" && mode != "codec")
    return 2;
  const bool codec = mode == "codec";
  try {
    require(std::filesystem::exists(dumpDir) ||
              std::filesystem::create_directories(dumpDir),
            "Cannot create dump dir");
    // The codec flag is read at the first cache allocation; VERIFY
    // byte-compares the encode/decode round trip before publication.
    setenv("NNTR_HTP_PREFIX_KV16", codec ? "1" : "0", 1);
    setenv("NNTR_HTP_PREFIX_KV16_VERIFY", codec ? "1" : "0", 1);
    setenv("NNTR_HTP_PREFIX_KV16_TRACE", tracePath.c_str(), 1);
    std::error_code traceEc;
    std::filesystem::remove(tracePath, traceEc);

    auto cfg = read(dir / "config.json");
    auto gen = read(dir / "generation_config.json");
    auto nc = read(dir / "nntr_config.json");
    const auto fixture = read(fixturePath);
    nntrainer::configureHtpRuntime(causallm::normalizeHtpRuntimeConfig(nc));
    causallm::registerAllModels();
    auto model =
      causallm::Factory::Instance().create("Gemma4ForCausalLM", cfg, gen, nc);
    model->initialize();
    model->load_weight(nc["model_file_name"].get<std::string>());
    model->repack_weight();
    auto *c = dynamic_cast<causallm::CausalLM *>(model.get());
    require(c != nullptr, "CausalLM unavailable");
    // Device radix prefix capture is what the codec hooks into; without it
    // there are no checkpoint events and no codec accounting to compare.
    c->enableDeviceKVManager(8, 8);

    EvidenceLogitsProcessor processor;
    c->setLogitsProcessor(&processor);

    json report = {{"schema", "nntrainer.kv16.device-evidence.v1"},
                   {"mode", mode},
                   {"hardware_validated", true},
                   {"config", nc},
                   {"runs", json::array()}};

    std::streamoff traceOffset = 0;
    json events = json::array();
    unsigned runIndex = 0;
    auto run = [&](const std::string &text, bool reset,
                   const std::string &name) {
      if (reset)
        c->setSessionTokenPosition(0);
      processor.beginRun(runIndex, dumpDir);
      const auto start = Clock::now();
      model->run(text, false, "", "", false);
      const auto &m = c->getPerformanceMetrics();
      require(!m.prompt_truncated_tokens, "Prompt truncated");
      require(processor.captured(), "No logits captured for run " + name);
      const std::string logitsFile =
        "logits-" + std::to_string(runIndex) + ".f32";
      const auto calibration = hashNewCalibration(tracePath, traceOffset);
      json event = {
        {"name", name},
        {"position", c->getKvLen()},
        {"prefill_tokens", m.prefill_tokens},
        {"call_plan_sha256", sha256Text("kv16-call-plan-v1\0" + text +
                                        std::to_string(m.prefill_tokens))},
        {"logits_sha256", sha256File(dumpDir / logitsFile)}};
      event["token_ids"] = c->getLastGeneratedTokenIds();
      // A run only produces an evidence observation when it actually went
      // through device prefix capture (fresh prefill of >= 64 tokens). The
      // warm-up and the query continuation are continuations or too short,
      // so they carry no calibration and no dump.
      if (!calibration.empty()) {
        const std::string kvFile =
          "kv-run-" + std::to_string(runIndex) + ".bin";
        c->saveDeviceKVCache((dumpDir / kvFile).string(), false);
        const auto kvSha = sha256KvDump(dumpDir / kvFile);
        event["calibration_sha256"] = calibration;
        event["id"] = "run-" + std::to_string(runIndex);
        event["kv_file"] = kvFile;
        // The resident session framing serializes the DSP INT8 K/V payload,
        // its six scale arrays and the decode state together, so these five
        // observations reference the same byte-exact dump by design.
        event["k_sha256"] = kvSha;
        event["v_sha256"] = kvSha;
        event["seed_kv_sha256"] = kvSha;
        event["seed_scales_sha256"] = kvSha;
        event["decode_state_sha256"] = kvSha;
        events.push_back(event);
      }
      report["runs"].push_back(
        {{"name", name},
         {"ids", c->getLastGeneratedTokenIds()},
         {"position", c->getKvLen()},
         {"prefill_tokens", m.prefill_tokens},
         {"ttft_ms", m.ttft_ms},
         {"decode_steady_ms", m.decode_steady_ms},
         {"decode_steady_tokens", m.decode_steady_tokens},
         {"kv_bytes", c->getDeviceKVCacheBytes()},
         {"wall_ms",
          std::chrono::duration<double, std::milli>(Clock::now() - start)
            .count()}});
      ++runIndex;
    };

    run("<bos><|turn>user\nSay READY.<turn|>\n<|turn>model\n", true, "warm");
    for (const auto &seed : fixture["seeds"])
      // Every seed opens an independent request: without the reset the
      // resident session turns the run into a continuation, which is
      // exactly the path that skips device prefix capture.
      run(seed["prompt"].get<std::string>(), true, "seed");
    run(fixture["query"].get<std::string>(), false, "final");

    json identity = {
      {"model_sha256", sha256File(nc["model_file_name"].get<std::string>())},
      {"tokenizer_sha256", sha256File(dir / "tokenizer.json")},
      {"numerical_config_sha256", sha256File(dir / "nntr_config.json")},
      {"device_fingerprint", prop("ro.product.model") + "/" +
                               prop("ro.soc.model") + "/android" +
                               prop("ro.build.version.release")},
      {"code_revision", [] {
         const char *revision = std::getenv("NNTR_KV16_CODE_REVISION");
         return revision && *revision ? revision : "unknown";
       }()}};
    const auto exeDir =
      std::filesystem::canonical("/proc/self/exe").parent_path();
    identity["arm_sha256"] = sha256File(exeDir / "libnntrainer.so");
    identity["dsp_sha256"] = sha256File(exeDir / "libnntr_hvx_skel.so");
    report["identity"] = identity;
    report["events"] = events;
    if (codec) {
      // Codec accounting over every checkpoint event of THIS run: the
      // compressed page count must be nonzero for the codec claim.
      size_t compressed = 0, raw = 0, stored = 0;
      std::ifstream trace(tracePath);
      std::string line;
      while (std::getline(trace, line)) {
        if (line.empty())
          continue;
        const auto record = json::parse(line);
        if (record.value("schema", "") != "nntrainer.kv16.checkpoint-trace.v1")
          continue;
        const auto &codecInfo = record.at("codec");
        compressed += codecInfo.at("compressed_pages").get<size_t>();
        raw += codecInfo.at("raw_bytes").get<size_t>();
        stored += codecInfo.at("stored_bytes").get<size_t>();
      }
      report["codec"] = {{"compressed_pages", compressed},
                         {"raw_bytes", raw},
                         {"stored_bytes", stored}};
    }
    {
      std::ofstream file(outputPath);
      file << report.dump(2) << '\n';
      file.close();
      require(static_cast<bool>(file), "Cannot write " + outputPath.string());
    }
    c->resetLogitsProcessor();
    std::cerr << "EVIDENCE written: " << outputPath << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "EVIDENCE_ERROR: " << error.what() << std::endl;
    return 1;
  }
}
