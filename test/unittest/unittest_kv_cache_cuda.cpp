// SPDX-License-Identifier: Apache-2.0
/**
 * @file unittest_kv_cache_cuda.cpp
 * @brief Actual CUDA KV allocation, checked publication and lifetime tests.
 * @author Codex (OpenAI)
 * @bug Covers eager single-stream execution; model and replay are separate.
 */
#include <cuda_context_manager.h>
#include <cuda_kv_storage.h>
#include <cuda_mem_allocator.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#include <gtest/gtest.h>
#include <kv_cache/kv_cache_manager.h>
#include <vector>

using namespace nntrainer;
using namespace nntrainer::kv;

namespace {
class CudaKv : public ::testing::Test {
protected:
  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
      GTEST_SKIP() << "No CUDA device";
  }
};
CacheSpec geometry() {
  LayerSpec layer;
  layer.num_kv_heads = 2;
  layer.head_dim = 8;
  layer.max_tokens = layer.capacity = 8;
  return {{layer}};
}
} // namespace

TEST_F(CudaKv, allocationDescriptorIncludesSlicesAndExpiresOnFree) {
  CudaMemAllocator allocator(true);
  void *pointer = nullptr;
  allocator.alloc(&pointer, 256, 64);
  auto info = allocator.describe(static_cast<char *>(pointer) + 128);
  ASSERT_EQ(info.kind, AllocationKind::CudaDevice);
  EXPECT_EQ(info.base, pointer);
  EXPECT_EQ(info.bytes, 256u);
  EXPECT_GE(info.alignment(), 64u);
  EXPECT_NE(info.context, 0u);
  EXPECT_EQ(info.device, cuda::ContextManager::Global().GetDeviceOrdinal());
  EXPECT_FALSE(info.hostAddressable());
  EXPECT_TRUE(info.deviceVisible());
  EXPECT_FALSE(info.openClSvm());
  EXPECT_EQ(allocator.describe(static_cast<char *>(pointer) + 256).kind,
            AllocationKind::Unknown);
  allocator.free(pointer);
  EXPECT_EQ(allocator.describe(pointer).kind, AllocationKind::Unknown);
}

TEST_F(CudaKv, deviceSnapshotIoPreservesGuardsAndPinsBacking) {
  auto allocator = std::make_shared<CudaMemAllocator>(true);
  auto operations = cuda::makeKvStorageOperations();
  auto storage = KvStorage::create(256, allocator, operations);
  auto retained = storage->memory();
  void *base = storage->data();
  ASSERT_FALSE(retained->isHostAddressable());
  ASSERT_FALSE(retained->isSVM());
  std::vector<unsigned char> input(64, 37), output(256);
  storage->write(input.data(), 64, input.size());
  storage->read(output.data(), 0, output.size());
  for (size_t i = 0; i < output.size(); ++i)
    EXPECT_EQ(output[i], i >= 64 && i < 128 ? 37 : 0) << i;
  EXPECT_THROW(storage->write(input.data(), 250, input.size()),
               std::out_of_range);
  auto &stream = cuda::StreamManager::Global();
  ASSERT_EQ(cudaMemsetAsync(base, 19, 256, stream.GetStream()), cudaSuccess);
  storage.reset();
  EXPECT_EQ(allocator->describe(base).kind, AllocationKind::CudaDevice);
  retained.reset();
  EXPECT_EQ(allocator->describe(base).kind, AllocationKind::Unknown);
}

TEST_F(CudaKv, checkedStepsWaitForProducerAndConsumer) {
  auto allocator = std::make_shared<CudaMemAllocator>(true);
  auto operations = cuda::makeKvStorageOperations();
  KvCacheManager manager;
  auto spec = geometry();
  manager.initialize(spec, allocator, operations);
  auto &stream = cuda::StreamManager::Global();
  for (size_t position = 0; position < 8; position += 2) {
    auto step = manager.beginStep(2);
    auto scope = manager.bindStep(step);
    auto key = manager.writeKey(step, 0)[0];
    auto value = manager.writeValue(step, 0)[0];
    auto access = KvStepScope::access(manager.keyMemory(0)->getAddr<void>(),
                                      manager.valueMemory(0)->getAddr<void>(),
                                      0, position, position + 2);
    ASSERT_NE(access, nullptr);
    EXPECT_THROW(key.data(), std::logic_error);
    EXPECT_THROW(value.writableData(), std::logic_error);
    access->validateWrite(key.writableAddress(), value.writableAddress(),
                          key.sizeBytes());
    ASSERT_EQ(cudaMemsetAsync(key.writableAddress(), int(position + 1),
                              key.sizeBytes(), stream.GetStream()),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(value.writableAddress(), int(position + 2),
                              value.sizeBytes(), stream.GetStream()),
              cudaSuccess);
    access->producerReady();
    auto history = manager.readKey(step, 0)[0];
    access->validateRead(history.address(),
                         manager.readValue(step, 0)[0].address(),
                         history.sizeBytes());
    std::vector<unsigned char> bytes(history.sizeBytes());
    ASSERT_EQ(cudaMemcpy(bytes.data(), history.address(), bytes.size(),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    const size_t row = spec.layers[0].rowBytes();
    for (size_t i = 0; i < bytes.size(); ++i)
      EXPECT_EQ(bytes[i], ((i / row) / 2) * 2 + 1);
    ASSERT_EQ(cudaMemsetAsync(value.writableAddress(), 51, value.sizeBytes(),
                              stream.GetStream()),
              cudaSuccess);
    manager.commit(std::move(step));
    EXPECT_EQ(manager.committedLength(), position + 2);
    EXPECT_THROW(key.writableAddress(), std::logic_error);
  }
  EXPECT_THROW(manager.beginStep(1), std::out_of_range);
  manager.reset();
  EXPECT_EQ(manager.committedLength(), 0u);
}

TEST_F(CudaKv, hostMappedStorageSupportsSequentialHandoff) {
  auto allocator = std::make_shared<CudaMemAllocator>(false);
  auto operations =
    cuda::makeKvStorageOperations(AllocationKind::CudaHostMapped);
  auto storage = KvStorage::create(256, allocator, operations);
  ASSERT_EQ(storage->allocationInfo().kind, AllocationKind::CudaHostMapped);
  EXPECT_TRUE(storage->memory()->isHostAddressable());
  EXPECT_FALSE(storage->memory()->isSVM());
  std::vector<unsigned char> pattern(256, 11), result(256);
  storage->write(pattern.data(), 0, 256);
  auto &stream = cuda::StreamManager::Global();
  ASSERT_EQ(cudaMemsetAsync(static_cast<char *>(storage->data()) + 64, 23, 64,
                            stream.GetStream()),
            cudaSuccess);
  storage->read(result.data(), 0, 256);
  for (size_t i = 0; i < result.size(); ++i)
    EXPECT_EQ(result[i], i >= 64 && i < 128 ? 23 : 11);
}

TEST_F(CudaKv, wrongProfileRejectsAllocation) {
  auto allocator = std::make_shared<CudaMemAllocator>(true);
  EXPECT_THROW(KvStorage::create(
                 256, allocator,
                 cuda::makeKvStorageOperations(AllocationKind::CudaHostMapped)),
               std::invalid_argument);
}

TEST_F(CudaKv, backingStatsCountCudaApiBytesRatherThanHostAlignmentRounding) {
  auto spec = geometry();
  spec.layers[0].num_kv_heads = 1;
  spec.layers[0].head_dim = 1;
  spec.layers[0].max_tokens = spec.layers[0].capacity = 3;
  KvCacheManager manager;
  manager.initialize(spec, std::make_shared<CudaMemAllocator>(true),
                     cuda::makeKvStorageOperations());
  EXPECT_EQ(manager.stats().backing_bytes, 12u);
}

TEST_F(CudaKv, managedKvRequiresSeparateCoherenceValidation) {
  EXPECT_THROW(cuda::makeKvStorageOperations(AllocationKind::CudaManaged),
               std::invalid_argument);
}

TEST_F(CudaKv, orderedGraphWaitsBeforePublishingDeviceWrites) {
  auto operations =
    cuda::makeKvStorageOperations(AllocationKind::CudaDevice, true);
  KvCacheManager manager;
  manager.initialize(geometry(), std::make_shared<CudaMemAllocator>(true),
                     operations);
  auto step = manager.beginStep(1);
  auto key = manager.writeKey(step, 0)[0];
  const auto *address = key.address();
  auto &stream = cuda::StreamManager::Global();
  ASSERT_TRUE(stream.beginCapture());
  auto status = cudaMemsetAsync(key.writableAddress(), 37, key.sizeBytes(),
                                stream.GetStream());
  if (status == cudaSuccess)
    manager.markReady(step, 0); // records dependency; must not wait on the host
  cudaGraph_t graph = nullptr;
  const bool ended = stream.endCapture(&graph);
  ASSERT_EQ(status, cudaSuccess);
  ASSERT_TRUE(ended);
  EXPECT_EQ(manager.committedLength(), 0u);
  cudaGraphExec_t exec = nullptr;
  ASSERT_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);
  ASSERT_EQ(cudaGraphLaunch(exec, stream.GetStream()), cudaSuccess);
  manager.commit(std::move(step)); // event completion after the launch
  EXPECT_EQ(manager.committedLength(), 1u);
  std::vector<unsigned char> result(key.sizeBytes());
  ASSERT_EQ(
    cudaMemcpy(result.data(), address, result.size(), cudaMemcpyDeviceToHost),
    cudaSuccess);
  for (auto byte : result)
    EXPECT_EQ(byte, 37);
  EXPECT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
  EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  EXPECT_THROW(
    cuda::makeKvStorageOperations(AllocationKind::CudaHostMapped, true),
    std::invalid_argument);
}

GTEST_API_ int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
