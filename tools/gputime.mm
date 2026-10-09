// GPU execution time on PyTorch's MPS stream, one command buffer at a time.
//
// PyTorch's own profiler records no GPU time on MPS, and its MPS logging
// profiler (PYTORCH_MPS_LOG_PROFILE_INFO) crashes on a large full reduction
// (torch 2.14). This reads the clock Metal itself keeps: every command
// buffer reports when the GPU started and finished it.
//
// An operation is timed between begin(tag) and end(tag). begin attaches a
// completion handler to the command buffer the operation is about to encode
// into; end attaches another to the buffer it finished in, and commits that
// one without waiting, so the CPU keeps queuing work. Two handlers because
// some operations commit for themselves - MPSGraph ones, a matmul, commit
// adaptively - and then the work is split over two buffers, or lands wholly
// in the first. Where both handlers sit on one buffer it is reported twice,
// with identical times; the caller drops the duplicate.
//
// collect() waits for the GPU and returns every record since the last
// collect. Times are host seconds, the clock now() reads, so GPU timestamps
// and CPU timestamps line up.
//
// Built on demand by tools/gpuprof.py (torch.utils.cpp_extension.load).

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

namespace {

struct Record {
  int64_t tag;
  double cpu;          // when mark() was called
  double gpu_start;    // when the GPU began the command buffer
  double gpu_end;      // and when it finished it
};

std::mutex mu;
std::vector<Record> records;
std::atomic<int64_t> in_flight{0};

double host_seconds() {
  static const mach_timebase_info_data_t tb = [] {
    mach_timebase_info_data_t t;
    mach_timebase_info(&t);
    return t;
  }();
  return double(mach_absolute_time()) * tb.numer / tb.denom * 1e-9;
}

}  // namespace

static void attach(at::mps::MPSStream* stream, int64_t tag) {
  const double cpu = host_seconds();
  in_flight++;
  stream->addCompletedHandler(^(id<MTLCommandBuffer> cb) {
    std::lock_guard<std::mutex> g(mu);
    records.push_back({tag, cpu, cb.GPUStartTime, cb.GPUEndTime});
    in_flight--;
  });
}

void begin(int64_t tag) {
  attach(at::mps::getCurrentMPSStream(), tag);
}

void end(int64_t tag) {
  auto* stream = at::mps::getCurrentMPSStream();
  attach(stream, tag);
  stream->synchronize(at::mps::SyncType::COMMIT);
}

std::vector<std::tuple<int64_t, double, double, double>> collect() {
  at::mps::getCurrentMPSStream()->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
  // completed handlers run just after completion, on Metal's own thread
  for (int i = 0; in_flight.load() > 0 && i < 20000; ++i)
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  std::lock_guard<std::mutex> g(mu);
  std::vector<std::tuple<int64_t, double, double, double>> out;
  out.reserve(records.size());
  for (const auto& r : records) out.emplace_back(r.tag, r.cpu, r.gpu_start, r.gpu_end);
  records.clear();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("begin", &begin, "an operation tagged `tag` is about to encode");
  m.def("end", &end, "it has encoded: commit, and time what it used under `tag`");
  m.def("collect", &collect, "wait for the GPU; every (tag, cpu, gpu_start, gpu_end) since the last collect");
  m.def("now", &host_seconds, "host seconds, the clock the GPU timestamps use");
}
