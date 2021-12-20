/**
 * \file codec.cu
 */

#include "codec.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace fewbit {

// Use maximum dimensionality of a thread block for compute capability from 3.5
// to 8.6 (and maybe futher).
constexpr auto kThreadsPerBlock = 1024u;

constexpr auto kWarpSize = 8u; // Reduce warp size.

static_assert(kThreadsPerBlock % kWarpSize == 0,
              "Number of threads per block should be multiple of kWarpSize.");

template <typename T> struct CodecTrait;

template <> struct CodecTrait<uint8_t> {
    static constexpr int32_t kMaxBitWidth = 8;

    __device__ static uint8_t GetMask(int32_t nobits) {
        static constexpr std::array<uint8_t, kMaxBitWidth> kMask = {
            0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0x7f, 0xff,
        };
        return kMask[nobits - 1];
    }
};

template <> struct CodecTrait<uint16_t> {
    static constexpr int32_t kMaxBitWidth = 16;

    __device__ static uint16_t GetMask(int32_t nobits) {
        static constexpr std::array<uint16_t, kMaxBitWidth> kMask = {
            0x0001, 0x0003, 0x0007, 0x000f, 0x001f, 0x003f, 0x007f, 0x00ff,
            0x01ff, 0x03ff, 0x07ff, 0x0fff, 0x1fff, 0x3fff, 0x7fff, 0xffff,
        };
        return kMask[nobits - 1];
    }
};

template <typename T>
__global__ void DeflateKernel(uint32_t size, int32_t nobits, int32_t const *inp,
                              T *out) {
    auto constexpr maxwidth = CodecTrait<T>::kMaxBitWidth;
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) {
        return;
    }

    // Calculate offset in input sequences.
    auto bitlength = maxwidth * tid;
    auto remainder = bitlength % nobits;
    inp += bitlength / nobits;

    T result;
    uint32_t shift;
    if (remainder > 0) {
        result = static_cast<T>(*inp) >> remainder;
        shift = nobits - remainder;
    } else {
        result = static_cast<T>(*inp) >> remainder;
        shift = nobits;
    }
    while (shift <= maxwidth) {
        result |= static_cast<T>(*(++inp)) << shift;
        shift += nobits;
    }

    out[tid] = result;
}

void Deflate(int32_t nobits, int32_t const *begin, int32_t const *end,
             uint8_t *inflated) {
    auto size_inp = end - begin;
    auto size_out = static_cast<uint32_t>((nobits * size_inp) / 8);
    dim3 noblocks((size_out - 1) / kThreadsPerBlock);
    dim3 nothreads(std::min(size_out, kThreadsPerBlock));
    DeflateKernel<uint8_t>
        <<<noblocks, nothreads>>>(size_out, nobits, begin, inflated);
}

template <typename T>
__global__ void InflateKernel(uint32_t size, int32_t nobits, T const *deflated,
                              int32_t *inflated) {
    auto constexpr maxwidth = CodecTrait<T>::kMaxBitWidth;
    auto const mask = CodecTrait<T>::GetMask(nobits);
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > size) {
        return;
    }

    auto length = nobits * tid;
    auto shift = length % maxwidth;
    deflated += length / maxwidth;
    inflated[tid] = (*deflated >> shift) & mask;
    if (shift + nobits > maxwidth) {
        ++deflated;
        shift = maxwidth - shift;
        inflated[tid] |= (*deflated << shift) & mask;
    }
}

void Inflate(int32_t nobits, int32_t *begin, int32_t *end,
             uint8_t const *deflated) {
    auto deflate_size = static_cast<uint32_t>(end - begin); // unpacked
    dim3 noblocks((deflate_size - 1) / kThreadsPerBlock);
    dim3 nothreads(std::min(deflate_size, kThreadsPerBlock));
    InflateKernel<uint8_t>
        <<<noblocks, nothreads>>>(deflate_size, nobits, deflated, begin);
}

__device__ int32_t BinarySearch(uint32_t size, float val, float const *elems) {
    float const *begin = elems;
    ptrdiff_t count = size;
    while (count > 0) {
        ptrdiff_t step = count / 2;
        float const *it = elems + step;
        if (*it < val) {
            elems = ++it;
            count -= step + 1;
        } else
            count = step;
    }
    return elems - begin;
}

__device__ int32_t LinearSearch(uint32_t size, float val, float const *elems) {
    for (auto it = 0; it != size; ++it) {
        if (val <= elems[it]) {
            return it;
        }
    }
    return size;
}

__device__ void DeflateWarpKernel(int32_t nobits, int32_t index,
                                  uint8_t *compressed) {
    auto const kNoWarps = warpSize / kWarpSize; // Expect 4 warps.
    auto warp = (threadIdx.x / kWarpSize) % kNoWarps;
    auto lane = threadIdx.x % kWarpSize;

    uint64_t value = static_cast<uint64_t>(index) << nobits * lane;
    value |= __shfl_xor_sync(0xff, value, 0b0001, kWarpSize);
    value |= __shfl_xor_sync(0xff, value, 0b0010, kWarpSize);
    value |= __shfl_xor_sync(0xff, value, 0b0100, kWarpSize);

    if (lane == 0) {
        // Every 8 input bytes are compressed to `nobits` output bytes. We use
        // byte sequence to encode compressed data, but it is also possible to
        // use word sequence instead. However, in that case, we should increase
        // number of synchronous communication within warp by one.
        auto tid = blockDim.x * blockIdx.x + threadIdx.x;
        auto bid = tid / kWarpSize; // Global index of compressed block.
        compressed += nobits * bid;
        for (auto it = 0, shift = 0; it != nobits; ++it, shift += 8) {
            compressed[it] = static_cast<uint8_t>(value >> shift);
        }
    }
}
__global__ void DeflateBlockKernel(int32_t nobits, uint32_t noelems,
                                   int32_t const *input, uint8_t *data) {
    // We should call per warp deflate kernel in order to properly communicate
    // among threads in the same warp.
    auto tid = blockIdx.x * blockDim.x + threadIdx.x; // Thread ID.
    auto val = tid < noelems ? input[tid] : 0;
    DeflateWarpKernel(nobits, val, data);
}

void DeflateBlock(int32_t nobits, uint32_t noelems, int32_t const *input,
                  uint8_t *data) {
    assert(noelems > 0); // Assert only non-empty sequences.
    auto nogroups = (noelems - 1) / kWarpSize + 1;
    dim3 noblocks((noelems - 1) / kThreadsPerBlock + 1);
    dim3 nothreads(std::min(kThreadsPerBlock, kWarpSize * nogroups));
    DeflateBlockKernel<<<noblocks, nothreads>>>(nobits, noelems, input, data);
}

__device__ inline float CalcGelu(float val) {
    return val * normcdf(val);
}

__global__ void GeluKernel(int32_t nobits, uint32_t noelems,
                           float const *bounds, float const *inputs,
                           float *outputs, uint8_t *state) {
    auto len = (1 << nobits) - 1;
    auto idx = 0;
    if (auto tid = blockIdx.x * blockDim.x + threadIdx.x; tid < noelems) {
        outputs[tid] = CalcGelu(inputs[tid]);
        idx = BinarySearch(len, inputs[tid], bounds); // Or LinearSearch.
    }

    // We should call per warp deflate kernel in order to properly communicate
    // among threads in the same warp.
    DeflateWarpKernel(nobits, idx, state);
}

void Gelu(uint32_t noelems, int32_t nobits, float const *bounds,
          float const *inputs, float *outputs, uint8_t *state) {
    assert(noelems > 0); // Assert only non-empty sequences.
    auto nogroups = (noelems - 1) / kWarpSize + 1;
    dim3 noblocks((noelems - 1) / kThreadsPerBlock + 1);
    dim3 nothreads(std::min(kThreadsPerBlock, kWarpSize * nogroups));
    GeluKernel<<<noblocks, nothreads>>>(nobits, noelems, bounds, inputs,
                                        outputs, state);
}

__device__ int32_t InflateWarpKernel(int32_t nobits, uint8_t const *data) {
    auto constexpr maxwidth = CodecTrait<uint8_t>::kMaxBitWidth;
    auto const mask = CodecTrait<uint8_t>::GetMask(nobits);
    auto tid = blockDim.x * blockIdx.x + threadIdx.x; // Thread ID.
    auto gid = tid / kWarpSize;                       // Group ID.
    auto lid = tid % kWarpSize;                       // Lane (local) ID.
    auto idx = (nobits * lid) / kWarpSize;            // Byte index in block.

    data += nobits * gid + idx;

    // Decode specific element from compressed block.
    int32_t shift = (nobits * lid) % kWarpSize;
    int32_t value = (data[0] >> shift) & mask;
    if (shift + nobits > maxwidth) {
        shift = maxwidth - shift;
        value |= (data[1] << shift) & mask;
    }

    return value;
}

__global__ void InflateBlockKernel(int32_t nobits, uint32_t noelems,
                                   uint8_t const *data, int32_t *output) {
    if (auto tid = blockIdx.x * blockDim.x + threadIdx.x; tid < noelems) {
        output[tid] = InflateWarpKernel(nobits, data);
    }
}

void InflateBlock(int32_t nobits, uint32_t noelems, uint8_t const *data,
                  int32_t *output) {
    assert(noelems > 0); // Assert only non-empty sequences.
    auto nogroups = (noelems - 1) / kWarpSize + 1;
    dim3 noblocks((noelems - 1) / kThreadsPerBlock + 1);
    dim3 nothreads(std::min(kThreadsPerBlock, kWarpSize * nogroups));
    InflateBlockKernel<<<noblocks, nothreads>>>(nobits, noelems, data, output);
}

__global__ void GeluBackwardKernel(int32_t nobits, uint32_t noelems,
                                   float const *levels, uint8_t const *state,
                                   float const *outgrads, float *ingrads) {
    if (auto tid = blockIdx.x * blockDim.x + threadIdx.x; tid < noelems) {
        auto idx = InflateWarpKernel(nobits, state);
        ingrads[tid] = levels[idx] * outgrads[tid];
    }
}

void GeluBackward(uint32_t noelems, int32_t nobits, float const *levels,
                  uint8_t const *state, float const *outgrads, float *ingrads) {
    assert(noelems > 0); // Assert only non-empty sequences.
    auto nogroups = (noelems - 1) / kWarpSize + 1;
    dim3 noblocks((noelems - 1) / kThreadsPerBlock + 1);
    dim3 nothreads(std::min(kThreadsPerBlock, kWarpSize * nogroups));
    GeluBackwardKernel<<<noblocks, nothreads>>>(nobits, noelems, levels, state,
                                                outgrads, ingrads);
}

/**
 * First of all, we want to introduce some macro definitions to avoid redundant
 * repeating in CUDA kernel declarations for stepwise activation functions.
 * Since all kernels below are element-wiseand share the same lattice topology
 * on device, we define routines for kernel invocation.
 */

#define DEFINE_KERNEL_TOPOLOGY(var)                                            \
    assert(var > 0);                                                           \
    auto nogroups = (var - 1) / kWarpSize + 1;                                 \
    dim3 noblocks((var - 1) / kThreadsPerBlock + 1);                           \
    dim3 nothreads(std::min(kThreadsPerBlock, kWarpSize *nogroups))

#define DEFINE_KERNEL_INVOCATION(kernel, numel, ...)                           \
    {                                                                          \
        DEFINE_KERNEL_TOPOLOGY(numel);                                         \
        kernel<<<noblocks, nothreads>>>(__VA_ARGS__);                          \
    }

/**
 * The only difference between forward pass kernels is a number of auxiliary
 * parameters. In case of 1-bit stepwise activation function the number of such
 * parameters does not exceed two. So, we can avoid this regularity with
 * a bunch of macroses.
 */

#define DEFINE_STEPWISE_FUNC_FORWARD(name)                                     \
    void name(uint32_t noelems, float const *inputs, float *outputs,           \
              uint8_t *state) {                                                \
        DEFINE_KERNEL_INVOCATION(name##Kernel, noelems, noelems, inputs,       \
                                 outputs, state);                              \
    }

#define DEFINE_STEPWISE_FUNC_FORWARD0(name) DEFINE_STEPWISE_FUNC_FORWARD(name)

#define DEFINE_STEPWISE_FUNC_FORWARD1(name, type)                              \
    void name(uint32_t noelems, float const *inputs, float *outputs,           \
              uint8_t *state, type arg) {                                      \
        DEFINE_KERNEL_INVOCATION(name##Kernel, noelems, noelems, inputs,       \
                                 outputs, state, arg);                         \
    }

#define DEFINE_STEPWISE_FUNC_FORWARD2(name, type1, type2)                      \
    void name(uint32_t noelems, float const *inputs, float *outputs,           \
              uint8_t *state, type1 arg1, type2 arg2) {                        \
        DEFINE_KERNEL_INVOCATION(name##Kernel, noelems, noelems, inputs,       \
                                 outputs, state, arg1, arg2);                  \
    }

#define DEFINE_STEPWISE_FUNC_BACKWARD_KERNEL(name)                             \
    __global__ void name##BackwardKernel(                                      \
        uint32_t noelems, uint8_t const *state,                                \
        float const *__restrict__ outgrads, float *__restrict__ ingrads) {     \
        if (auto tid = blockIdx.x * blockDim.x + threadIdx.x; tid < noelems) { \
            auto idx = InflateWarpKernel(1, state);                            \
            ingrads[tid] = idx * outgrads[tid];                                \
        }                                                                      \
    }

/**
 * Almost all kernels for backward pass are the same. We either let gradients
 * to pass or not. The only exception is leaky ReLU with custom negative slope.
 */

#define DEFINE_STEPWISE_FUNC_BACKWARD_FUNC(name)                               \
    void name##Backward(uint32_t noelems, uint8_t const *state,                \
                        float const *outgrads, float *ingrads) {               \
        DEFINE_KERNEL_INVOCATION(name##BackwardKernel, noelems, noelems,       \
                                 state, outgrads, ingrads);                    \
    }

#define DEFINE_STEPWISE_FUNC_BACKWARD(name)                                    \
    DEFINE_STEPWISE_FUNC_BACKWARD_KERNEL(name)                                 \
    DEFINE_STEPWISE_FUNC_BACKWARD_FUNC(name)

__global__ void HardshrinkKernel(uint32_t noelems, float const *inputs,
                                 float *outputs, uint8_t *state, float lambda) {
    auto index = 1;
    if (auto tid = blockIdx.x * blockDim.x + threadIdx.x; tid < noelems) {
        if (inputs[tid] < -lambda || inputs[tid] > lambda) {
            outputs[tid] = inputs[tid];
        } else {
            outputs[tid] = 0.0;
            index = 0;
        }
    }
    DeflateWarpKernel(1, index, state);
}

DEFINE_STEPWISE_FUNC_FORWARD1(Hardshrink, float);
DEFINE_STEPWISE_FUNC_BACKWARD(Hardshrink);

__global__ void HardsigmoidKernel(uint32_t noelems, float const *inputs,
                                  float *outputs, uint8_t *state) {
    auto index = 0;
    if (auto tid = blockIdx.x * blockDim.x + threadIdx.x; tid < noelems) {
        if (inputs[tid] <= -3.0) {
            outputs[tid] = 0.0;
        } else if (inputs[tid] >= 3.0) {
            outputs[tid] = 1.0;
        } else {
            outputs[tid] = inputs[tid] / 6 + 0.5;
            index = 1;
        }
    }
    DeflateWarpKernel(1, index, state);
}

DEFINE_STEPWISE_FUNC_FORWARD0(Hardsigmoid);

__global__ void HardsigmoidBackwardKernel(uint32_t noelems,
                                          uint8_t const *state,
                                          float const *outgrads,
                                          float *ingrads) {
    if (auto tid = blockIdx.x * blockDim.x + threadIdx.x; tid < noelems) {
        auto indx = InflateWarpKernel(1, state);
        auto mult = indx ? 1.0f / 6.0f : 0.0f;
        ingrads[tid] = mult * outgrads[tid];
    }
}

void HardsigmoidBackward(uint32_t noelems, uint8_t const *state,
                         float const *outgrads, float *ingrads) {
    DEFINE_KERNEL_TOPOLOGY(noelems);
    HardsigmoidBackwardKernel<<<noblocks, nothreads>>>(noelems, state, outgrads,
                                                       ingrads);
}

__global__ void HardtanhKernel(uint32_t noelems, float const *inputs,
                               float *outputs, uint8_t *state, float min_val,
                               float max_val) {
    auto index = 0;
    if (auto tid = blockIdx.x * blockDim.x + threadIdx.x; tid < noelems) {
        if (inputs[tid] <= min_val) {
            outputs[tid] = min_val;
        } else if (inputs[tid] >= max_val) {
            outputs[tid] = max_val;
        } else {
            outputs[tid] = inputs[tid];
            index = 1;
        }
    }
    DeflateWarpKernel(1, index, state);
}

DEFINE_STEPWISE_FUNC_FORWARD2(Hardtanh, float, float);
DEFINE_STEPWISE_FUNC_BACKWARD(Hardtanh);

__global__ void LeakyReluKernel(uint32_t noelems, float const *inputs,
                                float *outputs, uint8_t *state,
                                float negative_slope) {
    auto index = 0;
    if (auto tid = blockIdx.x * blockDim.x + threadIdx.x; tid < noelems) {
        if (inputs[tid] >= 0) {
            outputs[tid] = inputs[tid];
        } else {
            outputs[tid] = negative_slope * inputs[tid];
            index = 1;
        }
    }
    DeflateWarpKernel(1, index, state);
}

DEFINE_STEPWISE_FUNC_FORWARD1(LeakyRelu, float);

__global__ void LeakyReluBackwardKernel(uint32_t noelems, uint8_t const *state,
                                        float const *outgrads, float *ingrads,
                                        float negative_slope) {
    if (auto tid = blockIdx.x * blockDim.x + threadIdx.x; tid < noelems) {
        auto indx = InflateWarpKernel(1, state);
        auto mult = indx ? negative_slope : 1.0;
        ingrads[tid] = mult * outgrads[tid];
    }
}

void LeakyReluBackward(uint32_t noelems, uint8_t const *state,
                       float const *outgrads, float *ingrads,
                       float negative_slope) {
    DEFINE_KERNEL_TOPOLOGY(noelems);
    LeakyReluBackwardKernel<<<noblocks, nothreads>>>(noelems, state, outgrads,
                                                     ingrads, negative_slope);
}

__global__ void ReluKernel(uint32_t noelems, float const *inputs,
                           float *outputs, uint8_t *state) {
    auto index = 0;
    if (auto tid = blockIdx.x * blockDim.x + threadIdx.x; tid < noelems) {
        if (inputs[tid] <= 0) {
            outputs[tid] = 0.0;
        } else {
            outputs[tid] = inputs[tid];
            index = 1;
        }
    }
    DeflateWarpKernel(1, index, state);
}

DEFINE_STEPWISE_FUNC_FORWARD0(Relu);
DEFINE_STEPWISE_FUNC_BACKWARD(Relu);

__global__ void Relu6Kernel(uint32_t noelems, float const *inputs,
                            float *outputs, uint8_t *state) {
    auto index = 0;
    if (auto tid = blockIdx.x * blockDim.x + threadIdx.x; tid < noelems) {
        if (inputs[tid] <= 0.0) {
            outputs[tid] = 0.0;
        } else if (inputs[tid] >= 6.0) {
            outputs[tid] = 1.0;
        } else {
            outputs[tid] = inputs[tid];
            index = 1;
        }
    }
    DeflateWarpKernel(1, index, state);
}

DEFINE_STEPWISE_FUNC_FORWARD0(Relu6);
DEFINE_STEPWISE_FUNC_BACKWARD(Relu6);

__global__ void SoftshrinkKernel(uint32_t noelems, float const *inputs,
                                 float *outputs, uint8_t *state, float lambda) {
    auto index = 1;
    if (auto tid = blockIdx.x * blockDim.x + threadIdx.x; tid < noelems) {
        if (inputs[tid] < -lambda) {
            outputs[tid] = inputs[tid] + lambda;
        } else if (inputs[tid] > lambda) {
            outputs[tid] = inputs[tid] - lambda;
        } else {
            outputs[tid] = 0.0;
            index = 0;
        }
    }
    DeflateWarpKernel(1, index, state);
}

DEFINE_STEPWISE_FUNC_FORWARD1(Softshrink, float);
DEFINE_STEPWISE_FUNC_BACKWARD(Softshrink);

__global__ void ThresholdKernel(uint32_t noelems, float const *inputs,
                                float *outputs, uint8_t *state, float threshold,
                                float value) {
    auto index = 0;
    if (auto tid = blockIdx.x * blockDim.x + threadIdx.x; tid < noelems) {
        if (inputs[tid] <= threshold) {
            outputs[tid] = value;
        } else {
            outputs[tid] = inputs[tid];
            index = 1;
        }
    }
    DeflateWarpKernel(1, index, state);
}

DEFINE_STEPWISE_FUNC_FORWARD2(Threshold, float, float);
DEFINE_STEPWISE_FUNC_BACKWARD(Threshold);

} // namespace fewbit
