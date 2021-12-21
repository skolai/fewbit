/**
 * \file codec.h
 *
 * \note We always assume that we have enough memory to store or to access
 * entire block i.e. all `state` buffers are alligned according to `8 * nobits *
 * sizeof(T)` bytes border (either 8 bytes or 16 bytes).
 */

#pragma once

namespace fewbit {

void Deflate(int32_t nobits, int32_t const *begin, int32_t const *end,
             uint8_t *out);

void DeflateBlock(int32_t nobits, uint32_t noelems, int32_t const *input,
                  uint8_t *data);

void Inflate(int32_t nobits, int32_t *begin, int32_t *end, uint8_t const *inp);

void InflateBlock(int32_t nobits, uint32_t noelems, uint8_t const *data,
                  int32_t *output);

/**
 * Almost all stepwise activation functions (except leaky ReLU) has the same
 * signature and very similar implementations. So, we can use unified machinery
 * for declaration here.
 */

#define DECLARE_STEPWISE_FUNC_FORWARD(name, ...)                               \
    void name(uint32_t noelems, float const *inputs, float *outputs,           \
              uint8_t *state __VA_OPT__(, __VA_ARGS__))

#define DECLARE_STEPWISE_FUNC_BACKWARD(name, ...)                              \
    void name##Backward(uint32_t noelems, uint8_t const *state,                \
                        float const *outgrads,                                 \
                        float *ingrads __VA_OPT__(, __VA_ARGS__))

#define DECLARE_STEPWISE_FUNC(name, ...)                                       \
    DECLARE_STEPWISE_FUNC_FORWARD(name, __VA_ARGS__);                          \
    DECLARE_STEPWISE_FUNC_BACKWARD(name);

/**
 * The example of macro usage for hardshrink activation function with single
 * parameters.
 *
 *  DECLARE_STEPWISE_FUNC(Hardshrink, float lambda);
 *
 * This line expands to declarations for forward pass and backward pass function
 * for the activation function.
 *
 *  void Hardshrink(uint32_t noelems, float const *inputs, float *outputs,
 *                  uint8_t *state, float lambda);
 *
 *  void HardshrinkBackward(uint32_t noelems, float const *inputs,
 *                          float *outputs, uint8_t *state);
 */

DECLARE_STEPWISE_FUNC(Hardshrink, float lambda);
DECLARE_STEPWISE_FUNC(Hardsigmoid);
DECLARE_STEPWISE_FUNC(Hardtanh, float min_val, float max_val);
DECLARE_STEPWISE_FUNC(Relu);
DECLARE_STEPWISE_FUNC(Relu6);
DECLARE_STEPWISE_FUNC(Softshrink, float lambda);
DECLARE_STEPWISE_FUNC(Threshold, float threshold, float value);

DECLARE_STEPWISE_FUNC_BACKWARD(LeakyRelu, float negative_slope);
DECLARE_STEPWISE_FUNC_FORWARD(LeakyRelu, float negative_slope);

/**
 * In case of continous functions we have general kernel for backward pass and
 * should specialize only forward pass.
 */

#define DECLARE_CONTINOUS_FUNC(name, ...)                                      \
    void name(uint32_t noelems, float const *inputs, float *outputs,           \
              uint8_t *state, uint32_t nobits,                                 \
              float const *bounds __VA_OPT__(, __VA_ARGS__))

DECLARE_CONTINOUS_FUNC(Celu, double alpha);
DECLARE_CONTINOUS_FUNC(Elu, double alpha);
DECLARE_CONTINOUS_FUNC(Gelu);
DECLARE_CONTINOUS_FUNC(Hardswish);
DECLARE_CONTINOUS_FUNC(LogSigmoid);
DECLARE_CONTINOUS_FUNC(Mish);
DECLARE_CONTINOUS_FUNC(Selu);
DECLARE_CONTINOUS_FUNC(Sigmoid);
DECLARE_CONTINOUS_FUNC(Silu);
DECLARE_CONTINOUS_FUNC(Softplus, double beta, double threshold);
DECLARE_CONTINOUS_FUNC(Softsign);
DECLARE_CONTINOUS_FUNC(Tanh);
DECLARE_CONTINOUS_FUNC(Tanhshrink);

void StepwiseBackward(uint32_t noelems, uint8_t const *state,
                      float const *outgrads, float *ingrads, int32_t nobits,
                      float const *levels);

} // namespace fewbit
