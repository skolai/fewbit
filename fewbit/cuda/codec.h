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

void Gelu(uint32_t noelems, int32_t nobits, float const *bounds,
          float const *inputs, float *outputs, uint8_t *state);

void GeluBackward(uint32_t noelems, int32_t nobits, float const *levels,
                  uint8_t const *state, float const *outgrads, float *ingrads);

} // namespace fewbit
