#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace fewbit {

/**
 * Struct CodecTrait contains static definitions for instantiation of generic
 * compression/decompression routines for a specific element type.
 */
template <typename T> struct CodecTrait;

template <> struct CodecTrait<uint8_t> {
    static constexpr int32_t kMaxBitWidth = 8;

    static constexpr std::array<uint8_t, kMaxBitWidth> kMask = {
        0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0x7f, 0xff,
    };
};

template <> struct CodecTrait<uint16_t> {
    static constexpr int32_t kMaxBitWidth = 16;

    static constexpr std::array<uint16_t, kMaxBitWidth> kMask = {
        0x0001, 0x0003, 0x0007, 0x000f, 0x001f, 0x003f, 0x007f, 0x00ff,
        0x01ff, 0x03ff, 0x07ff, 0x0fff, 0x1fff, 0x3fff, 0x7fff, 0xffff,
    };
};

template <typename T>
void Deflate(int32_t const *begin, int32_t const *end, T *out,
             int32_t bitwidth) {
    auto constexpr maxwidth = CodecTrait<T>::kMaxBitWidth;

    int32_t shift = 0, threshold = maxwidth - bitwidth;
    int32_t bitlength = bitwidth * (end - begin);
    int32_t offset = static_cast<int32_t>(bitlength % maxwidth == 0);

    *out = 0;
    for (auto it = begin; it != end - offset; ++it) {
        if (shift < threshold) /* [[likely]] */ {
            *out |= static_cast<T>(*it) << shift; // full
            shift += bitwidth;
        } else /* [[unlikely]] */ {
            *out |= static_cast<T>(*it) << shift; // lo
            shift = shift + bitwidth - maxwidth;
            *(++out) = static_cast<T>(*it) >> (bitwidth - shift); // hi
        }
    }

    if (offset) {
        *out |= static_cast<T>(*(end - offset)) << shift;
    }
}

template <typename T>
void Inflate(int32_t *begin, int32_t *end, T const *in, int32_t bitwidth) {
    auto constexpr maxwidth = CodecTrait<T>::kMaxBitWidth;
    auto const mask = CodecTrait<T>::kMask[bitwidth - 1];

    int32_t shift = 0, threshold = maxwidth - bitwidth;
    int32_t bitlength = bitwidth * (end - begin);
    int32_t offset = static_cast<int32_t>(bitlength % maxwidth == 0);

    for (auto it = begin; it != end - offset; ++it) {
        if (shift < threshold) {
            *it = static_cast<int32_t>((*in >> shift) & mask); // full
            shift += bitwidth;
        } else {
            *it = static_cast<int32_t>(*in >> shift); // lo
            shift = shift + bitwidth - maxwidth;
            *it |= static_cast<int32_t>(*(++in) << (bitwidth - shift)); // hi
            *it &= mask;
        }
    }

    if (offset) {
        *(end - offset) = static_cast<int32_t>((*in >> shift) & mask);
    }
}

/**
 * Class Codec provides plain state and interface for simple compression codec.
 */
template <typename T = uint8_t> class Codec {
public:
    Codec(int32_t bitwidth = 3) : bitwidth_{bitwidth_} {
        assert(("Only positive bit width value is allowed.", bitwidth_ > 0));
        assert(("Compression word does not fit element type.",
                bitwidth_ <= CodecTrait<T>::kMaxBitWidth));
    }

    void Deflate(int32_t const *begin, int32_t const *end, T *out) const {
        Deflate(begin, end, out, bitwidth_);
    }

    void Inflate(int32_t *begin, int32_t *end, T const *in) const {
        Inflate(begin, end, in, bitwidth_);
    }

private:
    int32_t bitwidth_;
};

} // namespace fewbit
