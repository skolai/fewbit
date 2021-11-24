#include <array>
#include <cassert>
#include <memory>
#include <random>
#include <ranges>

#include <fewbit/codec.h>

void TestExample(void) {
    auto constexpr bitwidth = 3;

    std::array<int32_t, 4> inp = {0, 1, 4, 7};
    std::array<int32_t, 4> out = {0};
    std::array<uint8_t, 2> buffer = {0};

    fewbit::Deflate(inp.begin(), inp.end(), buffer.begin(), bitwidth);
    fewbit::Inflate(out.begin(), out.end(), buffer.begin(), bitwidth);

    for (size_t it = 0; it != inp.size(); ++it) {
        assert(inp[it] == out[it]);
    }
}

void TestRandomized(int32_t bitwidth, size_t length, size_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int32_t> dist(0, (1 << bitwidth) - 1);
    std::vector<int32_t> src;
    src.reserve(length);
    for (size_t it = 0; it != length; ++it) {
        src.push_back(dist(rng));
    }

    auto buffer_size = static_cast<size_t>(std::ceil(bitwidth / 8. * length));
    auto buffer = std::make_unique<uint8_t[]>(buffer_size);

    fewbit::Deflate(src.data(), src.data() + length, buffer.get(), bitwidth);

    auto dst = std::make_unique<int32_t[]>(length);

    fewbit::Inflate(dst.get(), dst.get() + length, buffer.get(), bitwidth);

    for (size_t it = 0; it != length; ++it) {
        assert(src[it] == dst[it]);
    }
}

void TestRandomized(void) {
    for (auto const bitwidth : std::ranges::iota_view{1, 9}) {
        TestRandomized(bitwidth, 256);
    }
}

int main() {
    TestExample();
    TestRandomized();
    return 0;
}
