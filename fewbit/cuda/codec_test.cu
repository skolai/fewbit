#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cuda.h>

#include <fewbit/cuda/codec.h>

using namespace fewbit;

int32_t constexpr nobits = 3;
int32_t constexpr length = 15;

std::array<float, 7> bounds = {
    -2.39798704e+00, -7.11248159e-01, -3.26290283e-01, -1.55338428e-04,
    +3.26182064e-01, +7.10855860e-01, +2.39811567e+00,
};

std::array<float, 8> levels = {
    -2.600090e-03, -8.883533e-02, 1.251944e-01, 3.720415e-01,
    +6.277958e-01, +8.746618e-01, 1.088807e+00, 1.002599e+00,
};

void TestCodec(void) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int32_t> dist(0, (1 << nobits) - 1);
    std::vector<int32_t> inp;
    inp.reserve(length);
    for (auto it = 0; it != length; ++it) {
        inp.push_back(dist(rng));
        std::cout << std::hex << inp.back() << ' ';
    }
    std::cout << '\n';

    int32_t length_out = static_cast<size_t>((nobits * length) / 8);
    std::vector<uint8_t> out(length_out);

    int32_t *dinp;
    cudaMalloc(&dinp, inp.size() * sizeof(int32_t));
    cudaMemcpy(dinp, inp.data(), inp.size() * sizeof(int32_t),
               cudaMemcpyKind::cudaMemcpyHostToDevice);

    uint8_t *dout;
    cudaMalloc(&dout, out.size() * sizeof(uint8_t));

    // Execute CUDA kernels.
    Deflate(nobits, dinp, dinp + inp.size(), dout);
    Inflate(nobits, dinp, dinp + inp.size(), dout);

    cudaMemcpy(inp.data(), dinp, inp.size() * sizeof(int32_t),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);

    for (auto it = 0; it != length; ++it) {
        std::cout << std::hex << inp[it] << ' ';
    }
    std::cout << '\n';
}

void TestCodecBlock(void) {
    std::vector<int32_t> index = {
        6, 5, 6, 1, 7, 0, 4, 2, 2, 3, 0, 4, 5, 5, 6, 7,
    };
    std::vector<uint8_t> state(index.size());

    int32_t *dindex;
    cudaMalloc(&dindex, index.size() * sizeof(int32_t));
    cudaMemcpy(dindex, index.data(), index.size() * sizeof(int32_t),
               cudaMemcpyKind::cudaMemcpyHostToDevice);

    uint8_t *dstate;
    cudaMalloc(&dstate, state.size() * sizeof(uint8_t));

    for (auto const idx : index) {
        std::cout << idx << ' ';
    }
    std::cout << '\n';

    DeflateBlock(nobits, length, dindex, dstate);
    InflateBlock(nobits, length, dstate, dindex);

    cudaMemcpy(index.data(), dindex, index.size() * sizeof(int32_t),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);

    for (auto const idx : index) {
        std::cout << idx << ' ';
    }
    std::cout << '\n';
}

void TestGelu(void) {
    std::array<float, 16> inp = {
        2.29811567e+00,  6.10855860e-01,  2.29811567e+00,  -8.11248159e-01,
        9.99900000e+02,  -2.49798704e+00, 2.26182064e-01,  -4.26290283e-01,
        -4.26290283e-01, -1.00155338e-01, -2.49798704e+00, 2.26182064e-01,
        6.10855860e-01,  6.10855860e-01,  2.29811567e+00,  9.99900000e+02,
    };

    float *dinp;
    cudaMalloc(&dinp, inp.size() * sizeof(float));
    cudaMemcpy(dinp, inp.data(), inp.size() * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);

    std::array<float, 16> inpgrads = {0};
    float *dinpgrads;
    cudaMalloc(&dinpgrads, inpgrads.size() * sizeof(float));

    std::array<float, 16> outgrads;
    std::fill(outgrads.begin(), outgrads.end(), 1.0);
    float *doutgrads;
    cudaMalloc(&doutgrads, outgrads.size() * sizeof(float));
    cudaMemcpy(doutgrads, outgrads.data(), outgrads.size() * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);

    float *dbounds;
    cudaMalloc(&dbounds, bounds.size() * sizeof(float));
    cudaMemcpy(dbounds, bounds.data(), bounds.size() * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);

    float *dlevels;
    cudaMalloc(&dlevels, levels.size() * sizeof(float));
    cudaMemcpy(dlevels, levels.data(), levels.size() * sizeof(float),
               cudaMemcpyKind::cudaMemcpyHostToDevice);

    float *dout;
    cudaMalloc(&dout, inp.size() * sizeof(float));

    uint8_t *dstate;
    cudaMalloc(&dstate, inp.size() * sizeof(uint8_t));

    // Execute kernels.
    Gelu(length, nobits, dbounds, dinp, dout, dstate);
    GeluBackward(length, nobits, dlevels, dstate, doutgrads, dinpgrads);

    cudaMemcpy(inpgrads.data(), dinpgrads, inpgrads.size() * sizeof(float),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);

    auto counter = 1;
    for (auto const val : inpgrads) {
        std::cout << std::scientific << val << ' ';
        if (counter != 1 && counter % 8 == 0) {
            std::cout << '\n';
        }
        ++counter;
    }
}

#define RUN_TEST(fn)                                                           \
    printf("Run test case %s\n", #fn);                                         \
    fn();                                                                      \
    printf("\n")

int main() {
    RUN_TEST(TestCodec);
    RUN_TEST(TestCodecBlock);
    RUN_TEST(TestGelu);
    return 0;
}
