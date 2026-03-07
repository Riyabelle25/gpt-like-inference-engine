CXX      := g++
# -mavx2 -mfma: enable AVX2 and FMA instruction sets (Milestone 4)
# -pthread:     enable std::thread support (Milestone 5)
# -O2:          standard optimisation — also enables auto-vectorisation
CXXFLAGS := -std=c++17 -Wall -Wextra -O2 -mavx2 -mfma -pthread

.PHONY: all test clean

all: test_tensor test_gpt2 gpt2 bench

test_tensor: tests/test_tensor.cpp ./tensor.h
	$(CXX) $(CXXFLAGS) -o ./bin/test_tensor tests/test_tensor.cpp

test_gpt2: tests/test_gpt2.cpp ./gpt2.h ./tensor.h
	$(CXX) $(CXXFLAGS) -o ./bin/test_gpt2 ./tests/test_gpt2.cpp

# Main inference binary — requires real model weights at runtime (see README)
gpt2: main.cpp loader.h tokenizer.h kvcache.h gpt2.h tensor.h
	$(CXX) $(CXXFLAGS) -o ./bin/gpt2 main.cpp

# Benchmark binary - micro-benchmarks matvec, matmul, attention_cached
bench: bench.cpp tensor.h gpt2.h kvcache.h
	$(CXX) $(CXXFLAGS) -o ./bin/bench bench.cpp

test: test_tensor test_gpt2
	./bin/test_tensor
	./bin/test_gpt2

clean:
	rm -f ./bin/test_tensor ./bin/test_gpt2 ./bin/gpt2 ./bin/bench