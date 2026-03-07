CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2

.PHONY: all test clean

all: test_tensor test_gpt2 gpt2

test_tensor: tests/test_tensor.cpp ./tensor.h
	$(CXX) $(CXXFLAGS) -o ./bin/test_tensor tests/test_tensor.cpp

test_gpt2: tests/test_gpt2.cpp ./gpt2.h ./tensor.h
	$(CXX) $(CXXFLAGS) -o ./bin/test_gpt2 ./tests/test_gpt2.cpp

# Main inference binary — requires real model weights at runtime (see README)
gpt2: main.cpp loader.h tokenizer.h kvcache.h gpt2.h tensor.h
	$(CXX) $(CXXFLAGS) -o ./bin/gpt2 main.cpp

test: test_tensor test_gpt2
	./bin/test_tensor
	./bin/test_gpt2

clean:
	rm -f ./bin/test_tensor ./bin/test_gpt2 ./bin/gpt2