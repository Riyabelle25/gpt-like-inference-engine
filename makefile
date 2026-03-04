# Makefile for GPT-2 Inference Engine — Milestone 1
# Usage:
#   make          -> build the test binary
#   make test     -> build and run tests
#   make clean    -> remove build artefacts

CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2

.PHONY: all test clean

all: test_tensor

test_tensor: test_tensor.cpp tensor.h
	$(CXX) $(CXXFLAGS) -o test_tensor test_tensor.cpp

test: test_tensor
	./test_tensor

clean:
	rm -f test_tensor