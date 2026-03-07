// main.cpp — GPT-2 Inference Engine: end-to-end text generation.
//
// Usage:
//   ./gpt2 <model_dir> "<prompt>" [max_tokens] [mode] [top_k/p] [temperature]
//
//   model_dir   : directory containing model.safetensors, vocab.json, merges.txt
//   prompt      : text to continue
//   max_tokens  : number of tokens to generate (default: 40)
//   mode        : greedy | topk | topp  (default: greedy)
//   top_k/p     : k for topk (default: 40), p for topp (default: 0.9)
//   temperature : float > 0 (default: 1.0)
//
// Example:
//   ./gpt2 ./gpt2-weights "The history of artificial intelligence" 60 topk 40 0.8
//
// Files required in model_dir:
//   model.safetensors   — from: huggingface.co/openai-community/gpt2
//   vocab.json        — same source
//   merges.txt           — same source
//
// Download with:
//   pip install huggingface_hub
//   python -c "from huggingface_hub import snapshot_download;
//              snapshot_download('openai-community/gpt2', local_dir='./gpt2-weights')"

#include "loader.h"
#include "tokenizer.h"
#include "kvcache.h"
#include <cstdio>
#include <string>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <model_dir> \"<prompt>\" [max_tokens] [greedy|topk|topp]"
            " [k_or_p] [temperature]\n", argv[0]);
        return 1;
    }

    std::string model_dir  = argv[1];
    std::string prompt     = argv[2];
    int   max_new_tokens   = (argc >= 4) ? std::atoi(argv[3]) : 40;
    std::string mode       = (argc >= 5) ? argv[4] : "greedy";
    float k_or_p           = (argc >= 6) ? std::atof(argv[5]) : 40.0f;
    float temperature      = (argc >= 7) ? std::atof(argv[6]) : 1.0f;

    // -----------------------------------------------------------------------
    // 1. Load tokenizer
    // -----------------------------------------------------------------------
    std::string encoder_path = model_dir + "/vocab.json";
    std::string vocab_path   = model_dir + "/merges.txt";
    auto tokenizer = GPT2Tokenizer::from_files(encoder_path, vocab_path);

    // -----------------------------------------------------------------------
    // 2. Load weights
    // -----------------------------------------------------------------------
    std::string weights_path = model_dir + "/model.safetensors";
    GPT2Config cfg;  // GPT-2 small defaults
    auto weights = load_gpt2_weights(weights_path, cfg);

    // -----------------------------------------------------------------------
    // 3. Tokenize prompt
    // -----------------------------------------------------------------------
    std::vector<int> token_ids = tokenizer.encode(prompt);
    std::printf("\nPrompt: \"%s\"\n", prompt.c_str());
    std::printf("Prompt tokens: %d  |  Max new tokens: %d  |  Mode: %s\n\n",
                (int)token_ids.size(), max_new_tokens, mode.c_str());

    // -----------------------------------------------------------------------
    // 4. Prefill — process the entire prompt in one forward pass
    // -----------------------------------------------------------------------
    // After prefill, the KV-cache holds K/V for every prompt token,
    // and we get logits for predicting the first new token.
    // -----------------------------------------------------------------------
    std::printf("Prefilling prompt (%d tokens)...\n", (int)token_ids.size());
    auto t0 = std::chrono::high_resolution_clock::now();

    KVCache cache(cfg);
    Tensor logits = prefill(token_ids, weights, cfg, cache);

    auto t1 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
    std::printf("Prefill done in %.1f ms\n\n", prefill_ms);

    // -----------------------------------------------------------------------
    // 5. Autoregressive decode loop
    // -----------------------------------------------------------------------
    // On each step:
    //   a. Sample next token from logits
    //   b. Print it
    //   c. Run decode_step() with the new token → get next logits
    //   d. Repeat
    // -----------------------------------------------------------------------
    std::printf("Generated text:\n%s", prompt.c_str());
    std::fflush(stdout);

    std::mt19937 rng(42);
    std::vector<int> generated;
    int pos = (int)token_ids.size();  // position of the next token to generate
    int eos  = tokenizer.eos_token_id();

    auto decode_t0 = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < max_new_tokens; ++step) {
        // Sample next token
        int next_token;
        if (mode == "greedy") {
            next_token = greedy(logits);
        } else if (mode == "topk") {
            next_token = top_k_sample(logits, (int)k_or_p, temperature, rng);
        } else if (mode == "topp") {
            next_token = top_p_sample(logits, k_or_p, temperature, rng);
        } else {
            std::fprintf(stderr, "Unknown mode: %s\n", mode.c_str());
            return 1;
        }

        // Stop at end-of-text
        if (next_token == eos) {
            std::printf("\n[<|endoftext|>]\n");
            break;
        }

        generated.push_back(next_token);

        // Decode and print this token immediately (streaming output)
        std::string tok_str = tokenizer.decode({next_token});
        std::printf("%s", tok_str.c_str());
        std::fflush(stdout);

        // Run one decode step with the KV-cache
        logits = decode_step(next_token, pos, weights, cfg, cache);
        ++pos;

        // Safety: don't exceed max context
        if (pos >= cfg.max_seq) {
            std::printf("\n[Context limit reached]\n");
            break;
        }
    }

    auto decode_t1 = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double,std::milli>(decode_t1 - decode_t0).count();

    std::printf("\n\n---\nGenerated %d tokens in %.1f ms  (%.1f ms/token)\n",
                (int)generated.size(),
                decode_ms,
                generated.empty() ? 0.0 : decode_ms / generated.size());

    return 0;
}