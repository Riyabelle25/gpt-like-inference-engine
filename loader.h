#pragma once
// loader.h — Load real GPT-2 weights from HuggingFace safetensors format.
//
// File format layout:
//   [8 bytes]  uint64_t header_len   — byte length of the JSON header
//   [header_len bytes] JSON string   — tensor metadata
//   [rest of file]     raw float32 data (little-endian)
//
// JSON header example for one tensor:
//   "wte.weight": {
//     "dtype": "F32",
//     "shape": [50257, 768],
//     "data_offsets": [0, 154389504]   ← byte range in the data section
//   }
//
// IMP: Conv1D transpose
//   OpenAI trained GPT-2 with TensorFlow's Conv1D, which stores weight
//   matrices as {in_dim, out_dim} (input-major). THe matvec() convention
//  here is {out_dim, in_dim}. So these four weight tensors must be transposed
//   after loading:
//     *.attn.c_attn.weight   {d_model, 3*d_model}  -> {3*d_model, d_model}
//     *.attn.c_proj.weight   {d_model, d_model}     -> {d_model, d_model}
//     *.mlp.c_fc.weight      {d_model, d_ff}        -> {d_ff, d_model}
//     *.mlp.c_proj.weight    {d_ff, d_model}        -> {d_model, d_ff}
//
// Reference: Andrej Karpathy, "Line by Line, Let's Reproduce GPT-2"
//   https://towardsdatascience.com/line-by-line-lets-reproduce-gpt-2-section-1

#include "gpt2.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <stdexcept>

// ===========================================================================
// Simple JSON value — just enough to parse the safetensors header.
// We only need three fields per tensor: dtype, shape, data_offsets.
// ===========================================================================

// Extract the string value for a JSON key in a flat (non-nested) object.
// e.g. extract_str_field(s, "dtype") returns "F32"
static std::string extract_str_field(const std::string& obj, const std::string& key) {
    // Find: "key": "value"
    std::string needle = "\"" + key + "\"";
    size_t pos = obj.find(needle);
    if (pos == std::string::npos) return "";
    pos = obj.find('"', pos + needle.size());
    if (pos == std::string::npos) return "";
    size_t end = obj.find('"', pos + 1);
    if (end == std::string::npos) return "";
    return obj.substr(pos + 1, end - pos - 1);
}

// Extract data_offsets: [start, end]  ->  returns {start, end}
static std::pair<uint64_t,uint64_t> extract_offsets(const std::string& obj) {
    size_t pos = obj.find("\"data_offsets\"");
    if (pos == std::string::npos) throw std::runtime_error("no data_offsets");
    pos = obj.find('[', pos);
    uint64_t start = (uint64_t)std::stoull(obj.c_str() + pos + 1);
    pos = obj.find(',', pos);
    uint64_t end   = (uint64_t)std::stoull(obj.c_str() + pos + 1);
    return {start, end};
}

// Extract shape array: [d0, d1, ...]  ->  returns vector<int>
static std::vector<int> extract_shape(const std::string& obj) {
    size_t pos = obj.find("\"shape\"");
    if (pos == std::string::npos) throw std::runtime_error("no shape");
    pos = obj.find('[', pos);
    size_t end = obj.find(']', pos);
    std::string arr = obj.substr(pos + 1, end - pos - 1);
    std::vector<int> shape;
    size_t cur = 0;
    while (cur < arr.size()) {
        while (cur < arr.size() && (arr[cur] == ' ' || arr[cur] == ',')) ++cur;
        if (cur >= arr.size()) break;
        shape.push_back(std::stoi(arr.c_str() + cur));
        while (cur < arr.size() && arr[cur] != ',' && arr[cur] != ']') ++cur;
    }
    return shape;
}

// ===========================================================================
// Transpose 2D tensor in-place: {rows, cols} -> {cols, rows}
// ===========================================================================
static void transpose2d(Tensor& t) {
    assert(t.ndim() == 2);
    int rows = t.shape[0];
    int cols = t.shape[1];
    std::vector<float> tmp(rows * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            tmp[c * rows + r] = t.data[r * cols + c];
    t.data  = std::move(tmp);
    t.shape = {cols, rows};
}

// ===========================================================================
// SafetensorsLoader — loads the file once, then return tensors by name.
// ===========================================================================
struct SafetensorsLoader {
    std::string header_json;        // full JSON text of the header
    std::vector<char> data_region;  // raw float32 bytes after the header

    // Load the entire file into memory.
    // The data_region is everything after the 8-byte length + header.
    explicit SafetensorsLoader(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("Cannot open: " + path);

        // Read 8-byte header length
        uint64_t header_len = 0;
        if (std::fread(&header_len, sizeof(uint64_t), 1, f) != 1)
            throw std::runtime_error("Failed to read header length");

        // Read JSON header
        header_json.resize(header_len);
        if (header_len > 0) {
            if (std::fread(const_cast<char*>(header_json.data()), 1, header_len, f) != header_len)
                throw std::runtime_error("Failed to read header JSON");
        }

        // Read remaining raw data into memory
        // Seek to find file size
        std::fseek(f, 0, SEEK_END);
        long file_size = std::ftell(f);
        long data_start = 8 + (long)header_len;
        long data_size  = file_size - data_start;

        if (data_size > 0) {
            data_region.resize(data_size);
            std::fseek(f, data_start, SEEK_SET);
            if (std::fread(data_region.data(), 1, data_size, f) != (size_t)data_size)
                throw std::runtime_error("Failed to read tensor data");
        }

        std::fclose(f);
        std::printf("  Loaded safetensors: %s  (header=%llu bytes, data=%.1f MB)\n",
                    path.c_str(), (unsigned long long)header_len,
                    data_size / 1e6);
    }

    // Retrieve a tensor by its exact name in the safetensors header.
    // Copies the float32 bytes from the data_region into a new Tensor.
    Tensor get(const std::string& name) const {
        // Find the tensor's JSON block: locate "name": { ... }
        std::string key = "\"" + name + "\"";
        size_t pos = header_json.find(key);
        if (pos == std::string::npos)
            throw std::runtime_error("Tensor not found: " + name);

        // Find the opening brace of this tensor's metadata object
        size_t brace = header_json.find('{', pos + key.size());
        if (brace == std::string::npos)
            throw std::runtime_error("Malformed header for: " + name);

        // Find the closing brace (simple scan — no nesting in these objects)
        size_t brace_end = header_json.find('}', brace);
        std::string obj = header_json.substr(brace, brace_end - brace + 1);

        // Parse shape and byte offsets
        auto shape = extract_shape(obj);
        auto [byte_start, byte_end] = extract_offsets(obj);

        // Validate: dtype must be F32
        std::string dtype = extract_str_field(obj, "dtype");
        if (dtype != "F32")
            throw std::runtime_error("Expected F32, got " + dtype + " for " + name);

        // Copy raw bytes into Tensor
        size_t n_floats = (byte_end - byte_start) / sizeof(float);
        Tensor t(shape);
        assert(t.data.size() == n_floats);
        std::memcpy(t.data.data(),
                    data_region.data() + byte_start,
                    n_floats * sizeof(float));
        return t;
    }
};


// ===========================================================================
// load_gpt2_weights() — map HuggingFace tensor names to GPT2Weights struct
// ===========================================================================
// HuggingFace GPT-2 safetensors names follow this scheme:
//   wte.weight          -> wte
//   wpe.weight          -> wpe
//   ln_f.weight/bias    -> ln_f_gamma/beta
//   h.{i}.ln_1.weight   -> blocks[i].ln1_gamma
//   h.{i}.ln_1.bias     -> blocks[i].ln1_beta
//   h.{i}.attn.c_attn.weight -> blocks[i].c_attn_w  (TRANSPOSE!)
//   h.{i}.attn.c_attn.bias   -> blocks[i].c_attn_b
//   h.{i}.attn.c_proj.weight -> blocks[i].c_proj_w  (TRANSPOSE!)
//   h.{i}.attn.c_proj.bias   -> blocks[i].c_proj_b
//   h.{i}.ln_2.weight   -> blocks[i].ln2_gamma
//   h.{i}.ln_2.bias     -> blocks[i].ln2_beta
//   h.{i}.mlp.c_fc.weight   -> blocks[i].mlp_fc_w   (TRANSPOSE!)
//   h.{i}.mlp.c_fc.bias     -> blocks[i].mlp_fc_b
//   h.{i}.mlp.c_proj.weight -> blocks[i].mlp_proj_w (TRANSPOSE!)
//   h.{i}.mlp.c_proj.bias   -> blocks[i].mlp_proj_b
// ===========================================================================
inline GPT2Weights load_gpt2_weights(const std::string& safetensors_path,
                                      const GPT2Config& cfg) {
    cfg.validate();
    SafetensorsLoader loader(safetensors_path);
    GPT2Weights w;

    std::printf("Loading weights...\n");

    // --- Embeddings ---
    w.wte = loader.get("wte.weight");   // {50257, 768}
    w.wpe = loader.get("wpe.weight");   // {1024, 768}

    // --- Final LayerNorm ---
    w.ln_f_gamma = loader.get("ln_f.weight");
    w.ln_f_beta  = loader.get("ln_f.bias");

    // --- Transformer blocks ---
    w.blocks.resize(cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; ++i) {
        std::string pfx = "h." + std::to_string(i);
        BlockWeights& b = w.blocks[i];

        // LayerNorm 1
        b.ln1_gamma = loader.get(pfx + ".ln_1.weight");
        b.ln1_beta  = loader.get(pfx + ".ln_1.bias");

        // QKV projection — MUST transpose: stored as {d_model, 3*d_model}
        // After transpose: {3*d_model, d_model}  (our {out, in} convention)
        b.c_attn_w = loader.get(pfx + ".attn.c_attn.weight");
        transpose2d(b.c_attn_w);
        b.c_attn_b = loader.get(pfx + ".attn.c_attn.bias");

        // Attention output projection — MUST transpose: {d_model, d_model}
        // After transpose: still {d_model, d_model} but transposed internally
        b.c_proj_w = loader.get(pfx + ".attn.c_proj.weight");
        transpose2d(b.c_proj_w);
        b.c_proj_b = loader.get(pfx + ".attn.c_proj.bias");

        // LayerNorm 2
        b.ln2_gamma = loader.get(pfx + ".ln_2.weight");
        b.ln2_beta  = loader.get(pfx + ".ln_2.bias");

        // FFN first linear — MUST transpose: {d_model, d_ff} -> {d_ff, d_model}
        b.mlp_fc_w = loader.get(pfx + ".mlp.c_fc.weight");
        transpose2d(b.mlp_fc_w);
        b.mlp_fc_b = loader.get(pfx + ".mlp.c_fc.bias");

        // FFN second linear — MUST transpose: {d_ff, d_model} -> {d_model, d_ff}
        b.mlp_proj_w = loader.get(pfx + ".mlp.c_proj.weight");
        transpose2d(b.mlp_proj_w);
        b.mlp_proj_b = loader.get(pfx + ".mlp.c_proj.bias");

        if ((i + 1) % 4 == 0)
            std::printf("  Loaded blocks 0..%d\n", i);
    }

    std::printf("All weights loaded.\n");
    return w;
}