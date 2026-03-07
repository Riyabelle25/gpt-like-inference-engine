#pragma once
// tokenizer.h — Minimal GPT-2 BPE tokenizer.
//
// GPT-2 uses Byte Pair Encoding (BPE), which works like this:
//
//   Inference (what we implement here):
//     1. Convert the input string to UTF-8 bytes, then apply a character-level
//        mapping (the "byte encoder") so every byte maps to a printable char.
//     2. Pre-tokenize on whitespace/punctuation (GPT-2 uses a regex for this;
//        we use a simplified version that handles the common cases).
//     3. For each word, apply BPE merges greedily in priority order
//        (the order they appear in merges.txt) until no more merges apply.
//     4. Look up each final sub-token in encoder.json to get its integer ID.
//
// Files needed (download from HuggingFace openai-community/gpt2):
//   encoder.json  — JSON dict: string token -> integer ID
//   merges.txt     — text file: one merge rule per line, e.g. "Ġt he"

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>

// ===========================================================================
// Byte encoder: maps each raw byte (0–255) to a unique printable Unicode char.
// ===========================================================================
// The forward map (byte -> char) is used when encoding.
// The reverse map (char -> byte) is used when decoding.
static std::unordered_map<int, std::string> make_byte_encoder() {

    std::unordered_map<int, std::string> encoder;
    for (int b = 0; b < 256; ++b) {
        if ((b >= 33 && b <= 126) || (b >= 160 && b <= 255)) {
            // These bytes map to the Unicode codepoint with the same value.
            // UTF-8 encode that codepoint.
            char buf[5] = {};
            if (b < 128) {
                buf[0] = (char)b;                        // 1-byte UTF-8
            } else {
                buf[0] = (char)(0xC0 | (b >> 6));       // 2-byte UTF-8
                buf[1] = (char)(0x80 | (b & 0x3F));
            }
            encoder[b] = std::string(buf);
        }
    }
    // The remaining 256 - len(printable_bytes) bytes map to U+0100 onwards
    int n = 256;
    for (int b = 0; b < 256; ++b) {
        if (encoder.find(b) == encoder.end()) {
            // Encode Unicode codepoint n as UTF-8
            char buf[5] = {};
            if (n < 0x80) {
                buf[0] = (char)n;
            } else if (n < 0x800) {
                buf[0] = (char)(0xC0 | (n >> 6));
                buf[1] = (char)(0x80 | (n & 0x3F));
            } else {
                buf[0] = (char)(0xE0 | (n >> 12));
                buf[1] = (char)(0x80 | ((n >> 6) & 0x3F));
                buf[2] = (char)(0x80 | (n & 0x3F));
            }
            encoder[b] = std::string(buf);
            ++n;
        }
    }
    return encoder;
}

// Build reverse map: UTF-8 string -> original byte value
static std::unordered_map<std::string, int> make_byte_decoder(
    const std::unordered_map<int, std::string>& enc) {
    std::unordered_map<std::string, int> dec;
    for (auto& [b, s] : enc) dec[s] = b;
    return dec;
}

// ===========================================================================
// ===========================================================================
// JSON string parser: extract a flat string->int map from vocab.json.
// ===========================================================================
// vocab.json looks like: {"!": 0, "\"": 1, ..., "\u0120world": 995, ...}
//
// IMPORTANT: HuggingFace's vocab.json stores non-ASCII tokens as JSON
// \uXXXX escape sequences, e.g. Ġ (U+0120) is written as \u0120.
// We must decode these into proper UTF-8 strings so our map keys match
// the UTF-8 strings produced by the byte encoder at encode() time.

static std::string codepoint_to_utf8(uint32_t cp) {
    char buf[4] = {};
    if (cp < 0x80) {
        buf[0] = (char)cp;
        return std::string(buf, 1);
    } else if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        return std::string(buf, 2);
    } else {
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F));
        return std::string(buf, 3);
    }
}

// Parse a JSON string literal starting right after the opening quote.
// Returns the decoded string and advances `pos` to just past the closing quote.
static std::string parse_json_string(const std::string& s, size_t& pos) {
    std::string out;
    while (pos < s.size()) {
        char c = s[pos];
        if (c == '"') { ++pos; break; }  // closing quote
        if (c == '\\' && pos + 1 < s.size()) {
            char esc = s[pos + 1];
            pos += 2;
            if (esc == 'u' && pos + 4 <= s.size()) {
                // \uXXXX — parse 4 hex digits into a codepoint, then UTF-8-encode
                uint32_t cp = 0;
                for (int i = 0; i < 4; ++i) {
                    cp <<= 4;
                    char h = s[pos++];
                    if      (h >= '0' && h <= '9') cp |= (h - '0');
                    else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
                }
                out += codepoint_to_utf8(cp);
            } else if (esc == '"')  { out += '"';  }
            else if (esc == '\\') { out += '\\'; }
            else if (esc == '/')  { out += '/';  }
            else if (esc == 'n')  { out += '\n'; }
            else if (esc == 'r')  { out += '\r'; }
            else if (esc == 't')  { out += '\t'; }
            else                  { out += esc;  }
        } else {
            out += c;
            ++pos;
        }
    }
    return out;
}

static std::unordered_map<std::string, int> parse_encoder_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open vocab.json: " + path);
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    std::unordered_map<std::string, int> enc;
    size_t pos = 0;
    while (pos < content.size()) {
        // Find next opening quote (start of a key)
        size_t q1 = content.find('"', pos);
        if (q1 == std::string::npos) break;
        pos = q1 + 1;

        // Parse the key string (handles \uXXXX and other escapes)
        std::string key = parse_json_string(content, pos);

        // Skip whitespace then expect ':'
        while (pos < content.size() && content[pos] == ' ') ++pos;
        if (pos >= content.size() || content[pos] != ':') continue;
        ++pos;

        // Skip whitespace then parse the integer value
        while (pos < content.size() && content[pos] == ' ') ++pos;
        if (pos < content.size() && std::isdigit((unsigned char)content[pos])) {
            int val = std::stoi(content.c_str() + pos);
            enc[key] = val;
            while (pos < content.size() && std::isdigit((unsigned char)content[pos])) ++pos;
        }
    }
    return enc;
}

// ===========================================================================
// GPT2Tokenizer
// ===========================================================================
struct GPT2Tokenizer {
    // token string -> integer ID (from encoder.json)
    std::unordered_map<std::string, int> encoder;
    // integer ID -> token string (reverse)
    std::unordered_map<int, std::string> decoder;

    std::map<std::pair<std::string,std::string>, int> bpe_ranks;

    // Byte -> character mappings
    std::unordered_map<int,std::string>    byte_enc;  // byte -> UTF-8 str
    std::unordered_map<std::string,int>    byte_dec;  // UTF-8 str -> byte

    static constexpr const char* SPACE_PREFIX = "\xC4\xa0"; // UTF-8 for U+0120

    GPT2Tokenizer() = default;

    // Load from the two vocab files.
    static GPT2Tokenizer from_files(const std::string& encoder_json_path,
                                     const std::string& vocab_bpe_path) {
        GPT2Tokenizer tok;

        // Build byte encoder/decoder
        tok.byte_enc = make_byte_encoder();
        tok.byte_dec = make_byte_decoder(tok.byte_enc);

        // Load encoder.json
        tok.encoder = parse_encoder_json(encoder_json_path);
        for (auto& [s, id] : tok.encoder)
            tok.decoder[id] = s;

        // Load merges.txt - each line is "token_a token_b" (space-separated)
        // Line 0 is a comment (#version...), skip it.
        // The rank of a merge rule = its line number (0-indexed after the comment).
        std::ifstream f(vocab_bpe_path);
        if (!f) throw std::runtime_error("Cannot open merges.txt: " + vocab_bpe_path);
        std::string line;
        int rank = 0;
        bool first = true;
        while (std::getline(f, line)) {
            if (first) { first = false; continue; } // skip #version line
            if (line.empty()) continue;
            size_t sp = line.find(' ');
            if (sp == std::string::npos) continue;
            std::string a = line.substr(0, sp);
            std::string b = line.substr(sp + 1);
            tok.bpe_ranks[{a, b}] = rank++;
        }

        printf("Tokenizer loaded: %zu tokens, %zu BPE merges\n",
               tok.encoder.size(), tok.bpe_ranks.size());
        return tok;
    }

    // -----------------------------------------------------------------------
    // BPE encode one "word" (a pre-tokenized chunk, already byte-encoded).
    // -----------------------------------------------------------------------
    // Starts with the word split into individual characters (each is a
    // byte-encoded token), then greedily merges the highest-priority pair
    // until no more merges apply.
    //
    // The characters in `word` are UTF-8 multi-byte strings representing
    // individual bytes after the byte encoder has been applied.
    std::vector<std::string> bpe(const std::string& word) const {
        // Split word into individual byte-encoded characters
        // Each "character" here may be a 1-3 byte UTF-8 sequence
        std::vector<std::string> chars;
        size_t i = 0;
        while (i < word.size()) {
            unsigned char c = (unsigned char)word[i];
            int char_len = 1;
            if      (c >= 0xF0) char_len = 4;
            else if (c >= 0xE0) char_len = 3;
            else if (c >= 0xC0) char_len = 2;
            chars.push_back(word.substr(i, char_len));
            i += char_len;
        }

        if (chars.size() <= 1) return chars;

        // Repeatedly find and apply the highest-priority merge
        while (true) {
            // Find the pair with the lowest rank (= highest priority)
            int best_rank = INT_MAX;
            int best_i    = -1;
            for (int j = 0; j + 1 < (int)chars.size(); ++j) {
                auto it = bpe_ranks.find({chars[j], chars[j+1]});
                if (it != bpe_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i    = j;
                }
            }
            if (best_i == -1) break; // no more merges

            // Merge chars[best_i] + chars[best_i+1] -> one token
            std::string merged = chars[best_i] + chars[best_i + 1];
            chars.erase(chars.begin() + best_i + 1);
            chars[best_i] = merged;
        }
        return chars;
    }

    // -----------------------------------------------------------------------
    // encode(text) -> vector<int> token IDs
    // -----------------------------------------------------------------------
    // GPT-2 pre-tokenizes using a specific regex. We approximate it:
    //   - A word can start with a space (-> Ġ prefix in the vocab)
    //   - Punctuation is split off
    //   - Contractions like "'s", "'t" are kept together
    //
    // For simplicity and correctness on typical inputs we do:
    //   Split on spaces/newlines but attach the leading space to the NEXT word.
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        std::vector<std::string> words;
        std::string cur;
        bool at_word_start = true;

        for (size_t idx = 0; idx < text.size(); ++idx) {
            unsigned char c = (unsigned char)text[idx];

            if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
                if (!cur.empty()) {
                    words.push_back(cur);
                    cur.clear();
                    at_word_start = false; // next word follows a space → gets Ġ
                }
            } else {
                if (!at_word_start && cur.empty()) {
                    // Prepend the Ġ character (byte_enc[' ']) to this word
                    cur += byte_enc.at(' ');
                }
                // Map this single raw byte through byte_enc.
                cur += byte_enc.at(c);
            }
        }
        if (!cur.empty()) words.push_back(cur);

        // BPE-encode each word and look up IDs
        for (const std::string& word : words) {
            auto tokens = bpe(word);
            for (const std::string& tok : tokens) {
                auto it = encoder.find(tok);
                if (it == encoder.end())
                    throw std::runtime_error("Unknown BPE token: " + tok);
                ids.push_back(it->second);
            }
        }
        return ids;
    }

    // -----------------------------------------------------------------------
    // decode(ids) -> string
    // -----------------------------------------------------------------------
    // Reverse: look up each ID in the decoder map, concatenate the byte-encoded
    // strings, then apply the byte decoder to get back the original bytes.
    std::string decode(const std::vector<int>& ids) const {
        std::string byte_str;
        for (int id : ids) {
            auto it = decoder.find(id);
            if (it == decoder.end())
                throw std::runtime_error("Unknown token id: " + std::to_string(id));
            byte_str += it->second;
        }

        // Reverse byte encoding: convert each byte-encoded char back to raw bytes
        // We scan the string recognising 1, 2, or 3 byte UTF-8 sequences
        // and map each back to its original byte value.
        std::string result;
        size_t i = 0;
        while (i < byte_str.size()) {
            unsigned char c = (unsigned char)byte_str[i];
            int clen = 1;
            if      (c >= 0xF0) clen = 4;
            else if (c >= 0xE0) clen = 3;
            else if (c >= 0xC0) clen = 2;

            std::string encoded_char = byte_str.substr(i, clen);
            auto it = byte_dec.find(encoded_char);
            if (it != byte_dec.end()) {
                result += (char)it->second;
            } else {
                // Pass through unknown sequences
                result += encoded_char;
            }
            i += clen;
        }
        return result;
    }

    // Convenience: get the end-of-text token ID (GPT-2 uses 50256)
    int eos_token_id() const {
        auto it = encoder.find("<|endoftext|>");
        if (it == encoder.end()) return 50256;
        return it->second;
    }
};