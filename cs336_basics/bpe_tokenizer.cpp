#include <nanobind/nanobind.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_set.h>
#include <memory>
#include <queue>
#include <algorithm>

struct MergedPair {
    std::string merged_str;
    int frequency;
    std::vector<std::string> words;
};

struct TrainedTokenizer {
    std::vector<std::string> merged_pairs;
    std::unordered_map<std::string, int> token_ids;
};

auto cmp = [](const MergedPair& a, const MergedPair& b) {
    return a.frequency != b.frequency ? a.frequency > b.frequency : a.merged_str > b.merged_str;
};

std::priority_queue<MergedPair, std::vector<MergedPair>, decltype(cmp)> freq_by_pair;
std::unordered_set<std::string> invalid_merged_pairs;

void initialize_priority_queue(std::unordered_map<std::string, int>& freqs) 
{
    std::unordered_set<std::string> mapped_pairs;
    std::vector<std::string> words(freqs.size());
    std::transform(freqs.begin(), freqs.end(), words.begin(), [](const auto& a) -> std::string {
        return a.first;
    });

    std::vector<MergedPair> pairs;
    //Reserve initial guess for numbe of pairs
    pairs.reserve(freqs.size()*2);

    for (int j = 0; j < words.size(); j++) {
        auto word = words[j];
        for (int i = 0; i < word.size()-1; i++) {
            int total_freq = 0;
            std::string merged_pair = std::string(1, word[i])+std::string(1, word[i+1]);
            if (mapped_pairs.find(merged_pair) != mapped_pairs.end()) continue;
            std::vector<std::string> words_pairs_exists;
            for (int k = j; k < words.size(); k++) {
                int count = 0;                                                                                                        
                size_t pos = 0;
                while ((pos = words[k].find(merged_pair, pos)) != std::string::npos) {                                                         
                    count++;                                                                                                          
                    pos += merged_pair.size();
                }
                total_freq += count*freqs[words[k]];
                if (count) {
                    words_pairs_exists.push_back(words[k]);
                }
            }
            mapped_pairs.insert(merged_pair);
            pairs.push_back(MergedPair{.merged_str=merged_pair, .frequency=total_freq, .words=std::move(words_pairs_exists)});
        }
    }
    decltype(freq_by_pair) tmp(pairs.begin(), pairs.end());
    std::swap(freq_by_pair, tmp);
}

void update_freqs_after_merge(std::unordered_map<std::string, int>& freqs, MergedPair& merged_pair)
{
    std::unordered_map<std::string, int> unmerged_freqs;
    std::unordered_map<std::string, std::unordered_set<std::string>> unmerged_words;

    for (auto word : merged_pair.words) {
        // update only those words where merged_str appears
        size_t pos = 0;
        while ((pos = word.find(merged_pair.merged_str, pos)) != std::string::npos) {                                                         
            if (pos) {
                // update prefix of str
                invalid_merged_pairs.insert(word.substr(pos-1, merged_pair.merged_str.size()));
                std::string new_pair = std::string(1, word[pos-1]) + merged_pair.merged_str;
                unmerged_freqs[new_pair] += freqs[word];
                unmerged_words[new_pair].insert(word);
            }

            if (pos + merged_pair.merged_str.size() < word.size()) {
                invalid_merged_pairs.insert(word.substr(pos+1, merged_pair.merged_str.size()));
                std::string new_pair = merged_pair.merged_str+std::string(1, word[pos+merged_pair.merged_str.size()]);
                unmerged_freqs[new_pair] += freqs[word];
                unmerged_words[new_pair].insert(word);
            }                                                                                                          
            pos += merged_pair.merged_str.size();
        }
    }

    for (auto merged : unmerged_freqs) {
        freq_by_pair.push(MergedPair{.merged_str=std::move(merged.first), .frequency=merged.second, .words=std::vector<std::string>(unmerged_words[merged.first].begin(), unmerged_words[merged.first].end())});
    }
}

TrainedTokenizer train_bpe_tokenizer(std::unordered_map<std::string, int> freqs, int vocab_size, int init_token_id)
{
    auto cur_token_id = ++init_token_id;
    std::vector<std::string> merged_pairs;
    merged_pairs.reserve(vocab_size);

    initialize_priority_queue(freqs);
    
    std::unordered_map<std::string, int> token_ids;

    while (vocab_size && !freq_by_pair.empty()) {
        auto max_freq_pair = freq_by_pair.top();
        freq_by_pair.pop();
        if (invalid_merged_pairs.find(max_freq_pair.merged_str) != invalid_merged_pairs.end()) {
            continue;
        }

        update_freqs_after_merge(freqs, max_freq_pair);
        merged_pairs.push_back(std::move(max_freq_pair.merged_str));
        token_ids[merged_pairs.back()] = cur_token_id++;
        vocab_size--;
    }
    return TrainedTokenizer{.merged_pairs=std::move(merged_pairs), .token_ids=std::move(token_ids)};
}

NB_MODULE(bpe_tokenizer, m) {
    m.def("train_bpe_tokenizer", &train_bpe_tokenizer, nanobind::rv_policy::move);

    nanobind::class_<TrainedTokenizer>(m, "TrainedTokenizer")
        .def(nanobind::init<decltype(std::declval<TrainedTokenizer>().merged_pairs), decltype(std::declval<TrainedTokenizer>().token_ids)>())
        .def_rw("merged_pairs", &TrainedTokenizer::merged_pairs)
        .def_rw("token_ids", &TrainedTokenizer::token_ids);
}
