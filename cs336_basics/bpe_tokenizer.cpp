#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl//queue.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_set.h>
#include <memory>

struct MergedPair {
    std::string merged_str;
    uint32_t frequency;
    std::vector<std::string>& words;
};

struct TrainedTokenizer {
    std::vector<std::string> merged_pairs;
    std::unordered_map<std::string, uint32_t> token_ids;
};

auto cmp = [](const MergedPair& a, const MergedPair& b) {
    return a.frequency != b.frequency ? a.frequency > b.frequency : a > b;
};

std::priority_queue<MergedPair, std::vector<MergedPair>, decltype(cmp)> freq_by_pair;


void initialize_priority_queue(const std::unordered_map<std::string, int>& freqs) 
{
    std::unordered_set<std::string> mapped_pairs;
    auto words = freqs.keys();
    std::vector<MergedPair> pairs;
    //Reserve initial guess for numbe of pairs
    pairs.reserve(freqs.keys().size()*2);

    for (int j = 0; j < words.size(); j++) {
        auto word = words[j];
        for (int i = 0; i < word.size()-1; i++) {
            int total_freq = 0;
            std::string merged_pair = std::string(word[i])+std::string(word[i+1]);
            if (mapped_pairs.find(merged_pair) != mapped_pairs.end()) continue;
            std::vector<std::string> words_pair_exists;
            for (int k = j; k < words.size(); k++) {
                int count = 0;                                                                                                        
                size_t pos = 0;
                while ((pos = words[k].find(merged_pair, pos)) != std::string::npos) {                                                         
                    count++;                                                                                                          
                    pos += merged_pair.size();
                }
                total_freq += count*freqs[words[k]];
                if (count) {
                    words_pairs_exist.push_back(words[k]);
                }
            }
            mapped_pairs.insert(merged_pair);
            pairs.push_back(MergedPair{.merged_str=merged_pair, .frequency=total_freq, .words=std::move(words_pairs_exist)});
            //freq_by_pair.push(MergedPair{.merged_str=merged_pair, .frequency=total_freq});
        }
    }
    decltype(freq_by_pair) tmp(pairs.begin(), pairs.end());
    std::swap(freq_by_pair, tmp);
}

void update_freqs_after_merge(std::unordered_map<std::string, int>& freqs, MergedPair& merged_pair)
{
    for (auto word : merged_pair.words) {
        // update only those words where merged_str appears
    }
}

TrainedTokenizer train_bpe_tokenizer(std::unordered_map<std::string, int> freqs, int vocab_size, int init_token_id)
{
    auto cur_token_id = ++init_token_id;
    std::vector<std::string> merged_pairs;
    merged_pairs.reserve(vocab_size);

    std::unordered_map<std::string, int> token_ids;

    while (vocab_size-- && !freq_by_pair.empty()) {
        auto max_freq_pair = freq_by_pair.top();
        freq_by_pair.pop();
        merged_pairs.push_back(std::move(max_freq_pair.merged_str));
        token_ids[merged_pairs.back()] = cur_token_id++;

        update_freqs_after_merge(freqs);
    }
    return TrainedTokenizerue{.merged_pairs=std::move(merged_pairs), .token_ids=std::move(token_ids)};
}

NB_MODULE(bpe_tokenizer, m) {
    m.def()

    nb::class_<TrainedTokenizer>(m, "TrainedTokenizer")
        .def(nb::init<decltype(std::declval<TrainedTokenizer>().merged_pairs), decltype(std::declval<TrainedTokenizer>().token_ids)>())
        .def_rw("merged_pairs", &TrainedTokenizer::merged_pairs)
        .def_rw("token_ids", &TrainedTokenizer::token_ids);
}
