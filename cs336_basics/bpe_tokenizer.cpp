#include <nanobind/nanobind.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_set.h>
#include <memory>
#include <set>
#include <utility>
#include <array>
#include <algorithm>
#include <type_traits>
#include <functional>
#include <iostream>


struct MergedPair {
    std::string merged_1, merged_2;
    int frequency;
    std::unordered_set<std::string> words;

    bool operator==(const MergedPair& other) {
        return this->merged_1 == other.merged_1 && this->merged_2 == other.merged_2;
    }
};

struct TrainedTokenizer {
    std::vector<std::vector<std::string>> merged_pairs;
    std::vector<std::string> token_ids;
};

template <typename T>
concept IsHashable = std::is_invocable_v<std::hash<T>, const T&>;

template <IsHashable T>
struct Hasher {
    std::size_t operator()(const std::pair<T,T>& p) const noexcept {
        std::size_t h1 = std::hash<T>{}(p.first);
        std::size_t h2 = std::hash<T>{}(p.second);
        // boost::hash_combine's mixing constant
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

auto cmp = [](const MergedPair& a, const MergedPair& b) {
    if (a.merged_1 == b.merged_1 && a.merged_2 == b.merged_2) return false;
    if (a.frequency == b.frequency) {
        if (a.merged_1 == b.merged_1) return a.merged_2 > b.merged_2;
        return a.merged_1 > b.merged_1;
    }
    return a.frequency > b.frequency;
};

std::set<MergedPair, decltype(cmp)> freq_by_pair;
std::unordered_map<std::string, std::vector<std::string>> word_to_prefix, word_to_suffix;
std::unordered_map<std::pair<std::string, std::string>, int, Hasher<std::string>> merged_freq_cache;

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
            std::string merged_1 = std::string(1, word[i]);
            std::string merged_2 = std::string(1, word[i+1]);
            std::string merged_pair = merged_1+merged_2;
            if (mapped_pairs.find(merged_pair) != mapped_pairs.end()) continue;
            std::unordered_set<std::string> words_pairs_exists;
            for (int k = j; k < words.size(); k++) {
                int count = 0;                                                                                                        
                size_t pos = 0;
                while ((pos = words[k].find(merged_pair, pos)) != std::string::npos) {                                                         
                    count++;
                    if (word_to_suffix.find(words[k]) == word_to_suffix.end()) {
                        word_to_suffix[words[k]] = std::vector<std::string>(words[k].size());
                        word_to_prefix[words[k]] = std::vector<std::string>(words[k].size());
                    }
                    word_to_suffix[words[k]][pos] = merged_2;
                    word_to_prefix[words[k]][pos+1] = merged_1;                                                                                                          
                    pos += merged_pair.size();
                }
                total_freq += count*freqs[words[k]];
                if (count) {
                    words_pairs_exists.insert(words[k]);
                }
            }
            mapped_pairs.insert(std::move(merged_pair));
            merged_freq_cache[{merged_1, merged_2}] = total_freq;
            pairs.push_back(MergedPair{.merged_1=std::move(merged_1), .merged_2=std::move(merged_2), .frequency=total_freq, .words=std::move(words_pairs_exists)});
        }
    }
    decltype(freq_by_pair) tmp(pairs.begin(), pairs.end());
    std::swap(freq_by_pair, tmp);
}

void update_freqs_after_merge(std::unordered_map<std::string, int>& freqs, MergedPair& merged_pair)
{
    std::unordered_map<std::string, MergedPair> new_merged_candidates;
    std::unordered_map<std::pair<std::string, std::string>, std::pair<int, std::unordered_set<std::string>>, Hasher<std::string>> updated_freqs;

    for (auto word : merged_pair.words) {
        // update only those words where merged_str appears
        size_t pos = 0;
        std::string merged_str = merged_pair.merged_1+merged_pair.merged_2;
        while ((pos = word.find(merged_str, pos)) != std::string::npos) {                                                         
            if (pos) {
                // here we need to reduce the frequency of prefix+merged_1 by freqs[word]
                // and update the prefix of word[pos+merged_1.size()] to be the prefix of merged_1
                updated_freqs[{word_to_prefix[word][pos], merged_pair.merged_1}].first += freqs[word];
                updated_freqs[{word_to_prefix[word][pos], merged_pair.merged_1}].second.insert(word);

                // Create new merged candidate of prefix+merged_1+merged2

                if (new_merged_candidates.find(word_to_prefix[word][pos]+merged_str) == new_merged_candidates.end()) {
                    new_merged_candidates[word_to_prefix[word][pos]+merged_str] = MergedPair{
                        .merged_1 = word_to_prefix[word][pos],
                        .merged_2 = merged_str,
                        .frequency = 0,
                        .words=std::unordered_set<std::string>()
                    };
                }
                new_merged_candidates[word_to_prefix[word][pos]+merged_str].frequency += freqs[word];
                new_merged_candidates[word_to_prefix[word][pos]+merged_str].words.insert(word);

                // Change suffix of prefix of merged_1 to be merged_str
                //word_to_suffix[word][pos-word_to_prefix[word][pos].size()] = merged_str;
            }

            if (pos + merged_str.size() < word.size()) {

                // Create new merged candidate of merged_1+merged2+suffix
                updated_freqs[{merged_pair.merged_2, word_to_suffix[word][pos+merged_pair.merged_1.size()]}].first += freqs[word];
                updated_freqs[{merged_pair.merged_2, word_to_suffix[word][pos+merged_pair.merged_1.size()]}].second.insert(word);

                if (new_merged_candidates.find(merged_str+word_to_suffix[word][pos+merged_pair.merged_1.size()]) == new_merged_candidates.end()) {
                    new_merged_candidates[merged_str+word_to_suffix[word][pos+merged_pair.merged_1.size()]] = MergedPair{
                        .merged_1 = merged_str,
                        .merged_2 = word_to_suffix[word][pos+merged_pair.merged_1.size()],
                        .frequency = 0,
                        .words=std::unordered_set<std::string>()
                    };
                }
                new_merged_candidates[merged_str+word_to_suffix[word][pos+merged_pair.merged_1.size()]].frequency += freqs[word];
                new_merged_candidates[merged_str+word_to_suffix[word][pos+merged_pair.merged_1.size()]].words.insert(word);

                // Change prefix of suffix of merged_2 to be merged_str
                word_to_prefix[word][pos+merged_str.size()] = merged_str;
            }
            word_to_prefix[word][pos+merged_pair.merged_1.size()] = word_to_prefix[word][pos];
            word_to_suffix[word][pos] = word_to_suffix[word][pos+merged_pair.merged_1.size()];

            pos += merged_str.size();
        }
    }
    std::for_each(new_merged_candidates.begin(), new_merged_candidates.end(), [&freq_by_pair](const auto& p) -> void {
        freq_by_pair.insert(p.second);
    });
    
    std::for_each(updated_freqs.begin(), updated_freqs.end(), [&freq_by_pair, &merged_freq_cache](const auto& p) -> void {
        MergedPair key;
        key.merged_1 = p.first.first;
        key.merged_2 = p.first.second;
        key.frequency = merged_freq_cache[{key.merged_1, key.merged_2}];
        auto itr = freq_by_pair.find(std::move(key));
        if (itr == freq_by_pair.end()) return;
        auto it = *itr;
        freq_by_pair.erase(it);
        it.frequency -= p.second.first;
        if (!it.frequency) return;
        std::for_each(p.second.second.begin(), p.second.second.end(), [&it](const auto& val) -> void {
            it.words.erase(val);
        });
        merged_freq_cache[{it.merged_1, it.merged_2}] = it.frequency;
        freq_by_pair.insert(std::move(it));
    });


}

TrainedTokenizer train_bpe_tokenizer(std::unordered_map<std::string, int> freqs, int n_merges, int init_token_id)
{
    auto cur_token_id = ++init_token_id;
    std::vector<std::vector<std::string>> merged_pairs;
    merged_pairs.reserve(n_merges);

    initialize_priority_queue(freqs);
    
    std::cout << freq_by_pair.size() << std::endl;
    std::vector<std::string> token_ids(cur_token_id+n_merges+1);

    while (n_merges && !freq_by_pair.empty()) {
        auto max_freq_pair = *freq_by_pair.begin();
        freq_by_pair.erase(max_freq_pair);

        update_freqs_after_merge(freqs, max_freq_pair);
        merged_pairs.push_back({std::move(max_freq_pair.merged_1), std::move(max_freq_pair.merged_2)});
        token_ids[cur_token_id++] = (merged_pairs.back()[0]+merged_pairs.back()[1]);
        n_merges--;
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
