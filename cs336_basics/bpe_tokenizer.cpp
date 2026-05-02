#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl//queue.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_set.h>

struct MergedPair {
    std::string merged_str;
    uint32_t frequency;
};

auto cmp = [](const MergedPair& a, const MergedPair& b) {
    return a.frequency != b.frequency ? a.frequency > b.frequency : a > b;
};

std::priority_queue<MergedPair, std::vector<MergedPair>, decltype(cmp)> freq_by_pair;


void initialize_priority_queue(const std::unordered_map<std::string, int>& freqs) 
{
    std::unordered_set<std::string> mapped_pairs;
    auto words = freqs.keys();

    for (int j = 0; j < words.size(); j++) {
        auto word = words[j];
        for (int i = 0; i < word.size()-1; i++) {
            int total_freq = 0;
            std::string merged_pair = std::string(word[i])+std::string(word[i+1]);
            if (mapped_pairs.find(merged_pair) != mapped_pairs.end()) continue;

            for (int k = j; k < words.size(); k++) {
                int count = 0;                                                                                                        
                size_t pos = 0;
                while ((pos = words[k].find(merged_pair, pos)) != std::string::npos) {                                                         
                    count++;                                                                                                          
                    pos += merged_pair.size();
                }
                total_freq += count*freqs[words[k]]
            }
            mapped_pairs.insert(merged_pair);
            freq_by_pair.push(MergedPair{.merged_str=merged_pair, .frequency=total_freq});
        }
    }
}

void update_freqs_after_merge(std::unordered_map<str, int>& freqs)
{

}
