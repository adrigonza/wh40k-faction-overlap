# Sample Output Description

This file describes the intended output format. The numbers below are illustrative examples only.

## `outputs/faction_subreddit_overlap.csv`

| faction | other_subreddit | users_in_faction_sample | users_seen_in_other_sub | probability | baseline_probability | lift | log2_lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tyranids | minipainting | 80 | 27 | 0.3375 | 0.1800 | 1.8750 | 0.9069 |
| Orks | kitbash | 80 | 23 | 0.2875 | 0.1000 | 2.8750 | 1.5236 |
| Necrons | PrintedWarhammer | 80 | 17 | 0.2125 | 0.1100 | 1.9318 | 0.9500 |

## `outputs/faction_similarity.csv`

A square faction-by-faction similarity matrix generated from each faction's normalized subreddit overlap profile.

## Graphs

The script generates PNG graphs in the `outputs/` folder:

- `faction_to_subreddit_network.png`
- `faction_similarity_network.png`
