# Warhammer 40k Faction Overlap

A non-commercial Python project for analyzing public Reddit community overlap among Warhammer 40,000 faction communities.

The goal is to create aggregate network graphs showing how different Warhammer 40k faction subreddits relate to other public Reddit communities based on public posting and commenting activity.

This project is intended for hobby research, community visualization, and data analysis practice.

## Project goals

This tool aims to answer questions such as:

- Which other public subreddits are commonly visited by users from each Warhammer 40k faction community?
- Which faction communities have similar Reddit participation patterns?
- Are some factions more connected to painting, lore, competitive play, kitbashing, or tabletop gaming communities?
- How do raw subreddit overlap probabilities compare with normalized affinity scores?

## What the project does

The planned Python workflow is:

1. Read recent public posts from selected Warhammer 40k faction subreddits.
2. Collect a limited sample of public users who participate in those communities.
3. Read recent public Reddit activity for those sampled users.
4. Count which other public subreddits those users also appear in.
5. Compute aggregate metrics such as:
   - `P(other subreddit | faction subreddit)`
   - baseline subreddit probability across the full sample
   - lift / affinity score
   - faction-to-faction similarity
6. Export anonymized aggregate CSV files.
7. Generate network graphs using Python.

## What the project does not do

This project does **not**:

- post to Reddit
- comment on Reddit
- vote on Reddit
- send messages to Reddit users
- moderate subreddits
- collect private messages
- collect deleted or private content
- sell or redistribute Reddit data
- create individual user profiles for publication

The project is designed to produce community-level aggregate statistics only.

## Privacy and data handling

This project is intended to minimize user-level data retention.

Usernames may be temporarily processed locally only to compute aggregate overlap statistics. The intended outputs are anonymized CSV files and graphs at the subreddit/community level.

The project should avoid publishing raw usernames, raw comments, raw submissions, or individual user histories.

## Example output

Example aggregate result:

| Faction | Other subreddit | Probability | Baseline probability | Lift |
|---|---:|---:|---:|---:|
| Tyranids | minipainting | 0.34 | 0.18 | 1.89 |
| Orks | kitbash | 0.29 | 0.10 | 2.90 |
| Necrons | PrintedWarhammer | 0.21 | 0.11 | 1.91 |

These numbers are illustrative examples, not real results.

## Planned visualizations

The project will generate two main graph types:

### 1. Faction-to-subreddit network

A bipartite graph linking each Warhammer 40k faction community to other public subreddits that appear unusually often among its sampled users.

### 2. Faction similarity network

A graph where each node is a Warhammer 40k faction and edges represent similarity between communities based on subreddit overlap profiles.

## Technology stack

Planned tools:

- Python
- PRAW
- pandas
- NumPy
- NetworkX
- matplotlib
- scikit-learn
- python-dotenv

## Quick start

Clone the repository:

```bash
git clone https://github.com/adrigonza/wh40k-faction-overlap.git
cd wh40k-faction-overlap
```

Create a virtual environment:

```bash
/usr/bin/python3 -m venv wh40k-env
source wh40k-env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` and add your Reddit API credentials:

```text
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=linux:wh40k-faction-overlap:v0.1.0 by /u/adrigonza
```

Run the prototype script:

```bash
python src/collect_overlap.py
```

Outputs will be written to the `outputs/` folder.

## GitHub Pages

The project landing page is in:

```text
docs/index.html
```

After enabling GitHub Pages from the `main` branch and `/docs` folder, the site should be available at:

```text
https://adrigonza.github.io/wh40k-faction-overlap/
```

## Status

Early planning / prototype stage.

## Intended use

This project is for personal, non-commercial analysis of public Reddit community patterns related to the Warhammer 40k hobby.

It is not intended to operate as a Reddit bot, moderation tool, advertising system, or user-profiling tool.

## Disclaimer

Warhammer 40,000 is a trademark of Games Workshop. This project is unofficial and not affiliated with Games Workshop or Reddit.

## License

MIT License.
