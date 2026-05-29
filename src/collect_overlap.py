"""Warhammer 40k faction overlap analysis.

Read-only prototype for estimating aggregate subreddit overlap among public Warhammer 40k
faction communities. This script is intended for personal, non-commercial analysis. It does not
post, comment, vote, message users, or moderate communities.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import praw
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Settings:
    """Runtime settings controlling Reddit sampling, graph filtering, and output size."""

    post_limit_per_faction: int = 120
    max_users_per_faction: int = 80
    comment_limit_per_user: int = 150
    submission_limit_per_user: int = 50
    sleep_every_n_users: int = 20
    sleep_seconds: int = 2
    min_users_for_edge: int = 3
    min_lift_for_edge: float = 1.5
    top_other_subs_per_faction: int = 10
    min_similarity_for_edge: float = 0.35


SETTINGS = Settings()

FACTION_SUBS = {
    "Space Marines": "spacemarines",
    "Blood Angels": "BloodAngels",
    "Dark Angels": "theunforgiven",
    "Space Wolves": "SpaceWolves",
    "Astra Militarum": "TheAstraMilitarum",
    "Adeptus Mechanicus": "AdeptusMechanicus",
    "Sisters of Battle": "sistersofbattle",
    "Custodes": "AdeptusCustodes",
    "Grey Knights": "Grey_Knights",
    "Imperial Knights": "ImperialKnights",
    "Chaos Space Marines": "Chaos40k",
    "Death Guard": "deathguard40k",
    "Thousand Sons": "ThousandSons",
    "World Eaters": "WorldEaters40k",
    "Chaos Knights": "ChaosKnights",
    "Orks": "orks",
    "T'au": "Tau40K",
    "Tyranids": "Tyranids",
    "Genestealer Cults": "genestealercult",
    "Necrons": "Necrontyr",
    "Aeldari": "Eldar",
    "Drukhari": "Drukhari",
    "Leagues of Votann": "LeaguesofVotann",
}

EXCLUDE_SUBS = {
    "AskReddit",
    "pics",
    "funny",
    "gaming",
    "videos",
    "news",
    "worldnews",
    "todayilearned",
    "aww",
    "memes",
    "politics",
    "popular",
    "all",
    "Warhammer40k",
    "Warhammer",
    "WarhammerCompetitive",
}


def get_reddit_client() -> praw.Reddit:
    """Create a PRAW client from environment variables."""
    load_dotenv(PROJECT_ROOT / ".env")

    required = [
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "REDDIT_USER_AGENT",
    ]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Missing Reddit credentials: "
            + ", ".join(missing)
            + "\nCopy .env.example to .env and fill in your credentials."
        )

    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
        ratelimit_seconds=300,
    )


def get_recent_authors_from_subreddit(
    reddit: praw.Reddit,
    subreddit_name: str,
    post_limit: int,
    max_users: int,
) -> set[str]:
    """Collect a limited sample of public post authors from a subreddit."""
    subreddit = reddit.subreddit(subreddit_name)
    authors: set[str] = set()

    listings = [
        subreddit.new(limit=max(1, post_limit // 3)),
        subreddit.hot(limit=max(1, post_limit // 3)),
        subreddit.top(time_filter="year", limit=max(1, post_limit // 3)),
    ]

    for listing in listings:
        for post in listing:
            if post.author is not None:
                authors.add(str(post.author))
            if len(authors) >= max_users:
                return authors

    return authors


def get_user_public_subreddits(
    reddit: praw.Reddit,
    username: str,
    comment_limit: int,
    submission_limit: int,
) -> set[str]:
    """Collect public subreddits where a user recently commented or posted."""
    seen_subreddits: set[str] = set()

    try:
        redditor = reddit.redditor(username)

        for comment in redditor.comments.new(limit=comment_limit):
            subreddit = str(comment.subreddit)
            if subreddit not in EXCLUDE_SUBS:
                seen_subreddits.add(subreddit)

        for submission in redditor.submissions.new(limit=submission_limit):
            subreddit = str(submission.subreddit)
            if subreddit not in EXCLUDE_SUBS:
                seen_subreddits.add(subreddit)

    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"  Skipping u/{username}: {exc}")

    return seen_subreddits


def collect_overlap_data(reddit: praw.Reddit) -> dict[str, dict[str, set[str]]]:
    """Collect faction -> user -> set(public subreddits)."""
    faction_to_user_subs: dict[str, dict[str, set[str]]] = {}

    for faction, subreddit_name in FACTION_SUBS.items():
        print(f"Collecting users from r/{subreddit_name} for {faction}...")
        users = get_recent_authors_from_subreddit(
            reddit=reddit,
            subreddit_name=subreddit_name,
            post_limit=SETTINGS.post_limit_per_faction,
            max_users=SETTINGS.max_users_per_faction,
        )
        print(f"  Found {len(users)} users. Reading public user activity...")

        faction_to_user_subs[faction] = {}
        for index, username in enumerate(sorted(users), start=1):
            faction_to_user_subs[faction][username] = get_user_public_subreddits(
                reddit=reddit,
                username=username,
                comment_limit=SETTINGS.comment_limit_per_user,
                submission_limit=SETTINGS.submission_limit_per_user,
            )

            if index % SETTINGS.sleep_every_n_users == 0:
                print(f"  Processed {index}/{len(users)} users")
                time.sleep(SETTINGS.sleep_seconds)

        print()

    return faction_to_user_subs


def build_probability_table(
    faction_to_user_subs: dict[str, dict[str, set[str]]],
) -> pd.DataFrame:
    """Compute probability, baseline probability, lift, and log2 lift."""
    all_user_subs = [subs for user_map in faction_to_user_subs.values() for subs in user_map.values()]
    total_users = len(all_user_subs)

    if total_users == 0:
        return pd.DataFrame()

    baseline_counts: Counter[str] = Counter()
    for subs in all_user_subs:
        baseline_counts.update(subs)

    baseline_prob = {sub: count / total_users for sub, count in baseline_counts.items()}

    rows = []
    for faction, user_map in faction_to_user_subs.items():
        faction_user_count = len(user_map)
        faction_sub_counts: Counter[str] = Counter()

        for subs in user_map.values():
            faction_sub_counts.update(subs)

        for other_subreddit, count in faction_sub_counts.items():
            probability = count / faction_user_count if faction_user_count else 0
            baseline_probability = baseline_prob.get(other_subreddit, 0)
            lift = probability / baseline_probability if baseline_probability else np.nan
            log2_lift = np.log2(lift) if lift and lift > 0 else np.nan

            rows.append(
                {
                    "faction": faction,
                    "other_subreddit": other_subreddit,
                    "users_in_faction_sample": faction_user_count,
                    "users_seen_in_other_sub": count,
                    "probability": probability,
                    "baseline_probability": baseline_probability,
                    "lift": lift,
                    "log2_lift": log2_lift,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["faction", "lift"],
        ascending=[True, False],
    )


def plot_faction_to_subreddit_network(df: pd.DataFrame) -> None:
    """Create a bipartite faction-to-subreddit network graph."""
    if df.empty:
        print("No data available for faction-to-subreddit graph.")
        return

    filtered = df[
        (df["users_seen_in_other_sub"] >= SETTINGS.min_users_for_edge)
        & (df["lift"] >= SETTINGS.min_lift_for_edge)
    ].copy()

    filtered = (
        filtered.sort_values(["faction", "lift"], ascending=[True, False])
        .groupby("faction")
        .head(SETTINGS.top_other_subs_per_faction)
    )

    if filtered.empty:
        print("No edges passed the filtering thresholds.")
        return

    graph = nx.Graph()
    for _, row in filtered.iterrows():
        graph.add_node(row["faction"], node_type="faction")
        graph.add_node(row["other_subreddit"], node_type="subreddit")
        graph.add_edge(
            row["faction"],
            row["other_subreddit"],
            weight=float(row["lift"]),
        )

    pos = nx.spring_layout(graph, seed=42, weight="weight", k=0.45)
    faction_nodes = [node for node, data in graph.nodes(data=True) if data["node_type"] == "faction"]
    subreddit_nodes = [node for node, data in graph.nodes(data=True) if data["node_type"] == "subreddit"]
    edge_widths = [max(0.5, min(5, graph[u][v]["weight"])) for u, v in graph.edges()]

    plt.figure(figsize=(18, 14))
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=faction_nodes,
        node_size=900,
        node_shape="s",
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=subreddit_nodes,
        node_size=350,
        node_shape="o",
    )
    nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.35)
    nx.draw_networkx_labels(graph, pos, font_size=8)
    plt.title("Warhammer 40k factions linked to unusually common subreddits")
    plt.axis("off")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "faction_to_subreddit_network.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved {output_path}")


def plot_faction_similarity_network(df: pd.DataFrame) -> None:
    """Create a faction-only similarity graph based on log2 lift profiles."""
    if df.empty:
        print("No data available for faction similarity graph.")
        return

    support = df.groupby("other_subreddit")["users_seen_in_other_sub"].sum()
    keep_cols = support[support >= SETTINGS.min_users_for_edge].index

    matrix = df.pivot_table(
        index="faction",
        columns="other_subreddit",
        values="log2_lift",
        fill_value=0,
    )
    matrix = matrix.loc[:, matrix.columns.intersection(keep_cols)]

    if matrix.empty or matrix.shape[0] < 2:
        print("Not enough data for faction similarity graph.")
        return

    similarities = cosine_similarity(matrix)
    sim_df = pd.DataFrame(similarities, index=matrix.index, columns=matrix.index)
    sim_df.to_csv(OUTPUT_DIR / "faction_similarity.csv")

    graph = nx.Graph()
    factions = list(sim_df.index)
    graph.add_nodes_from(factions)

    for index, faction_a in enumerate(factions):
        for faction_b in factions[index + 1 :]:
            similarity = float(sim_df.loc[faction_a, faction_b])
            if similarity >= SETTINGS.min_similarity_for_edge:
                graph.add_edge(faction_a, faction_b, weight=similarity)

    if graph.number_of_edges() == 0:
        print("No faction similarity edges passed the filtering threshold.")
        return

    pos = nx.spring_layout(graph, seed=42, weight="weight")
    edge_widths = [1 + 6 * graph[u][v]["weight"] for u, v in graph.edges()]
    edge_labels = {(u, v): f"{data['weight']:.2f}" for u, v, data in graph.edges(data=True)}

    plt.figure(figsize=(14, 12))
    nx.draw_networkx_nodes(graph, pos, node_size=1000)
    nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.4)
    nx.draw_networkx_labels(graph, pos, font_size=9)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)
    plt.title("Similarity between Warhammer 40k factions based on subreddit overlap")
    plt.axis("off")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "faction_similarity_network.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved {output_path}")


def main() -> None:
    """Run the complete collection, export, and plotting pipeline."""
    reddit = get_reddit_client()
    faction_to_user_subs = collect_overlap_data(reddit)
    df = build_probability_table(faction_to_user_subs)

    output_csv = OUTPUT_DIR / "faction_subreddit_overlap.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")

    if not df.empty:
        print("\nTop aggregate overlaps by lift:")
        print(
            df[df["users_seen_in_other_sub"] >= SETTINGS.min_users_for_edge]
            .sort_values("lift", ascending=False)
            .head(25)[
                [
                    "faction",
                    "other_subreddit",
                    "users_seen_in_other_sub",
                    "probability",
                    "lift",
                ]
            ]
            .to_string(index=False)
        )

    plot_faction_to_subreddit_network(df)
    plot_faction_similarity_network(df)


if __name__ == "__main__":
    main()
