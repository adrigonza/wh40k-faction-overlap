import os
import time
from collections import defaultdict, Counter

import praw
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# 1. Configuration
# -----------------------------

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

# Sampling knobs.
POST_LIMIT_PER_FACTION = 300
COMMENT_LIMIT_PER_USER = 300
MAX_USERS_PER_FACTION = 250

# Filter out giant/default-ish subs and obvious noise.
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

MIN_USERS_FOR_EDGE = 5
MIN_LIFT_FOR_EDGE = 1.5
TOP_OTHER_SUBS_PER_FACTION = 12


# -----------------------------
# 2. Reddit client
# -----------------------------

reddit = praw.Reddit(
    client_id=os.environ["REDDIT_CLIENT_ID"],
    client_secret=os.environ["REDDIT_CLIENT_SECRET"],
    user_agent=os.environ["REDDIT_USER_AGENT"],
)


# -----------------------------
# 3. Data collection
# -----------------------------


def get_recent_authors_from_subreddit(subreddit_name, post_limit=300, max_users=250):
    """
    Collect recent unique authors from a faction subreddit.
    Uses hot/new/top-style samples rather than trying to exhaust Reddit.
    """
    sub = reddit.subreddit(subreddit_name)
    authors = set()

    listings = [
        sub.new(limit=post_limit // 3),
        sub.hot(limit=post_limit // 3),
        sub.top(time_filter="year", limit=post_limit // 3),
    ]

    for listing in listings:
        for post in listing:
            if post.author is not None:
                authors.add(str(post.author))
            if len(authors) >= max_users:
                return authors

    return authors


def get_user_subreddits(username, comment_limit=300):
    """
    Collect subreddits where a user recently commented.
    Returns a set, not counts, so each user contributes at most once per subreddit.
    """
    subs = set()

    try:
        redditor = reddit.redditor(username)

        for comment in redditor.comments.new(limit=comment_limit):
            subreddit = str(comment.subreddit)
            if subreddit not in EXCLUDE_SUBS:
                subs.add(subreddit)

        # Optional: include submitted posts too.
        for submission in redditor.submissions.new(limit=comment_limit // 3):
            subreddit = str(submission.subreddit)
            if subreddit not in EXCLUDE_SUBS:
                subs.add(subreddit)

    except Exception as e:
        # Deleted, suspended, private-ish, rate limit, etc.
        pass

    return subs


def collect_overlap_data():
    """
    Returns:
      faction_users: faction -> set of sampled users
      faction_to_user_subs: faction -> dict user -> set(subreddits)
    """
    faction_users = {}
    faction_to_user_subs = {}

    for faction, sub in FACTION_SUBS.items():
        print(f"Collecting users from r/{sub} for {faction}...")
        users = get_recent_authors_from_subreddit(
            sub,
            post_limit=POST_LIMIT_PER_FACTION,
            max_users=MAX_USERS_PER_FACTION,
        )

        faction_users[faction] = users
        faction_to_user_subs[faction] = {}

        print(f"  Found {len(users)} users. Reading user histories...")

        for i, username in enumerate(users, start=1):
            user_subs = get_user_subreddits(username, COMMENT_LIMIT_PER_USER)
            faction_to_user_subs[faction][username] = user_subs

            if i % 25 == 0:
                print(f"  Processed {i}/{len(users)} users")
                time.sleep(2)

        print()

    return faction_users, faction_to_user_subs


# -----------------------------
# 4. Probability and lift matrix
# -----------------------------


def build_probability_tables(faction_to_user_subs):
    """
    Computes:
      P(other_sub | faction)
      baseline P(other_sub)
      lift
    """
    rows = []

    all_users = []
    all_user_subs = []

    for faction, user_map in faction_to_user_subs.items():
        for username, subs in user_map.items():
            all_users.append((faction, username))
            all_user_subs.append(subs)

    baseline_counts = Counter()
    for subs in all_user_subs:
        for sub in subs:
            baseline_counts[sub] += 1

    total_users = len(all_user_subs)

    baseline_prob = {
        sub: count / total_users
        for sub, count in baseline_counts.items()
        if total_users > 0
    }

    for faction, user_map in faction_to_user_subs.items():
        n_users = len(user_map)
        sub_counts = Counter()

        for username, subs in user_map.items():
            for sub in subs:
                sub_counts[sub] += 1

        for other_sub, count in sub_counts.items():
            p = count / n_users if n_users else 0
            base = baseline_prob.get(other_sub, 0)

            if base > 0:
                lift = p / base
                log_lift = np.log2(lift)
            else:
                lift = np.nan
                log_lift = np.nan

            rows.append(
                {
                    "faction": faction,
                    "other_subreddit": other_sub,
                    "users_in_faction_sample": n_users,
                    "users_seen_in_other_sub": count,
                    "probability": p,
                    "baseline_probability": base,
                    "lift": lift,
                    "log2_lift": log_lift,
                }
            )

    df = pd.DataFrame(rows)

    return df.sort_values(["faction", "lift"], ascending=[True, False])


# -----------------------------
# 5. Bipartite network:
#    faction -> other subreddit
# -----------------------------


def plot_faction_to_subreddit_network(df):
    """
    Draws a bipartite graph where edges are faction -> other subreddit.
    Edge weight uses lift.
    """
    filtered = df[
        (df["users_seen_in_other_sub"] >= MIN_USERS_FOR_EDGE)
        & (df["lift"] >= MIN_LIFT_FOR_EDGE)
    ].copy()

    # Keep top N other subs per faction by lift.
    filtered = (
        filtered.sort_values(["faction", "lift"], ascending=[True, False])
        .groupby("faction")
        .head(TOP_OTHER_SUBS_PER_FACTION)
    )

    G = nx.Graph()

    factions = sorted(filtered["faction"].unique())
    other_subs = sorted(filtered["other_subreddit"].unique())

    for faction in factions:
        G.add_node(faction, node_type="faction")

    for sub in other_subs:
        G.add_node(sub, node_type="subreddit")

    for _, row in filtered.iterrows():
        G.add_edge(
            row["faction"],
            row["other_subreddit"],
            weight=row["lift"],
            probability=row["probability"],
            users=row["users_seen_in_other_sub"],
        )

    plt.figure(figsize=(18, 14))

    pos = nx.spring_layout(G, k=0.45, seed=42, weight="weight")

    faction_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "faction"]
    subreddit_nodes = [
        n for n, d in G.nodes(data=True) if d["node_type"] == "subreddit"
    ]

    edge_widths = [max(0.5, min(5, G[u][v]["weight"])) for u, v in G.edges()]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=faction_nodes,
        node_size=900,
        node_shape="s",
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=subreddit_nodes,
        node_size=350,
        node_shape="o",
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        alpha=0.35,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8,
    )

    plt.title("Warhammer 40k factions linked to unusually common subreddits")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return G, filtered


# -----------------------------
# 6. Faction-to-faction similarity graph
# -----------------------------


def build_faction_similarity(df, value_col="log2_lift"):
    """
    Builds faction vectors over other subreddits, then compares factions
    using cosine similarity.
    """
    matrix = df.pivot_table(
        index="faction",
        columns="other_subreddit",
        values=value_col,
        fill_value=0,
    )

    # Remove noisy columns with tiny support.
    support = df.groupby("other_subreddit")["users_seen_in_other_sub"].sum()
    keep_cols = support[support >= MIN_USERS_FOR_EDGE].index
    matrix = matrix.loc[:, matrix.columns.intersection(keep_cols)]

    sim = cosine_similarity(matrix)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)

    return matrix, sim_df


def plot_faction_similarity_network(sim_df, min_similarity=0.35):
    """
    Draws a faction-only graph.
    This is usually the best comparison graph.
    """
    G = nx.Graph()

    factions = list(sim_df.index)
    for faction in factions:
        G.add_node(faction)

    for i, faction_a in enumerate(factions):
        for faction_b in factions[i + 1 :]:
            similarity = sim_df.loc[faction_a, faction_b]

            if similarity >= min_similarity:
                G.add_edge(
                    faction_a,
                    faction_b,
                    weight=similarity,
                )

    plt.figure(figsize=(14, 12))

    pos = nx.spring_layout(G, seed=42, weight="weight")

    edge_widths = [1 + 6 * G[u][v]["weight"] for u, v in G.edges()]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=1000,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        alpha=0.4,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=9,
    )

    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=7,
    )

    plt.title("Similarity between Warhammer 40k factions based on subreddit overlap")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return G


# -----------------------------
# 7. Run everything
# -----------------------------

if __name__ == "__main__":
    faction_users, faction_to_user_subs = collect_overlap_data()

    df = build_probability_tables(faction_to_user_subs)

    df.to_csv("warhammer40k_faction_subreddit_overlap.csv", index=False)

    print("\nTop overlaps by lift:")
    print(
        df[df["users_seen_in_other_sub"] >= MIN_USERS_FOR_EDGE]
        .sort_values("lift", ascending=False)
        .head(30)[
            [
                "faction",
                "other_subreddit",
                "users_seen_in_other_sub",
                "probability",
                "lift",
            ]
        ]
    )

    bipartite_graph, filtered_edges = plot_faction_to_subreddit_network(df)

    matrix, sim_df = build_faction_similarity(df)
    sim_df.to_csv("warhammer40k_faction_similarity.csv")

    faction_graph = plot_faction_similarity_network(sim_df, min_similarity=0.35)
