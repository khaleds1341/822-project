import itertools
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

# ==========================
# DATASET
# ==========================
transactions = [
    {"pencil", "pen", "notebook"},
    {"pencil", "eraser", "ruler"},
    {"pen", "notebook", "marker"},
    {"pencil", "pen", "eraser"},
    {"notebook", "marker", "ruler"},
    {"pencil", "pen", "notebook", "marker"},
    {"eraser", "pen", "ruler"},
    {"pencil", "notebook", "marker"},
    {"pen", "notebook", "ruler"},
    {"pencil", "pen", "eraser", "notebook"}
]

min_support = 3

# ==========================
# SUPPORT COUNT FUNCTION
# ==========================
def support_count(transactions, itemsets):
    return {
        itemset: sum(1 for t in transactions if itemset.issubset(t))
        for itemset in itemsets
    }

# ==========================
# DIC ALGORITHM (STRUCTURE TRACKED)
# ==========================
levels = []                     # each level = dict of itemsets → support
frequent_itemsets = {}

active_candidates = {frozenset([i]) for t in transactions for i in t}
k = 1

while active_candidates:
    counts = support_count(transactions, active_candidates)
    freq = {i: c for i, c in counts.items() if c >= min_support}
    if not freq:
        break

    levels.append(freq)
    frequent_itemsets.update(freq)

    next_candidates = set()
    keys = list(freq.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            union = keys[i] | keys[j]
            if len(union) == k + 1:
                if all(frozenset(s) in frequent_itemsets
                       for s in itertools.combinations(union, k)):
                    next_candidates.add(union)

    active_candidates = next_candidates
    k += 1

# ==========================
# EXCEL OUTPUT
# ==========================
df = pd.DataFrame([
    {
        "Itemset": ", ".join(sorted(itemset)),
        "Support": support,
        "MinSupport": min_support
    }
    for itemset, support in frequent_itemsets.items()
])

print("Saving Excel...")
print(os.getcwd())
df.to_excel("output.xlsx", index=False)
print("Excel saved!")

# ==========================
# DIC DATA-STRUCTURE DIAGRAM
# ==========================
fig, ax = plt.subplots(figsize=(14, 4 + len(levels) * 2))
ax.axis("off")

ax.set_xlim(0, 14)
ax.set_ylim(0, len(levels) * 2 + 2)

node_pos = {}

# Draw nodes level by level
for lvl, itemsets in enumerate(levels):
    y = len(levels) * 2 - lvl * 2
    x_positions = list(range(2, 2 + len(itemsets) * 2, 2))

    for x, (iset, sup) in zip(x_positions, itemsets.items()):
        circle = Circle((x, y), 0.35, fill=False, linewidth=1.5)
        ax.add_patch(circle)
        ax.text(
            x, y,
            f"{','.join(iset)}\n{sup}",
            ha="center", va="center", fontsize=8
        )
        node_pos[(lvl, iset)] = (x, y)

# Draw expansion links
for lvl in range(len(levels) - 1):
    for parent in levels[lvl]:
        for child in levels[lvl + 1]:
            if parent.issubset(child):
                x1, y1 = node_pos[(lvl, parent)]
                x2, y2 = node_pos[(lvl + 1, child)]
                arrow = FancyArrowPatch(
                    (x1, y1 - 0.35),
                    (x2, y2 + 0.35),
                    arrowstyle="->",
                    linewidth=1.2
                )
                ax.add_patch(arrow)

plt.title("DIC Data-Structure Diagram (Candidate Expansion with Support Counts)", fontsize=14)
plt.show()
