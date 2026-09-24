# Replication of the 5.1 evaluation, 2026-09-24

Same code, same frozen questions (M5, M7, M9), same five systems, two fresh runs each,
judged the same way. Nothing from the 09-23 runs is reused. `rerun.py` points the main
script's output files into this folder.

Headline: the refined agent's M7 improvement did not reproduce. On 09-23 it retrieved
3 and then 1 chunks from DreamCoder's own paper; here it retrieved 0 in both runs, and
neither of its first plans wrote a DreamCoder-only query. Everything else in the 09-23
results reproduced: the agent stops after one search round (12 of 12 runs), never uses
search_in_paper, matches the same-size baseline on which papers it reaches (12 of 12),
finds JoinABLe where the baseline does not, leaves 79.53% unused in every answer that
had it (8 of 8), and passes 1 of 60 grader rows (a different row this time).
