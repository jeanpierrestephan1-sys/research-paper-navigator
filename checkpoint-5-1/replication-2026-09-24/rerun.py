"""Independent replication of the 5.1 evaluation, 2026-09-24.

Same code, same frozen questions, same five systems, two fresh runs each, judged the same
way. Everything is written into THIS folder, so nothing from the 09-23 runs is reused.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import capstone_checkpoint_5_1_agent_rag as m

m.ANSWERS = HERE / "checkpoint_5_1_answers.json"
m.RESULTS = HERE / "checkpoint_5_1_results.csv"
m.TRACE_LOG = HERE / "checkpoint_5_1_traces.log"

if __name__ == "__main__":
    step = sys.argv[1]
    if step == "answer":
        m.generate(("baseline", "agent_v1", "baseline_matched", "agent_v2", "baseline_matched_v2"))
    elif step == "judge":
        m.judge()
    elif step == "report":
        m.report()
