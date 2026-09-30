"""Fresh re-run of the Checkpoint 6.1 audit, 2026-09-30. Writes ONLY to this scratch folder."""
import sys
from pathlib import Path
CP61 = Path("/Users/jean-pierre.stephan/Projects/mit-rag-course/capstone/checkpoint-6-1")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(CP61))
import capstone_checkpoint_6_1_audit as a
a.m.TRACE_LOG = OUT / "rerun_traces.log"
a.m.LOG_PATH = OUT / "rerun_agent.log"
a.PROBES_FILE = OUT / "rerun_probes.json"
a.COST_FILE = OUT / "rerun_cost.csv"
step = sys.argv[1]
{"probes": a.probes, "report": a.report, "cost": a.cost}[step]()
