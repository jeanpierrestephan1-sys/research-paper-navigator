"""Second-day re-run of the Checkpoint 6.1 probes (2026-09-30). Imports the audit script from the
parent folder unchanged and redirects every output file into this folder, so the first day's
files are never touched. Same commands as the audit script: probes, cost, report."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import capstone_checkpoint_6_1_audit as a  # noqa: E402

a.m.TRACE_LOG = HERE / "rerun_traces.log"
a.m.LOG_PATH = HERE / "rerun_agent.log"
a.PROBES_FILE = HERE / "rerun_probes.json"
a.COST_FILE = HERE / "rerun_cost.csv"

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    {"probes": a.probes, "report": a.report, "cost": a.cost}.get(step, lambda: print(__doc__))()
