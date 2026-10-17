"""Run with python -m tests.benchmarks.agent_overhead."""
import json
import statistics
import subprocess
import sys


def cold_import():
    code = """
import json, sys, time, tracemalloc
start = time.perf_counter()
from loomable import Agent
seconds = time.perf_counter() - start
print(json.dumps([seconds * 1000, sum(n.startswith('loomable') for n in sys.modules)]))
"""
    samples = [json.loads(subprocess.check_output([sys.executable, "-c", code])) for _ in range(5)]
    code = """
import tracemalloc
tracemalloc.start()
from loomable import Agent
print(tracemalloc.get_traced_memory()[1])
"""
    peak = int(subprocess.check_output([sys.executable, "-c", code]))
    return round(statistics.median(s[0] for s in samples), 2), samples[0][1], round(peak / 1024, 2)


if __name__ == "__main__":
    elapsed, modules, peak = cold_import()
    print(json.dumps({"cold_import_ms": elapsed, "loomable_modules": modules,
                      "import_python_peak_kib": peak}))
