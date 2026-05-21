#!/usr/bin/env python
"""
View evolutionary algorithm training statistics.
Usage: python inspect_evolution.py
"""

import os


# Identify iteration rows in evolution logs.
def _is_iteration_line(line):
    return line.startswith("Iteration") or line.startswith("\u8fed\u4ee3")


# Identify rows that record a new best result.
def _is_new_best_line(line):
    return "New best" in line or "\u65b0\u6700\u4f18" in line


# Extract the completed training count from a log row.
def _extract_training_count(line):
    for token in ("Train=", "\u8bad\u7ec3="):
        if token not in line:
            continue
        start = line.find(token) + len(token)
        end = line.find(",", start)
        if end == -1:
            end = len(line)
        try:
            return int(line[start:end].strip())
        except ValueError:
            return None
    return None


# Print a readable summary of evolution.txt.
def inspect_evolution_stats():
    """Display statistics from evolution.txt."""
    log_file = "evolution.txt"

    if not os.path.exists(log_file):
        print(f"File {log_file} does not exist")
        print("Hint: evolution.txt is generated automatically when run_evolutionary_search.py runs")
        return

    print("\n" + "=" * 80)
    print("Evolutionary Algorithm Training Statistics")
    print("=" * 80 + "\n")

    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
        print(content)

    lines = content.split("\n")
    iterations = [line for line in lines if _is_iteration_line(line)]

    if iterations:
        print("\n" + "=" * 80)
        print("Quick Statistics")
        print("=" * 80)
        print(f"Total iterations: {len(iterations)}")

        new_best_lines = [line for line in iterations if _is_new_best_line(line)]
        print(f"New best count: {len(new_best_lines)}")

        total = _extract_training_count(iterations[-1])
        if total is not None:
            print(f"Total completed trainings: {total}")

        if new_best_lines:
            print("\nNew best occurrences:")
            for line in new_best_lines:
                print(f"  {line.strip()}")

        print("=" * 80 + "\n")


if __name__ == "__main__":
    inspect_evolution_stats()
