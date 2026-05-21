import argparse
import os
import random
from typing import List, Tuple


# Generate edges for an Erdos-Renyi graph instance.
def generate_er_edges(n: int, p: float, rng: random.Random) -> List[Tuple[int, int]]:
    edges = []
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            if rng.random() < p:
                edges.append((u, v))
    return edges


# Write one MIS instance to disk.
def write_instance(file_path: str, n: int, edges: List[Tuple[int, int]]) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{n} {len(edges)}\n")
        for u, v in edges:
            f.write(f"{u} {v} 1\n")


# Return the default ER graph generation settings.
def build_default_configs() -> List[Tuple[int, float, int]]:
    return [
        (500, 0.1, 10),
        (800, 0.0625, 10),
        (1000, 0.05, 10),
        (2000, 0.025, 10),
    ]


# Generate a batch of MIS benchmark instances.
def generate_batch(output_dir: str, base_seed: int) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    created_files = []

    configs = build_default_configs()
    seed_cursor = base_seed

    for n, p, count in configs:
        for i in range(1, count + 1):
            rng = random.Random(seed_cursor)
            edges = generate_er_edges(n=n, p=p, rng=rng)
            file_name = f"MIS_ER_n{n}_p{str(p).replace('.', '')}_{i}.txt"
            file_path = os.path.join(output_dir, file_name)
            write_instance(file_path=file_path, n=n, edges=edges)
            created_files.append(file_path)
            seed_cursor += 1

    return created_files


# Parse command-line arguments.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MIS instances on Erdos-Renyi random graphs.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__)),
        help="Directory to save generated instances.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260316,
        help="Base random seed.",
    )
    return parser.parse_args()


# Command-line entry point.
def main() -> None:
    args = parse_args()
    files = generate_batch(output_dir=args.output_dir, base_seed=args.seed)
    print("Generated MIS ER instances:")
    for path in files:
        print(path)

if __name__ == "__main__":
    main()
