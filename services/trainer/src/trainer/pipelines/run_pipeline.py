from __future__ import annotations

import subprocess
STEPS = [
    ["python", "-m", "trainer.pipelines.ingest_movielens", "--bootstrap-schema"],
    ["python", "-m", "trainer.pipelines.build_embeddings"],
    ["python", "-m", "trainer.pipelines.generate_candidates"],
    ["python", "-m", "trainer.pipelines.build_ranking_dataset"],
    ["python", "-m", "trainer.pipelines.train_ranker"],
    ["python", "-m", "trainer.pipelines.generate_recommendations"],
]


def main() -> None:
    for cmd in STEPS:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    print("Pipeline completed.")


if __name__ == "__main__":
    main()
