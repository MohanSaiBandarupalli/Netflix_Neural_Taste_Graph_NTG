# src/ntg/models/ranker.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass(frozen=True)
class RankerConfig:
    # Inputs (leakage-safe)
    train_path: Path = Path("data/processed/splits/train.parquet")
    val_path: Path = Path("data/processed/splits/val.parquet")

    # Optional input (graph). If missing -> popularity-only fallback.
    graph_path: Path = Path("outputs/graph/item_item.parquet")

    # Outputs
    out_topk_path: Path = Path("outputs/recommendations/topk.parquet")
    out_metrics_path: Path = Path("reports/ranking_metrics.json")

    # Ranking params
    k: int = 50
    candidates_per_user: int = 300

    # Guardrails
    min_rating: float = 4.0
    max_hist_items_per_user: int = 200
    min_user_hist: int = 5

    # Score weights
    w_graph: float = 1.0
    w_pop: float = 0.15

    # DuckDB runtime
    threads: int = 4
    memory_limit: str = "6GB"
    tmp_dir: Path = Path("data/interim/duckdb_tmp")


def build_topk_and_eval(cfg: RankerConfig) -> None:
    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_topk_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Required inputs
    for p in [cfg.train_path, cfg.val_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    # Optional graph (do NOT fail tests if missing)
    graph_available = cfg.graph_path.exists()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log("Connecting DuckDB (in-memory)")
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={cfg.threads};")
    con.execute(f"PRAGMA memory_limit='{cfg.memory_limit}';")
    con.execute(f"PRAGMA temp_directory='{cfg.tmp_dir.as_posix()}';")
    con.execute("PRAGMA enable_progress_bar=true;")
    con.execute("PRAGMA preserve_insertion_order=false;")

    log("Loading TRAIN + VAL")
    con.execute(
        f"""
        CREATE OR REPLACE VIEW train AS
        SELECT
            CAST(user_id AS BIGINT) AS user_id,
            CAST(item_id AS BIGINT) AS item_id,
            CAST(rating AS DOUBLE)  AS rating
        FROM read_parquet('{cfg.train_path.as_posix()}');
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW val AS
        SELECT
            CAST(user_id AS BIGINT) AS user_id,
            CAST(item_id AS BIGINT) AS item_id
        FROM read_parquet('{cfg.val_path.as_posix()}');
        """
    )

    if graph_available:
        log("Loading graph")
        con.execute(
            f"""
            CREATE OR REPLACE VIEW graph AS
            SELECT
                CAST(src_item AS BIGINT) AS src_item,
                CAST(dst_item AS BIGINT) AS dst_item,
                CAST(cosine AS DOUBLE)   AS cosine
            FROM read_parquet('{cfg.graph_path.as_posix()}');
            """
        )
    else:
        # Deterministic fallback: empty graph
        log("Graph missing -> running popularity-only fallback (deterministic)")
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE graph (
                src_item BIGINT,
                dst_item BIGINT,
                cosine   DOUBLE
            );
            """
        )
        con.execute("CREATE OR REPLACE VIEW graph AS SELECT * FROM graph;")

    # ---------------------------------------------------------------------
    # Leakage guard: "seen" means ANY TRAIN interaction (not just positives)
    # ---------------------------------------------------------------------
    log("Building user_seen (TRAIN-only, all ratings)")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE user_seen AS
        SELECT DISTINCT user_id, item_id
        FROM train;
        """
    )

    # Popularity prior (TRAIN-only)
    log("Computing item popularity prior (TRAIN-only)")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE item_pop AS
        SELECT item_id, COUNT(*) AS pop_cnt
        FROM train
        GROUP BY item_id;
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE item_pop_norm AS
        SELECT item_id, LOG(1 + pop_cnt) AS pop_log
        FROM item_pop;
        """
    )

    # Deterministic user history (TRAIN-only, positive-only, capped)
    # IMPORTANT: NO RANDOM() -> determinism for tests
    log("Building user history (TRAIN-only, positive-only, capped, deterministic)")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE user_hist AS
        SELECT user_id, item_id
        FROM (
            SELECT
                user_id,
                item_id,
                rating,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY rating DESC, item_id ASC
                ) AS rn
            FROM train
            WHERE rating >= {cfg.min_rating}
        )
        WHERE rn <= {cfg.max_hist_items_per_user};
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE user_hist_cnt AS
        SELECT user_id, COUNT(*) AS n_hist
        FROM user_hist
        GROUP BY user_id;
        """
    )

    # Ensure we cover users even if they have 0 positives (so we can still recommend)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE all_users AS
        SELECT DISTINCT user_id FROM train
        UNION
        SELECT DISTINCT user_id FROM val;
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE eligible AS
        SELECT
            u.user_id,
            COALESCE(h.n_hist, 0) AS n_hist
        FROM all_users u
        LEFT JOIN user_hist_cnt h USING(user_id);
        """
    )

    # Popularity fallback list (make it larger so after excluding seen items we still fill Top-K)
    pop_pool_limit = int(cfg.candidates_per_user + cfg.max_hist_items_per_user + cfg.k)
    log("Preparing popularity fallback list")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE pop_fallback AS
        SELECT item_id, pop_log
        FROM item_pop_norm
        ORDER BY pop_log DESC, item_id ASC
        LIMIT {pop_pool_limit};
        """
    )

    # Candidate generation only if graph exists (or graph table might be empty)
    if graph_available:
        log("Generating candidates from graph neighbors (excluding already-seen)")
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE cand_scored AS
            WITH neigh AS (
                SELECT
                    h.user_id,
                    g.dst_item AS cand_item,
                    SUM(g.cosine) AS graph_mass
                FROM user_hist h
                JOIN graph g
                  ON h.item_id = g.src_item
                GROUP BY 1,2
            ),
            filtered AS (
                SELECT n.*
                FROM neigh n
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM user_seen s
                    WHERE s.user_id = n.user_id AND s.item_id = n.cand_item
                )
            ),
            with_pop AS (
                SELECT
                    f.user_id,
                    f.cand_item,
                    f.graph_mass,
                    COALESCE(p.pop_log, 0.0) AS pop_log,
                    ({cfg.w_graph} * f.graph_mass + {cfg.w_pop} * COALESCE(p.pop_log, 0.0)) AS score
                FROM filtered f
                LEFT JOIN item_pop_norm p
                  ON p.item_id = f.cand_item
            )
            SELECT * FROM with_pop;
            """
        )

        log("Capping candidates per user")
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE cand_pool AS
            SELECT user_id, cand_item AS item_id, score
            FROM (
                SELECT
                    user_id,
                    cand_item,
                    score,
                    ROW_NUMBER() OVER (
                        PARTITION BY user_id
                        ORDER BY score DESC, cand_item ASC
                    ) AS rn
                FROM cand_scored
            )
            WHERE rn <= {cfg.candidates_per_user};
            """
        )
    else:
        # No graph -> empty candidate pool
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE cand_pool (
                user_id BIGINT,
                item_id BIGINT,
                score   DOUBLE
            );
            """
        )

    # Build Top-K recommendations:
    # - If graph missing -> always popularity fallback
    # - Else: popularity fallback for low-history, candidate-based for others
    log("Building Top-K recommendations")
    if not graph_available:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE topk_raw AS
            SELECT
                e.user_id,
                p.item_id,
                p.pop_log AS score
            FROM eligible e
            JOIN pop_fallback p ON TRUE;
            """
        )
    else:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE topk_raw AS
            WITH pool_or_pop AS (
                -- popularity fallback for low-history users
                SELECT
                    e.user_id,
                    p.item_id AS item_id,
                    p.pop_log AS score
                FROM eligible e
                JOIN pop_fallback p ON TRUE
                WHERE e.n_hist < {cfg.min_user_hist}

                UNION ALL

                -- candidate-based scoring for sufficient-history users
                SELECT
                    e.user_id,
                    c.item_id,
                    c.score
                FROM eligible e
                JOIN cand_pool c
                  ON e.user_id = c.user_id
                WHERE e.n_hist >= {cfg.min_user_hist}
            )
            SELECT * FROM pool_or_pop;
            """
        )

    # ---------------------------------------------------------------------
    # FINAL LEAKAGE GATE (Netflix/FAANG-style): remove ALL TRAIN-seen items
    # ---------------------------------------------------------------------
    log("Applying final leakage gate (exclude ALL TRAIN-seen items)")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE topk_noseen AS
        SELECT r.*
        FROM topk_raw r
        WHERE NOT EXISTS (
            SELECT 1 FROM user_seen s
            WHERE s.user_id = r.user_id AND s.item_id = r.item_id
        );
        """
    )

    # Rank and slice to K
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE topk AS
        SELECT
            user_id,
            item_id,
            score,
            ROW_NUMBER() OVER (
                PARTITION BY user_id
                ORDER BY score DESC, item_id ASC
            ) AS rank
        FROM topk_noseen;
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE topk_k AS
        SELECT user_id, item_id, score, rank
        FROM topk
        WHERE rank <= {cfg.k};
        """
    )

    # Health / coverage metrics (observability)
    log("Computing health metrics (coverage, empty users, rec counts)")
    health_row = con.execute(
        f"""
        WITH per_user AS (
          SELECT user_id, COUNT(*) AS n_recs
          FROM topk_k
          GROUP BY user_id
        ),
        u AS (
          SELECT COUNT(DISTINCT user_id) AS n_users FROM eligible
        )
        SELECT
          (SELECT n_users FROM u) AS n_users,
          (SELECT COUNT(*) FROM per_user WHERE n_recs >= {cfg.k}) AS users_with_k,
          (SELECT COUNT(*) FROM per_user WHERE n_recs = 0) AS users_with_0,
          (SELECT AVG(n_recs) FROM per_user) AS avg_recs_per_user,
          (SELECT MIN(n_recs) FROM per_user) AS min_recs_per_user,
          (SELECT MAX(n_recs) FROM per_user) AS max_recs_per_user
        """
    ).fetchone()

    health = {
        "n_users": int(health_row[0] or 0),
        "users_with_k": int(health_row[1] or 0),
        "users_with_0": int(health_row[2] or 0),
        "avg_recs_per_user": float(health_row[3] or 0.0),
        "min_recs_per_user": int(health_row[4] or 0),
        "max_recs_per_user": int(health_row[5] or 0),
        "pct_users_with_k": float((health_row[1] or 0) / (health_row[0] or 1)),
    }

    log(f"Writing: {cfg.out_topk_path}")
    con.execute(
        f"""
        COPY (
            SELECT user_id, item_id, score, rank
            FROM topk_k
            ORDER BY user_id ASC, rank ASC
        ) TO '{cfg.out_topk_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    # Load output and enforce hard contracts (fail fast)
    topk_df = pd.read_parquet(cfg.out_topk_path)

    if topk_df.empty:
        raise AssertionError("topk.parquet is empty — pipeline produced no recommendations")

    required = {"user_id", "item_id", "score", "rank"}
    missing = required - set(topk_df.columns)
    if missing:
        raise AssertionError(f"topk.parquet missing columns: {sorted(missing)}")

    # Rank sanity
    if topk_df["rank"].min() != 1:
        raise AssertionError("Ranks should start at 1 per user")
    if (topk_df["rank"] > cfg.k).any():
        raise AssertionError("Found rank > k in output")

    # Leakage sanity: ensure none of the recommended pairs are in TRAIN
    train_df = pd.read_parquet(cfg.train_path)[["user_id", "item_id"]]
    seen = set(map(tuple, train_df.values))
    leak = sum((u, i) in seen for u, i in map(tuple, topk_df[["user_id", "item_id"]].values))
    if leak != 0:
        raise AssertionError(f"Leakage detected: {leak} recommended items appear in TRAIN")

    # Offline eval on VAL (requires your metrics.py to exist)
    log("Evaluating on VAL")
    gt_val = pd.read_parquet(cfg.val_path)[["user_id", "item_id"]].copy()

    from ntg.evaluation.metrics import RankingMetricsConfig, evaluate_topk

    metrics = evaluate_topk(
        topk=topk_df,
        ground_truth=gt_val,
        cfg=RankingMetricsConfig(k_list=(5, 10, 20, cfg.k)),
    )

    report = {
        "run_id": run_id,
        "model_version": "ranker_v1",
        "split": "val",
        "k": cfg.k,
        "candidates_per_user": cfg.candidates_per_user,
        "min_rating": cfg.min_rating,
        "max_hist_items_per_user": cfg.max_hist_items_per_user,
        "min_user_hist": cfg.min_user_hist,
        "weights": {"w_graph": cfg.w_graph, "w_pop": cfg.w_pop},
        "graph_available": graph_available,
        "duckdb": {
            "threads": cfg.threads,
            "memory_limit": cfg.memory_limit,
            "tmp_dir": str(cfg.tmp_dir),
        },
        "health": health,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "notes": (
            "Leakage-safe: TRAIN-only history/popularity. Final leakage gate excludes ALL TRAIN-seen "
            "items (not just positive history). Deterministic ordering (no RANDOM). Includes health "
            "metrics and hard contracts."
        ),
    }

    cfg.out_metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Wrote: {cfg.out_topk_path}")
    log(f"Wrote: {cfg.out_metrics_path}")


def main() -> None:
    build_topk_and_eval(RankerConfig())


if __name__ == "__main__":
    main()
