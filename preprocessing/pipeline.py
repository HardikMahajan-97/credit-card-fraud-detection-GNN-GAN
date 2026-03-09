"""
End-to-end preprocessing pipeline orchestrator.

Chains: load → clean → engineer features → normalize → (optionally) build graphs.
Configurable via config.yaml; supports caching of intermediate results.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from preprocessing.cleaner import DataCleaner
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.graph_builder import GraphBuilder
from utils.logger import get_logger

logger = get_logger(__name__)


class PreprocessingPipeline:
    """
    Orchestrate the full preprocessing chain.

    Args:
        cfg: Configuration dictionary (typically from config.yaml).
        cache_dir: Directory to cache intermediate DataFrames.
    """

    def __init__(
        self,
        cfg: Optional[Dict] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        cfg = cfg or {}
        preproc_cfg = cfg.get("preprocessing", {})
        graph_cfg = cfg.get("graph", {})

        self.cleaner = DataCleaner(
            outlier_method=preproc_cfg.get("outlier_method", "iqr"),
            outlier_threshold=float(preproc_cfg.get("outlier_threshold", 3.0)),
        )
        self.engineer = FeatureEngineer(
            scaler_type=preproc_cfg.get("scaler", "robust"),
            time_windows=preproc_cfg.get("time_windows", [1, 6, 12, 24]),
        )
        self.graph_builder = GraphBuilder(
            window_size=int(graph_cfg.get("window_size", 1000)),
            max_neighbors=int(graph_cfg.get("max_neighbors", 50)),
            temporal_encoding_dim=int(graph_cfg.get("temporal_encoding_dim", 16)),
        )
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        build_graphs: bool = False,
        use_cache: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[List]]:
        """
        Execute the full preprocessing pipeline.

        Args:
            df: Raw transaction DataFrame.
            build_graphs: Whether to build PyG graph objects.
            use_cache: Load/save intermediate results from disk.

        Returns:
            (processed_df, graphs_or_None)
        """
        if use_cache and self.cache_dir:
            cached = self._load_cache("processed.parquet")
            if cached is not None:
                logger.info("Loaded preprocessed data from cache.")
                graphs = None
                if build_graphs:
                    graphs = self.graph_builder.build_graphs(cached)
                return cached, graphs

        logger.info("Running preprocessing pipeline …")
        df = self.cleaner.fit_transform(df)
        df = self.engineer.fit_transform(df)

        if use_cache and self.cache_dir:
            self._save_cache(df, "processed.parquet")

        graphs = None
        if build_graphs:
            graphs = self.graph_builder.build_graphs(df)

        return df, graphs

    @property
    def feature_columns(self) -> List[str]:
        """Return feature column names determined after fit."""
        return self.engineer.feature_columns

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _save_cache(self, df: pd.DataFrame, filename: str) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)  # type: ignore[arg-type]
        path = os.path.join(self.cache_dir, filename)  # type: ignore[arg-type]
        df.to_parquet(path, index=False)
        logger.info(f"Cached preprocessed data to {path}")

    def _load_cache(self, filename: str) -> Optional[pd.DataFrame]:
        path = os.path.join(self.cache_dir, filename)  # type: ignore[arg-type]
        if os.path.exists(path):
            return pd.read_parquet(path)
        return None
