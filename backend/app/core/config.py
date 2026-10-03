from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List
from pathlib import Path


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+psycopg2://videocontrol:videocontrol@localhost:5432/videocontrol"
    MODEL_NAME: str = "google/siglip2-so400m-patch14-384"
    MEDIA_ROOT: str = "./data/media"
    FRAME_ROOT: str = "./data/frames"
    CLIP_ROOT: str = "./data/clips"
    BATCH_SIZE: int = 8
    FRAME_DIFF_THRESHOLD: float = 0.08
    MIN_FRAME_INTERVAL_SECONDS: float = 0.5
    MAX_FRAME_INTERVAL_SECONDS: float = 3.0
    MAX_UPLOAD_MB: int = 2048
    TEMPORAL_GROUP_WINDOW: float = 3.0
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"
    LOG_LEVEL: str = "INFO"
    HF_HOME: str = "./data/models"

    # ------------------------------------------------------------------
    # Relational graph layer (Phase 1) and graph-aware retrieval (Phase 2).
    #
    # Everything below is ADDITIVE. No pre-existing field was renamed,
    # retyped, or given a different default, so the 14 original settings
    # keep their exact previous meaning and defaults.
    #
    # The four *_WEIGHT values are a convex combination only if they sum to
    # 1.0. They are not enforced to sum to 1; see `graph_ranking_service`
    # which normalises by their sum, so any non-negative set is valid and
    # the resulting score stays in [0, 1].
    # ------------------------------------------------------------------

    # Ranking weights. FinalScore = a*Vector + b*Graph + c*Temporal + d*Confidence
    VECTOR_WEIGHT: float = 0.65
    GRAPH_WEIGHT: float = 0.20
    TEMPORAL_WEIGHT: float = 0.10
    CONFIDENCE_WEIGHT: float = 0.05

    # Bounded graph expansion. Both are HARD caps: traversal stops at whichever
    # is reached first. There is no unbounded BFS/DFS anywhere in the codebase.
    GRAPH_MAX_DEPTH: int = 2
    GRAPH_MAX_NODES: int = 200

    # An edge is only written when the measured cosine similarity between two
    # frame embeddings is >= this value. Default 0.80 is deliberately strict:
    # the point is high-precision SIMILAR_TO edges, not dense noise.
    GRAPH_SIMILARITY_THRESHOLD: float = 0.80

    # Temporal proximity scale (seconds) for the temporal scoring term.
    # Distinct from TEMPORAL_GROUP_WINDOW, which controls result de-duplication
    # in the existing /search path and has no scoring role.
    TEMPORAL_WINDOW_SECONDS: float = 5.0

    # Scene grouping: retained keyframes whose timestamps fall within this
    # window of each other are clustered into one SCENE node.
    GRAPH_SCENE_WINDOW_SECONDS: float = 3.0

    # Bound on the number of SIMILAR_TO edges written per frame (highest
    # measured similarity first). Prevents a dense similarity graph.
    GRAPH_SIMILAR_TOP_K: int = 5

    # Absolute safety cap on edges written for one video during one indexing
    # run. If exceeded the run logs a warning and truncates (deterministically,
    # by descending weight) rather than silently growing without bound.
    GRAPH_MAX_EDGES_PER_VIDEO: int = 20000

    # Multiplier applied to top_k when choosing how many vector candidates to
    # seed graph expansion from, mirroring the existing /search over-fetch.
    GRAPH_CANDIDATE_MULTIPLIER: int = 3

    # Internal coefficients of the GraphScore sub-terms. Exposed as settings
    # so the ablation study can vary them without code edits. They are
    # normalised by their sum at scoring time, so they need not sum to 1.
    GRAPH_DIRECT_COEFF: float = 0.40
    GRAPH_SCENE_COEFF: float = 0.25
    GRAPH_EDGE_CONF_COEFF: float = 0.20
    GRAPH_TEMPORAL_COEFF: float = 0.15

    # Master switch. When false, indexing builds frames and embeddings exactly
    # as before and writes no graph rows at all. Used by the evaluation
    # harness to produce a true vector-only condition.
    GRAPH_ENABLED: bool = True

    # Which concept/object/text extraction backend to use. Only "null" is
    # registered in this repository, because no detector, OCR engine, ASR
    # model or captioner exists here. See services/graph_extraction_service.py.
    # An unknown value is a hard error, never a silent fallback to "null".
    GRAPH_EXTRACTOR: str = "null"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def ranking_weights(self) -> dict:
        """The four ranking weights, keyed by the name used in the README formula.

        Returned as a dict so the scoring service and the evaluation harness
        read exactly one source of truth for these values.
        """
        return {
            "vector": self.VECTOR_WEIGHT,
            "graph": self.GRAPH_WEIGHT,
            "temporal": self.TEMPORAL_WEIGHT,
            "confidence": self.CONFIDENCE_WEIGHT,
        }

    class Config:
        # Look for .env in current dir and parent (project root)
        env_file = (".env", "../.env")
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
