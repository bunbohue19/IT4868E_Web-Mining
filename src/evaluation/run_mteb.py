import os
import argparse
import sys
import numpy as np
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
CACHE_DIR = PROJECT_ROOT / ".cache"

# Ensure imports work when running as a script
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Point all caches to a workspace-local directory before importing mteb
CACHE_DIR.mkdir(parents=True, exist_ok=True)
for _env_var in (
    "HF_HOME",
    "XDG_CACHE_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "MTEB_CACHE",
    "TOKENIZERS_PARALLELISM",
):
    if _env_var == "TOKENIZERS_PARALLELISM":
        os.environ.setdefault(_env_var, "false")
    else:
        os.environ.setdefault(_env_var, str(CACHE_DIR))

from mteb import MTEB, get_tasks
from embedding.embedding_model import ONNXEmbeddings


def configure_cache() -> Path:
    """Use a workspace-local cache so MTEB/HF downloads don't hit unwritable dirs."""
    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    for env_var in (
        "HF_HOME",
        "XDG_CACHE_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "MTEB_CACHE",
    ):
        os.environ.setdefault(env_var, str(cache_dir))
    return cache_dir


class ONNXMTEBModel:
    """EncoderProtocol-compatible wrapper for ONNXEmbeddings."""

    def __init__(self, model_path: Path, model_name: str | None = None, revision: str | None = None):
        self.model_path = model_path
        self.model_name = model_name or model_path.name
        self.revision = revision
        self.embedder = ONNXEmbeddings(model_path=model_path)
        sample_vec = self.embedder.embed_query("テスト")
        self._embedding_dim = len(sample_vec)

    @staticmethod
    def _flatten_inputs(inputs) -> List[str]:
        """Flatten DataLoader/iterables to a plain list of strings."""
        if isinstance(inputs, (list, tuple)):
            items = inputs
        else:
            items = []
            for batch in inputs:
                if isinstance(batch, dict):
                    if "text" in batch:
                        items.extend(batch["text"])
                    else:
                        for v in batch.values():
                            try:
                                items.extend(v)
                            except TypeError:
                                items.append(v)
                else:
                    try:
                        items.extend(batch)
                    except TypeError:
                        items.append(batch)
        return [x if isinstance(x, str) else str(x) for x in items]

    def encode(
        self,
        inputs,
        *,
        task_metadata=None,
        hf_split=None,
        hf_subset=None,
        prompt_type=None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        **_: dict,
    ) -> np.ndarray:
        sentences = self._flatten_inputs(inputs)
        return self.embedder.encode(
            sentences,
            batch_size=batch_size,
            normalize=normalize_embeddings,
            show_progress=show_progress_bar,
        )

    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        return np.dot(embeddings1, embeddings2.T)

    def similarity_pairwise(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        return np.sum(embeddings1 * embeddings2, axis=1)

    def get_sentence_embedding_dimension(self) -> int:
        return self._embedding_dim

    @property
    def mteb_model_meta(self):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run JMTEB evaluations against the ONNX embedding model."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "model/gte-multilingual-base_onnx",
        help="Path to the exported ONNX model directory.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "JSTS",
            "JSICK",
            "JapaneseSentimentClassification",
            "LivedoorNewsClustering.v2",
        ],
        help="JMTEB task names to run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for encoding during evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results",
        help="Directory to store MTEB results.",
    )
    return parser.parse_args()


def main(
    model_path: Path,
    task_names: List[str],
    batch_size: int,
    output_dir: Path,
    cache_dir: Optional[Path] = None,
) -> None:
    cache_dir = cache_dir or configure_cache()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = ONNXMTEBModel(model_path=model_path)
    available_tasks = {task.metadata.name: task for task in get_tasks()}
    missing = [name for name in task_names if name not in available_tasks]
    if missing:
        raise ValueError(
            f"Tasks not found in MTEB registry: {', '.join(missing)}. "
            f"Available example: {', '.join(sorted(available_tasks)[:5])}"
        )

    selected_tasks = [available_tasks[name] for name in task_names]

    evaluation = MTEB(tasks=selected_tasks)
    evaluation.run(
        model=model,
        batch_size=batch_size,
        encode_kwargs={"show_progress_bar": True},
        output_folder=str(output_dir),
    )


if __name__ == "__main__":
    args = parse_args()
    cache = configure_cache()
    main(
        model_path=args.model_path,
        task_names=args.tasks,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        cache_dir=cache,
    )
