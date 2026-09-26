"""Explicit environment checks and optional tiny integration smoke tests."""

import importlib
import importlib.metadata
import os
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from arxiv_agent.settings import Settings

PACKAGES = {
    "langgraph": "langgraph.graph",
    "ollama": "ollama",
    "sentence-transformers": "sentence_transformers",
    "transformers": "transformers",
    "chromadb": "chromadb",
    "PyMuPDF": "pymupdf",
    "arxiv": "arxiv",
    "httpx": "httpx",
    "pydantic": "pydantic",
    "pydantic-settings": "pydantic_settings",
}


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def find_model_assets(settings: Settings, *, download: bool) -> tuple[Path, Path]:
    # Keep downloaded assets inside the project and avoid an extra Xet cache in the user's home.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    cache = str(settings.model_cache_dir.resolve())
    common = {"cache_dir": cache, "local_files_only": not download}
    embedding = snapshot_download(
        settings.embedding_model,
        revision=settings.embedding_revision,
        allow_patterns=[
            "config.json",
            "config_sentence_transformers.json",
            "sentence_bert_config.json",
            "modules.json",
            "1_Pooling/config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
        ],
        **common,
    )
    tokenizer = snapshot_download(
        settings.tokenizer_model,
        revision=settings.tokenizer_revision,
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ],
        **common,
    )
    for root, files in [
        (
            embedding,
            ["model.safetensors", "modules.json", "1_Pooling/config.json", "tokenizer.json"],
        ),
        (tokenizer, ["tokenizer.json", "tokenizer_config.json"]),
    ]:
        missing = [name for name in files if not (Path(root) / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete cached snapshot {root}: missing {missing}")
    return Path(embedding), Path(tokenizer)


def ollama_check(settings: Settings) -> Check:
    import httpx

    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            response = client.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
            models = response.json()["models"]
        if not isinstance(models, list) or any(not isinstance(model, dict) for model in models):
            raise ValueError("Unexpected Ollama model list format")
        found = next((m for m in models if m.get("name") == settings.generation_model), None)
        if not found:
            return Check(
                "ollama", False, f"Model missing: run ollama pull {settings.generation_model}"
            )
        return Check("ollama", True, f"{found['name']}; digest={found.get('digest', 'unknown')}")
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        return Check(
            "ollama",
            False,
            f"Cannot inspect local Ollama ({exc}). Start the Ollama app or run ollama serve; "
            "check localhost permissions if it is already running.",
        )


def run_doctor(
    settings: Settings, *, download_models: bool = False, smoke: bool = False
) -> Iterator[Check]:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    yield Check(
        "python",
        sys.version_info[:2] == (3, 11),
        f"{sys.version.split()[0]} at {sys.executable}; Python 3.11 is required",
    )
    imports_ok = True
    for package, module in PACKAGES.items():
        try:
            importlib.import_module(module)
            yield Check(package, True, importlib.metadata.version(package))
        except Exception as exc:
            imports_ok = False
            yield Check(package, False, f"{exc}; reinstall from requirements.lock inside .venv")
    try:
        settings.prepare_directories()
        yield Check("runtime directories", True, str(settings.data_dir.resolve()))
    except OSError as exc:
        yield Check("runtime directories", False, f"{exc}; choose writable data/output/cache paths")
        return
    if not imports_ok:
        yield Check(
            "integration", False, "Fix dependency import failures before running model checks"
        )
        return
    server = ollama_check(settings)
    yield server
    assets = None
    try:
        assets = find_model_assets(settings, download=download_models)
        yield Check(
            "model assets",
            True,
            f"MiniLM revision={assets[0].name}; Qwen tokenizer={assets[1].name}",
        )
    except Exception as exc:
        yield Check(
            "model assets",
            False,
            f"{exc}; run python -m arxiv_agent doctor --download-models with internet access",
        )
    if not smoke:
        return
    if server.ok:
        try:
            import httpx

            with httpx.Client(timeout=settings.model_timeout_seconds, trust_env=False) as client:
                response = client.post(
                    f"{settings.ollama_base_url}/api/chat",
                    json={
                        "model": settings.generation_model,
                        "stream": False,
                        "messages": [{"role": "user", "content": 'Return exactly {"ok":true}.'}],
                        "format": {
                            "type": "object",
                            "properties": {"ok": {"type": "boolean"}},
                            "required": ["ok"],
                            "additionalProperties": False,
                        },
                        "options": {
                            "temperature": settings.temperature,
                            "num_ctx": settings.context_tokens,
                            "num_predict": 32,
                        },
                        "keep_alive": 0,
                    },
                )
                response.raise_for_status()
                body = response.json()
            import json

            content = json.loads(body["message"]["content"])
            ok = content == {"ok": True} and body.get("done") is True
            yield Check(
                "Qwen generation",
                ok,
                f"Structured response={content}; context={settings.context_tokens}",
            )
        except Exception as exc:
            yield Check("Qwen generation", False, f"{exc}; inspect Ollama and available memory")
    else:
        yield Check("Qwen generation", False, "Skipped: local Ollama/model is unavailable")
    if not assets:
        yield Check("embedding/vector smoke", False, "Skipped: model assets are missing")
        return
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer

        torch.set_num_threads(2)
        encoder = SentenceTransformer(str(assets[0]), device="cpu", local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(str(assets[1]), local_files_only=True)
        text = "A synthetic check of local semantic retrieval."
        vectors = encoder.encode([text], normalize_embeddings=True, show_progress_bar=False)
        if vectors.shape != (1, 384):
            raise ValueError(f"Unexpected vector shape {vectors.shape}")
        if not tokenizer.encode(text):
            raise ValueError("Tokenizer produced no tokens")
        yield Check(
            "embedding/tokenizer",
            True,
            f"CPU embedding shape={vectors.shape}; Qwen tokens={len(tokenizer.encode(text))}",
        )
        # Use a subprocess for actual close/reopen verification; no private Chroma shutdown APIs.
        import json
        import subprocess

        import chromadb
        from chromadb.config import Settings as ChromaSettings

        with tempfile.TemporaryDirectory(prefix="arxiv-doctor-", dir=settings.data_dir) as temp:
            payload = Path(temp) / "embedding.json"
            payload.write_text(json.dumps(vectors.tolist()), encoding="utf-8")
            write_code = (
                "import json,sys,chromadb; from chromadb.config import Settings; "
                "c=chromadb.PersistentClient(path=sys.argv[1], "
                "settings=Settings(anonymized_telemetry=False)); "
                "c.get_or_create_collection('doctor_smoke').add(ids=['smoke-1'], "
                "embeddings=json.load(open(sys.argv[2])), "
                "documents=['synthetic environment check'])"
            )
            result = subprocess.run(
                [sys.executable, "-c", write_code, str(Path(temp) / "vectors"), str(payload)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if result.returncode:
                raise RuntimeError(f"Chroma writer failed: {result.stderr[-2000:]}")
            client = chromadb.PersistentClient(
                path=str(Path(temp) / "vectors"),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            collection = client.get_collection("doctor_smoke")
            result = collection.query(query_embeddings=vectors.tolist(), n_results=1)
            if result["ids"] != [["smoke-1"]]:
                raise ValueError("Persisted vector was not retrieved")
            yield Check(
                "Chroma persistence", True, "Stored in one process; reopened and queried in another"
            )
    except Exception as exc:
        yield Check(
            "embedding/vector smoke",
            False,
            f"{exc}; inspect model cache and dependency compatibility",
        )
