"""Official arXiv API metadata lookup and bounded topic discovery."""

import hashlib
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic, sleep

import arxiv
import requests
from pydantic import ValidationError

from arxiv_agent.contracts import Candidate, Intent, PaperMetadata, SessionState, Stage
from arxiv_agent.services.base import StageFailure
from arxiv_agent.services.input_understanding import InputUnderstandingServices, normalize_arxiv_id
from arxiv_agent.settings import Settings

logger = logging.getLogger(__name__)
_VERSION = re.compile(r"^(.*)v([1-9]\d*)$")


class _TimedSession(requests.Session):
    def __init__(self, timeout: float, interval: float):
        super().__init__()
        self.timeout = timeout
        self.interval = interval
        self.last_attempt: float | None = None

    def get(self, url, **kwargs):  # noqa: ANN001
        if self.last_attempt is not None:
            sleep(max(0, self.interval - (monotonic() - self.last_attempt)))
        self.last_attempt = monotonic()
        kwargs.setdefault("timeout", self.timeout)
        return super().get(url, **kwargs)


def _base_version(value: str) -> tuple[str, int | None]:
    match = _VERSION.fullmatch(value)
    return (match.group(1), int(match.group(2))) if match else (value, None)


def _metadata(result: arxiv.Result) -> PaperMetadata:
    short_id = normalize_arxiv_id(result.get_short_id())
    base, version = _base_version(short_id)
    if version is None:
        raise ValueError("API record has no explicit paper version")
    authors = [author.name.strip() for author in result.authors if author.name.strip()]
    categories = list(dict.fromkeys([result.primary_category, *result.categories]))
    return PaperMetadata(
        arxiv_id=base,
        version=version,
        title=" ".join(result.title.split()),
        authors=authors,
        abstract=" ".join(result.summary.split()),
        categories=[category for category in categories if category],
        published=result.published.date(),
        updated=result.updated.date(),
        abstract_url=f"https://arxiv.org/abs/{short_id}",
        pdf_url=f"https://arxiv.org/pdf/{short_id}",
    )


class ArxivDiscoveryServices:
    """One arxiv.Client per service instance enforces spacing across its API calls."""

    def __init__(
        self,
        settings: Settings,
        understanding: InputUnderstandingServices,
        client: arxiv.Client | None = None,
    ):
        self.settings = settings
        self.understanding = understanding
        if client is None:
            client = arxiv.Client(
                page_size=max(settings.candidate_count, settings.candidate_oldest_count),
                delay_seconds=settings.api_interval_seconds,
                num_retries=0,  # Graph owns the strict two-retry budget.
            )
            client._session = _TimedSession(
                settings.api_timeout_seconds, settings.api_interval_seconds
            )
        self.client = client

    def run_stage(self, stage: Stage, state: SessionState) -> dict:
        if stage == Stage.UNDERSTAND:
            return self.understanding.run_stage(stage, state)
        if stage == Stage.LOOKUP:
            return self._lookup(state)
        if stage == Stage.SEARCH:
            return self._search(state)
        if stage == Stage.BROADEN:
            return self._broaden(state)
        raise StageFailure("NOT_IMPLEMENTED", f"{stage} is not a discovery stage.", "Run discover.")

    def _cache_path(self, kind: str, key: str) -> Path:
        digest = hashlib.sha256(f"{kind}:{key}".encode()).hexdigest()
        return self.settings.data_dir / "metadata" / f"{digest}.json"

    def _cached(self, kind: str, key: str) -> list[PaperMetadata] | None:
        path = self._cache_path(kind, key)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            saved = datetime.fromisoformat(record["cached_at"])
            if saved.tzinfo is None or datetime.now(UTC) - saved > timedelta(hours=24):
                return None
            papers = [PaperMetadata.model_validate(item) for item in record["papers"]]
            return papers if kind != "lookup" or len(papers) == 1 else None
        except (OSError, ValueError, KeyError, TypeError, ValidationError):
            return None

    def _save(self, kind: str, key: str, papers: list[PaperMetadata]) -> None:
        path = self._cache_path(kind, key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as file:
                json.dump(
                    {
                        "cached_at": datetime.now(UTC).isoformat(),
                        "papers": [paper.model_dump(mode="json") for paper in papers],
                    },
                    file,
                )
                temporary = Path(file.name)
            temporary.replace(path)
        except OSError as exc:
            logger.warning("Could not write metadata cache: %s", exc)

    def _fetch(self, search: arxiv.Search) -> list[arxiv.Result]:
        try:
            return list(self.client.results(search))
        except arxiv.HTTPError as exc:
            retryable = exc.status == 429 or 500 <= exc.status < 600
            raise StageFailure(
                "ARXIV_API_ERROR",
                f"arXiv API returned HTTP {exc.status}.",
                "Try again later or check the arXiv API status.",
                retryable=retryable,
            ) from exc
        except (requests.exceptions.RequestException, arxiv.UnexpectedEmptyPageError) as exc:
            raise StageFailure(
                "ARXIV_NETWORK_ERROR",
                f"arXiv API request failed: {type(exc).__name__}.",
                "Check the internet connection and retry.",
                retryable=True,
            ) from exc
        except arxiv.ArxivError as exc:
            raise StageFailure(
                "ARXIV_API_ERROR", "arXiv API response could not be read.",
                "Retry later or use a specific arXiv ID.", retryable=True,
            ) from exc

    def _lookup(self, state: SessionState) -> dict:
        if not state.intent or state.intent.kind != "lookup" or not state.intent.arxiv_id:
            raise StageFailure(
                "INVALID_INTENT", "Lookup needs an arXiv ID.", "Provide an ID or URL."
            )
        requested = state.intent.arxiv_id
        base, requested_version = _base_version(requested)
        cached = self._cached("lookup", requested)
        from_cache = cached is not None
        if cached is None:
            results = self._fetch(arxiv.Search(id_list=[requested], max_results=1))
            if not results:
                raise StageFailure(
                    "PAPER_NOT_FOUND", f"No arXiv paper was found for {requested}.",
                    "Check the ID and version on arxiv.org.",
                )
            try:
                cached = [_metadata(results[0])]
            except (ValueError, AttributeError, TypeError, ValidationError) as exc:
                raise StageFailure(
                    "BAD_METADATA", "arXiv returned incomplete paper metadata.",
                    "Retry later or choose another paper.",
                ) from exc
        paper = cached[0]
        if paper.arxiv_id.casefold() != base.casefold():
            raise StageFailure(
                "PAPER_NOT_FOUND", "arXiv returned a different paper.", "Check the ID."
            )
        if requested_version is not None and paper.version != requested_version:
            raise StageFailure(
                "VERSION_NOT_FOUND",
                f"Requested {requested}, but arXiv returned v{paper.version}.",
                "Check the requested version on arxiv.org or omit the version for the latest.",
            )
        if not from_cache:
            self._save("lookup", requested, cached)
        return {"selected_paper": paper}

    def _search(self, state: SessionState) -> dict:
        intent = state.intent
        if not intent or intent.kind != "topic" or not intent.arxiv_query:
            raise StageFailure("INVALID_INTENT", "Topic search needs a query.", "Provide a topic.")
        key = (
            f"v2|{intent.arxiv_query}|{intent.search_terms}|"
            f"{self.settings.candidate_count}|{self.settings.candidate_oldest_count}|"
            f"{self.settings.candidate_phrase_count}"
        )
        papers = self._cached("search", key)
        warnings = list(state.warnings)
        if papers is None:
            results = self._fetch(
                arxiv.Search(query=intent.arxiv_query, max_results=self.settings.candidate_count)
            )[: self.settings.candidate_count]
            if results and self.settings.candidate_oldest_count:
                try:
                    results.extend(self._fetch(arxiv.Search(
                        query=intent.arxiv_query,
                        max_results=self.settings.candidate_oldest_count,
                        sort_by=arxiv.SortCriterion.SubmittedDate,
                        sort_order=arxiv.SortOrder.Ascending,
                    ))[: self.settings.candidate_oldest_count])
                except StageFailure as exc:
                    logger.warning("Oldest-paper search failed: %s", exc)
                    warnings.append("Oldest-paper discovery was unavailable.")
            first_term = intent.search_terms[0].strip() if intent.search_terms else ""
            if (
                results and self.settings.candidate_phrase_count
                and len(first_term.split()) >= 3
            ):
                safe_phrase = re.sub(r'[^A-Za-z0-9 -]', ' ', first_term).strip()
                phrase_query = f'all:"{safe_phrase}"'
                if intent.date_from and intent.date_to:
                    phrase_query += (
                        f" AND submittedDate:[{intent.date_from:%Y%m%d}0000 "
                        f"TO {intent.date_to:%Y%m%d}2359]"
                    )
                try:
                    results.extend(self._fetch(arxiv.Search(
                        query=phrase_query,
                        max_results=self.settings.candidate_phrase_count,
                    ))[: self.settings.candidate_phrase_count])
                except StageFailure as exc:
                    logger.warning("Phrase search failed: %s", exc)
                    warnings.append("Phrase-based discovery was unavailable.")
            papers = []
            seen = set()
            for result in results:
                try:
                    paper = _metadata(result)
                except (ValueError, AttributeError, TypeError, ValidationError) as exc:
                    logger.warning("Skipping malformed arXiv result: %s", exc)
                    warnings.append("Skipped one incomplete arXiv result.")
                    continue
                if paper.arxiv_id not in seen:
                    papers.append(paper)
                    seen.add(paper.arxiv_id)
            if results and not papers:
                raise StageFailure(
                    "BAD_METADATA", "All arXiv results had incomplete metadata.",
                    "Retry later or use a specific arXiv ID.",
                )
            self._save("search", key, papers)
        return {"candidates": [Candidate(paper=paper) for paper in papers], "warnings": warnings}

    def _broaden(self, state: SessionState) -> dict:
        intent = state.intent
        if not intent or intent.kind != "topic" or not intent.search_terms:
            raise StageFailure("INVALID_INTENT", "No topic terms to broaden.", "Provide a topic.")
        # An OR query is a superset of the initial AND query. Keep the original date interval.
        terms = intent.search_terms
        if len(terms) == 1:
            terms = terms[0].split()
        clauses = [f'all:"{term}"' for term in terms]
        query = "(" + " OR ".join(clauses) + ")"
        if intent.date_from and intent.date_to:
            query += (
                f" AND submittedDate:[{intent.date_from:%Y%m%d}0000 "
                f"TO {intent.date_to:%Y%m%d}2359]"
            )
        broadened = Intent.model_validate(intent.model_dump() | {"arxiv_query": query})
        return {
            "intent": broadened,
            "warnings": [*state.warnings, "Initial search was empty; broadened topic once."],
        }
