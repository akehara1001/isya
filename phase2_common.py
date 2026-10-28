#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for Phase 2 physician-data downloads."""

from __future__ import annotations

import csv
import html
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
BASE_FILES_URL = "https://www.e-stat.go.jp/stat-search/files"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "physician-phase2-downloader/1.0"
)
REQUEST_SLEEP_SECONDS = 2.0
FULLWIDTH_TRANSLATION = str.maketrans("０１２３４５６７８９．，－", "0123456789.,-")


@dataclass(frozen=True)
class EstatYear:
    """e-Stat classification IDs for one physician-statistics year."""

    year: int
    era_label: str
    tstat: str
    tclass1: str
    tclass2: str


PHYSICIAN_YEARS: tuple[EstatYear, ...] = (
    EstatYear(2024, "令和6年", "000001135683", "000001236302", "000001236305"),
    EstatYear(2022, "令和4年", "000001135683", "000001215100", "000001218642"),
    EstatYear(2020, "令和2年", "000001135683", "000001163706", "000001163708"),
    EstatYear(2018, "平成30年", "000001135683", "000001135684", "000001135686"),
    EstatYear(2016, "平成28年", "000001030962", "000001109395", "000001109396"),
    EstatYear(2014, "平成26年", "000001030962", "000001077021", "000001077022"),
    EstatYear(2012, "平成24年", "000001030962", "000001063286", "000001063287"),
    EstatYear(2010, "平成22年", "000001030962", "000001047384", "000001047385"),
    EstatYear(2008, "平成20年", "000001030962", "000001033840", "000001033841"),
    EstatYear(2006, "平成18年", "000001030962", "000001024034", "000001030963"),
    EstatYear(2004, "平成16年", "000001030962", "000001024372", "000001030965"),
    EstatYear(2002, "平成14年", "000001030962", "000001024886", "000001030967"),
    EstatYear(2000, "平成12年", "000001030962", "000001024909", "000001030970"),
    EstatYear(1998, "平成10年", "000001030962", "000001024925", "000001030972"),
    EstatYear(1996, "平成8年", "000001030962", "000001024939", "000001030974"),
)


class RateLimitedFetcher:
    """HTTP fetcher with User-Agent, retries, and polite sleep."""

    def __init__(self, sleep_seconds: float = REQUEST_SLEEP_SECONDS) -> None:
        self.sleep_seconds = sleep_seconds
        self._last_request_at: float | None = None

    def get(self, url: str) -> tuple[bytes, str | None]:
        """Fetch bytes from a URL."""

        if self._last_request_at is not None:
            remaining = self.sleep_seconds - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        last_error: BaseException | None = None
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    data = response.read()
                    self._last_request_at = time.monotonic()
                    return data, response.headers.get("Content-Type")
            except urllib.error.HTTPError as exc:
                self._last_request_at = time.monotonic()
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 3:
                    raise RuntimeError(f"HTTP {exc.code}: {url}") from exc
            except Exception as exc:
                self._last_request_at = time.monotonic()
                last_error = exc
                if attempt == 3:
                    raise RuntimeError(f"Request failed: {url} ({exc})") from exc
            time.sleep(self.sleep_seconds)
        raise RuntimeError(f"Request failed: {url} ({last_error})")


def decode_html(data: bytes, content_type: str | None = None) -> str:
    """Decode HTML bytes."""

    encodings: list[str | None] = []
    if content_type:
        match = re.search(r"charset=([\w\-.]+)", content_type, flags=re.I)
        if match:
            encodings.append(match.group(1))
    encodings.extend(["utf-8", "cp932", "shift_jis", "euc_jp"])
    for encoding in encodings:
        if not encoding:
            continue
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def decode_csv_text(data: bytes) -> tuple[str, str]:
    """Decode e-Stat CSV bytes."""

    for encoding in ("cp932", "utf-8-sig", "utf-8", "shift_jis"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("cp932", errors="replace"), "cp932-replace"


def read_csv_rows(path: Path) -> list[list[str]]:
    """Read CSV rows from a local file with automatic Japanese encoding."""

    text, _ = decode_csv_text(path.read_bytes())
    return list(csv.reader(text.splitlines()))


def html_to_text(fragment: str) -> str:
    """Convert an HTML fragment to compact text."""

    fragment = re.sub(r"(?i)<br\s*/?>", "\n", fragment)
    fragment = re.sub(r"(?i)</(p|div|li|tr|th|td|dt|dd|h[1-6]|span)>", "\n", fragment)
    text = re.sub(r"<[^>]+>", " ", fragment)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def normalize_text(value: object) -> str:
    """Normalize full-width digits and Japanese spacing."""

    text = "" if value is None else str(value)
    return text.translate(FULLWIDTH_TRANSLATION).replace("\u3000", " ").strip()


def compact(value: object) -> str:
    """Return whitespace-free normalized text."""

    return re.sub(r"\s+", "", normalize_text(value))


def parse_code_name(value: object) -> tuple[str | None, str, str]:
    """Parse an e-Stat row/column geography label into code, name, kind."""

    text = compact(value)
    if not text:
        return None, "", "blank"
    if "全国" in text:
        return None, "全国", "national"
    if any(term in text for term in ("総数", "合計", "計")) and not re.match(r"^\d", text):
        return None, text, "total"
    match = re.match(r"^(?P<code>\d{2,5})(?P<name>.*)$", text)
    if not match:
        return None, text, "unknown"
    code = match.group("code")
    name = match.group("name")
    kind = {2: "pref", 4: "smsa", 5: "municipality"}.get(len(code), "unknown")
    return code, name, kind


def numeric_value(value: object) -> float:
    """Parse a numeric e-Stat cell, treating '-' as zero for count matrices."""

    text = normalize_text(value).replace(",", "")
    if text in {"", "-", "－", "…", "X", "x"}:
        return 0.0
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else 0.0


def build_physician_page_url(item: EstatYear, include_tclass2: bool = True) -> str:
    """Build a physician-statistics file-list page URL."""

    query = {
        "page": "1",
        "layout": "datalist",
        "toukei": "00450026",
        "tstat": item.tstat,
        "cycle": "7",
        "tclass1": item.tclass1,
        "tclass3val": "0",
        "metadata": "1",
        "data": "1",
    }
    if include_tclass2:
        query["tclass2"] = item.tclass2
    return f"{BASE_FILES_URL}?{urllib.parse.urlencode(query)}"


def extract_article_blocks(page_html: str) -> Iterable[str]:
    """Yield all result cards from an e-Stat page."""

    yield from (match.group(0) for match in re.finditer(r"(?is)<article\b.*?</article>", page_html))


def extract_csv_url(article_html: str, page_url: str) -> str | None:
    """Extract the CSV file-download URL from a result card."""

    for match in re.finditer(r"(?is)<a\b[^>]*href\s*=\s*['\"][^'\"]*file-download[^'\"]*['\"][^>]*>.*?</a>", article_html):
        anchor = match.group(0)
        text = html_to_text(anchor).lower()
        if "csv" not in text and 'data-file_type="CSV"' not in anchor:
            continue
        href_match = re.search(r"""href\s*=\s*(["'])(?P<href>.*?)(\1)""", anchor, flags=re.I | re.S)
        if href_match:
            return urllib.parse.urljoin(page_url, html.unescape(href_match.group("href")))
    return None


def find_csv_candidates(
    fetcher: RateLimitedFetcher,
    urls: Iterable[str],
    predicate: Callable[[str], bool],
) -> list[tuple[str, str]]:
    """Find CSV candidates as (download_url, title) pairs."""

    candidates: list[tuple[str, str]] = []
    for url in urls:
        logging.info("Fetching e-Stat page: %s", url)
        data, content_type = fetcher.get(url)
        page_html = decode_html(data, content_type)
        for article in extract_article_blocks(page_html):
            title = html_to_text(article)
            if not predicate(title):
                continue
            csv_url = extract_csv_url(article, url)
            if csv_url:
                candidates.append((csv_url, title))
    return candidates


def download_csv(fetcher: RateLimitedFetcher, url: str, destination: Path, overwrite: bool = False) -> bool:
    """Download a CSV file as raw bytes."""

    if destination.exists() and not overwrite:
        logging.info("Already exists, skipped: %s", destination.name)
        return True
    data, _ = fetcher.get(url)
    if not data or data[:200].lstrip().lower().startswith((b"<!doctype", b"<html")):
        logging.warning("Downloaded content does not look like CSV: %s", url)
        return False
    destination.write_bytes(data)
    logging.info("Saved %s (%s bytes)", destination.name, destination.stat().st_size)
    return True


def load_smsa_master() -> pd.DataFrame:
    """Load the final SMSA physician panel with normalized codes."""

    df = pd.read_csv(DATA_DIR / "data/superseded/physician_panel_smsa__PRE_REPAIR.csv", dtype={"smsa_code": str, "pref_code": str})
    df["smsa_code"] = df["smsa_code"].map(lambda v: str(int(float(v))).zfill(4))
    df["pref_code"] = df["pref_code"].map(lambda v: str(int(float(v))).zfill(2))
    return df


def smsa_codes() -> list[str]:
    """Return final 330 SMSA codes."""

    return load_smsa_master()["smsa_code"].tolist()


# load_pref_population() was removed on 2026-10-06 (round 40 §1). It read pref_population.csv, a
# file generated on 2026-06-13 by download_smsa_population.py as physicians divided by the
# published per-100,000 rate (a derivative, not a data source). Its only caller,
# estimate_smsa_population.py, was retired with it to ../archive/population_scripts_retired_20261006/.


def dataframe_matrix(codes: list[str]) -> pd.DataFrame:
    """Create a zero square matrix DataFrame with string codes."""

    return pd.DataFrame(0.0, index=codes, columns=codes)
