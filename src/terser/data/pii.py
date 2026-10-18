"""PII detection and scrubbing for prompt text, run before anything is written to ``data/interim/``.

Plan row **2.1.3**. The policy (what counts as PII in Terser, the per-entity actions, the
allowlists and why they exist, and the manual review procedure) lives in ``docs/pii_policy.md``;
this module is its implementation.

Two detection layers:

``regex``
    Dependency-free and always available. Covers e-mail addresses, phone numbers (NANP and
    ``+``-prefixed international), payment cards (Luhn-checked), IBANs (mod-97 checked), IPv4 and
    IPv6 addresses, credentials embedded in URLs, API keys and secrets, US SSNs and a conservative
    street-address pattern. Every candidate is passed through an allowlist first (RFC 2606
    example domains, private/documentation IP ranges, public test card numbers, placeholder
    values such as ``YOUR_API_KEY``, ...).

``presidio``
    The regex layer **plus** Microsoft Presidio's spaCy-based NER (``en_core_web_sm``) for
    ``PERSON`` and ``LOCATION``. A person's name is only treated as PII when the surrounding text
    suggests a private individual (see :func:`private_person_context`); names in factual
    questions ("who wrote Frankenstein?") are recorded as suppressed, never scrubbed.

Each :class:`Detection` records the backend that produced it. :func:`scrub` applies a
per-entity :class:`Action` policy and returns a :class:`ScrubResult`.

Batch usage::

    python -m terser.data.pii --in data/pilot/prompts.jsonl --text-field prompt \\
        --out /tmp/scrubbed.jsonl --report /tmp/p2_pii_pilot.md --seed 0

The direct CLI uses strict typed redaction on every string value, without demonstration-value
exemptions. The legacy ``scan`` subcommand retains configurable audit policies. NER is opt-in.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import ipaddress
import json
import math
import random
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_POLICY",
    "Action",
    "Detection",
    "Detector",
    "EntityType",
    "RowStatus",
    "ScrubResult",
    "detect_regex",
    "detect_strict",
    "scrub_text",
    "scrub_record",
    "luhn_valid",
    "iban_valid",
    "main",
    "make_detector",
    "presidio_available",
    "private_person_context",
    "render_report",
    "sample_indices",
    "scan_records",
    "scrub",
    "shannon_entropy",
    "write_sample_csv",
]


class EntityType(StrEnum):
    """The PII entity types Terser detects. The redaction placeholder is ``<VALUE>``."""

    EMAIL = "EMAIL"
    PHONE = "PHONE"
    CREDIT_CARD = "CREDIT_CARD"
    IBAN = "IBAN"
    IP_ADDRESS = "IP_ADDRESS"
    URL_CREDENTIAL = "URL_CREDENTIAL"
    API_KEY = "API_KEY"
    SECRET = "SECRET"
    HIGH_ENTROPY_TOKEN = "HIGH_ENTROPY_TOKEN"
    US_SSN = "US_SSN"
    STREET_ADDRESS = "STREET_ADDRESS"
    PERSON = "PERSON"
    LOCATION = "LOCATION"

    @property
    def placeholder(self) -> str:
        """The typed placeholder substituted by the ``redact`` action, e.g. ``<EMAIL>``."""
        return f"<{self.value}>"


class Action(StrEnum):
    """What to do with a detection.

    ``redact`` replaces the span with a typed placeholder; ``drop_row`` removes the whole record
    from the output; ``flag_for_review`` leaves the text unchanged but marks the record as held
    for a human; ``ignore`` records the detection in the report and does nothing else.
    """

    REDACT = "redact"
    DROP_ROW = "drop_row"
    FLAG_FOR_REVIEW = "flag_for_review"
    IGNORE = "ignore"


class RowStatus(StrEnum):
    """Outcome for one record, in increasing order of severity."""

    CLEAN = "clean"
    REDACTED = "redacted"
    FLAGGED = "flagged"
    DROPPED = "dropped"


DEFAULT_POLICY: Mapping[EntityType, Action] = {
    EntityType.EMAIL: Action.REDACT,
    EntityType.PHONE: Action.REDACT,
    EntityType.CREDIT_CARD: Action.DROP_ROW,
    EntityType.IBAN: Action.REDACT,
    EntityType.IP_ADDRESS: Action.REDACT,
    EntityType.URL_CREDENTIAL: Action.REDACT,
    EntityType.API_KEY: Action.REDACT,
    EntityType.SECRET: Action.REDACT,
    EntityType.HIGH_ENTROPY_TOKEN: Action.FLAG_FOR_REVIEW,
    EntityType.US_SSN: Action.DROP_ROW,
    EntityType.STREET_ADDRESS: Action.FLAG_FOR_REVIEW,
    EntityType.PERSON: Action.FLAG_FOR_REVIEW,
    EntityType.LOCATION: Action.IGNORE,
}
"""Default per-entity actions; see the policy table in ``docs/pii_policy.md``."""


@dataclass(frozen=True)
class Detection:
    """One detected (or allowlisted) PII span.

    Attributes:
        start: Start character offset in the scanned text.
        end: End offset (exclusive).
        entity_type: The entity type.
        score: Detector confidence in ``[0, 1]``.
        backend: ``"regex"`` or ``"presidio"``.
        text: The matched text. Never written to scrubbed outputs or reports.
        suppressed: Allowlist/heuristic reason when the span is *not* treated as PII, else None.
    """

    start: int
    end: int
    entity_type: EntityType
    score: float
    backend: str
    text: str
    suppressed: str | None = None

    def to_dict(self, action: Action | None = None) -> dict[str, Any]:
        """JSON-safe summary without the matched text."""
        out: dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "type": self.entity_type.value,
            "score": round(self.score, 3),
            "backend": self.backend,
        }
        if action is not None:
            out["action"] = action.value
        if self.suppressed is not None:
            out["suppressed"] = self.suppressed
        return out


@dataclass(frozen=True)
class ScrubResult:
    """Result of scrubbing one text.

    Attributes:
        original: The input text.
        scrubbed: Text with ``redact`` spans replaced, or None when the row is dropped.
        status: The row outcome.
        backend: The backend that ran (``"regex"`` or ``"presidio"``).
        detections: Active detections (overlaps resolved), each paired with its action.
        suppressed: Allowlisted candidates, kept for reporting and allowlist tuning.
    """

    original: str
    scrubbed: str | None
    status: RowStatus
    backend: str
    detections: tuple[tuple[Detection, Action], ...]
    suppressed: tuple[Detection, ...] = field(default=())

    @property
    def counts(self) -> dict[str, int]:
        """Counts of non-overlapping active entities, without matched values."""
        return dict(Counter(d.entity_type.value for d, _ in self.detections))

    def detections_json(self) -> list[dict[str, Any]]:
        """Active detections as JSON-safe dicts (no matched text)."""
        return [d.to_dict(a) for d, a in self.detections]


# --------------------------------------------------------------------------------------------
# Validators and helpers
# --------------------------------------------------------------------------------------------


def luhn_valid(digits: str) -> bool:
    """Return True if ``digits`` (digits only) passes the Luhn checksum."""
    if not digits.isdigit() or len(digits) < 2:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# ISO 13616 lengths for the most common IBAN countries; others fall back to 15..34.
_IBAN_LENGTHS: Mapping[str, int] = {
    "AD": 24, "AT": 20, "BE": 16, "BG": 22, "CH": 21, "CY": 28, "CZ": 24, "DE": 22, "DK": 18,
    "EE": 20, "ES": 24, "FI": 18, "FR": 27, "GB": 22, "GR": 27, "HR": 21, "HU": 28, "IE": 22,
    "IS": 26, "IT": 27, "LI": 21, "LT": 20, "LU": 20, "LV": 21, "MC": 27, "MT": 31, "NL": 18,
    "NO": 15, "PL": 28, "PT": 25, "RO": 24, "SA": 24, "SE": 24, "SI": 19, "SK": 24, "TR": 26,
}  # fmt: skip


def iban_valid(candidate: str) -> bool:
    """Return True if ``candidate`` (spaces allowed) is a structurally valid IBAN (mod-97 == 1)."""
    s = candidate.replace(" ", "").upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    expected = _IBAN_LENGTHS.get(s[:2])
    if expected is not None and len(s) != expected:
        return False
    rearranged = s[4:] + s[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


def shannon_entropy(s: str) -> float:
    """Shannon entropy of ``s`` in bits per character."""
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


# --------------------------------------------------------------------------------------------
# Allowlists
# --------------------------------------------------------------------------------------------

# RFC 2606 / RFC 6761 reserved names: example.{com,net,org} and the .example/.test/.invalid/
# .localhost TLDs can never belong to a real mailbox.
_RESERVED_EMAIL_DOMAIN = re.compile(
    r"(?:^|\.)(?:example\.(?:com|net|org|edu)|[^.]+\.example|[^.]+\.test|[^.]+\.invalid|"
    r"[^.]+\.localhost|localhost|example|test|invalid)$",
    re.IGNORECASE,
)
# Placeholder local parts / domains that are obviously not a real address.
_PLACEHOLDER_EMAIL = re.compile(
    r"^(?:(?:your|my|some|user|name|first|last|john|jane)[._-]?(?:name|email|user|doe|smith)?|"
    r"foo|bar|baz|test|someone|somebody|nobody|noreply|no-reply|x+)@"
    r"(?:(?:your|my|some|company|domain|email|mail)[.-]?(?:domain|company|site|provider)?"
    r"|foo|bar|baz|x+)\.(?:com|org|net|io)$",
    re.IGNORECASE,
)

# Publicly documented test card numbers (Stripe, Braintree, PayPal sandbox docs).
_TEST_CARDS = frozenset(
    {
        "4111111111111111", "4242424242424242", "4012888888881881", "4000056655665556",
        "5555555555554444", "5105105105105100", "2223003122003222", "378282246310005",
        "371449635398431", "6011111111111117", "6011000990139424", "3056930009020004",
        "30569309025904", "38520000023237", "3566002020360505", "4222222222222",
    }
)  # fmt: skip

# IBANs printed in standards documents and bank tutorials.
_EXAMPLE_IBANS = frozenset(
    {"GB82WEST12345698765432", "DE89370400440532013000", "FR1420041010050500013M02606",
     "GB29NWBK60161331926819", "NL91ABNA0417164300"}
)  # fmt: skip

# Famous / fictional / placeholder SSNs (123-45-6789 is never issued; 078-05-1120 is the
# Woolworth wallet-card number the SSA voided in 1938).
_EXAMPLE_SSNS = frozenset({"123-45-6789", "078-05-1120", "111-11-1111", "123 45 6789"})

# Addresses that name a public landmark or a well-known fictional residence.
_FAMOUS_ADDRESSES = re.compile(
    r"^(?:221b\s+baker|742\s+evergreen|1600\s+pennsylvania|10\s+downing|4\s+privet|"
    r"124\s+conch|42\s+wallaby|12\s+grimmauld|31\s+spooner|1640\s+riverside|350\s+fifth|1\s+infinite|"
    r"1\s+apple\s+park|1600\s+amphitheatre|123\s+(?:main|fake|elm|any)|"
    r"1234\s+(?:main|elm))\b",
    re.IGNORECASE,
)

# Whole placeholder tokens only: a word must stand alone ("test-key", not "Zq8TestKx92"), a
# template must be the entire value ("<token>", "${API_KEY}"), so a real secret that merely
# contains "$", "test" or "myapi" is not waved through.
_PLACEHOLDER_VALUE = re.compile(
    r"^(?:<[^>]*>|\{[^}]*\}|\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*)$"
    r"|(?:^|[\W_])(?:example|sample|dummy|placeholder|changeme|change_me|redacted|replace|"
    r"insert|todo|fake|test|demo)(?:$|[\W_\d])"
    r"|^(?:your|my)[_-]?(?:api[_-]?key|secret|token|key|pass(?:word)?)(?:$|[\W_])"
    r"|example$|^example"  # documented vendor examples, e.g. AWS's "AKIA...EXAMPLE"
    r"|x{4,}|\*{3,}|\.\.\.",
    re.IGNORECASE,
)
_TRIVIAL_PASSWORDS = frozenset(
    {"password", "pass", "passwd", "pwd", "secret", "admin", "root", "user", "guest", "test",
     "1234", "12345", "123456", "postgres", "mysql", "changeme", "hunter2", "letmein", "token"}
)  # fmt: skip


def _looks_placeholder(value: str) -> bool:
    """True for values that are obviously illustrative rather than a live credential."""
    v = value.strip("\"'`")
    if v.lower() in _TRIVIAL_PASSWORDS or _PLACEHOLDER_VALUE.search(v):
        return True
    # Repetitive filler such as "aaaaaaaaaaaaaaaa" or "0000000000000000".
    return len(set(v)) <= 3


# Well-known public DNS resolvers: routable, but they identify a service, never a person.
_PUBLIC_RESOLVERS = frozenset(
    {"8.8.8.8", "8.8.4.4", "1.1.1.1", "1.0.0.1", "9.9.9.9", "149.112.112.112",
     "208.67.222.222", "208.67.220.220", "2001:4860:4860::8888", "2001:4860:4860::8844",
     "2606:4700:4700::1111", "2606:4700:4700::1001"}
)  # fmt: skip

# Git-over-SSH remotes ("git@github.com:org/repo.git") look like e-mail addresses.
_GIT_FORGES = frozenset({"github.com", "gitlab.com", "bitbucket.org", "ssh.dev.azure.com"})

# Attribute names that follow "@" in code (numpy matmul "a@b.shape", decorators) but are not TLDs.
_CODE_ATTRIBUTE_TLDS = frozenset(
    {"shape", "data", "value", "values", "item", "items", "size", "dtype", "grad", "weight",
     "cuda", "numpy", "tolist", "json", "keys", "copy", "setter", "getter", "deleter", "cache",
     "property", "abstractmethod", "staticmethod", "classmethod", "fixture", "mark", "route"}
)  # fmt: skip


def _ip_allowlist_reason(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Non-routable addresses cannot identify a person on the public internet."""
    if addr.compressed in _PUBLIC_RESOLVERS:
        return "public_dns_resolver"
    if addr.is_loopback:
        return "loopback_ip"
    if addr.is_unspecified:
        return "unspecified_ip"
    if addr.is_link_local:
        return "link_local_ip"
    if addr.is_multicast:
        return "multicast_ip"
    if addr.is_private:  # includes RFC 1918, RFC 5737/3849 documentation ranges, 0/8, 240/4
        return "private_or_documentation_ip"
    if addr.is_reserved:
        return "reserved_ip"
    return None


# --------------------------------------------------------------------------------------------
# Regex layer
# --------------------------------------------------------------------------------------------

_EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9](?:[A-Za-z0-9._%+-]{0,62}[A-Za-z0-9_%+-])?"
    r"@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24}(?![\w-])"
)
_PHONE_INTL = re.compile(
    r"(?<![\w+])\+\d{1,3}(?:[ .-]?\(\d{1,4}\)|[ .-]?\d{1,4})(?:[ .-]?\d{2,4}){1,4}(?![\w-])"
)
_PHONE_NANP = re.compile(
    r"(?<![\w+.-])(?:1[ .-])?(?:\(\d{3}\)\s?|\d{3}[ .-])\d{3}[ .-]\d{4}(?![\w.-]*\d)(?!\w)"
)
# Seven-digit local numbers ("555-0142") are only considered right after a phone cue word.
_PHONE_LOCAL = re.compile(
    r"(?:\b(?:call|phone|tel|text|cell|mobile|number|contact|reach|at|on)\b[:.]?\s+)"
    r"([2-9]\d{2}[.-]\d{4})(?![\w.-]*\d)(?!\w)",
    re.IGNORECASE,
)
_CARD = re.compile(r"(?<![\w.-])(?:\d[ -]?){12,18}\d(?![\w-])")
_IBAN = re.compile(r"(?<![\w])[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?(?!\w)")
_IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.]*\d)(?!\.\w)")
_IPV6 = re.compile(
    r"(?<![\w:.])(?:[0-9A-Fa-f]{1,4}:){1,7}(?::|(?::[0-9A-Fa-f]{1,4}){1,7}|[0-9A-Fa-f]{1,4})"
    r"(?:%\w+)?(?![\w:])"
)
_VERSION_CONTEXT = re.compile(
    r"(?:\bv|version\s*|[=~<>!]=\s*|@|\bver\.?\s*|\b(?:python|py|node|java|gcc|clang|go|ruby|"
    r"php|perl|rust|cuda|ubuntu|debian|macos|ios|android|windows|chrome|firefox|numpy|pandas|"
    r"torch|pytorch|tensorflow|django|flask|react|kernel|release)\s+)$",
    re.IGNORECASE,
)
_URL_USERINFO = re.compile(r"\b[a-z][a-z0-9+.-]{1,15}://([^\s/:@]+):([^\s/@]+)@", re.IGNORECASE)
_URL_QUERY_SECRET = re.compile(
    r"[?&](?:access_token|token|api[_-]?key|apikey|key|secret|client_secret|password|passwd|pwd|"
    r"auth|sig|signature|session|sessionid)=([^&#\s\"'<>]+)",
    re.IGNORECASE,
)
_API_KEYS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key", re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA|AGPA|AIDA|AROA)[A-Z0-9]{16}(?!\w)")),
    ("github_token", re.compile(r"(?<!\w)gh[pousr]_[A-Za-z0-9]{36,255}(?!\w)")),
    ("github_pat", re.compile(r"(?<!\w)github_pat_[A-Za-z0-9_]{22,255}(?!\w)")),
    ("slack_token", re.compile(r"(?<!\w)xox[abposr]-[A-Za-z0-9-]{10,}(?!\w)")),
    ("prefixed_secret_key", re.compile(r"(?<!\w)sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}(?!\w)")),
    ("google_api_key", re.compile(r"(?<!\w)AIza[0-9A-Za-z_-]{35}(?!\w)")),
    ("stripe_key", re.compile(r"(?<!\w)[rs]k_live_[0-9A-Za-z]{16,}(?!\w)")),
    ("jwt", re.compile(r"(?<!\w)eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")),
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?:api[_-]?key|apikey|secret(?:[_-]?key)?|access[_-]?token|auth[_-]?token|token|"
    r"password|passwd|pwd|client[_-]?secret|aws_secret_access_key)[\"']?\s*[:=]\s*[\"']?"
    r"([A-Za-z0-9_\-+/=.!@#$%^*]{8,})",  # no "&": it separates URL query parameters
    re.IGNORECASE,
)
_HIGH_ENTROPY = re.compile(r"(?<![\w/+=-])[A-Za-z0-9_\-+/]{32,}={0,2}(?![\w/+=-])")
_SSN = re.compile(r"(?<![\w-])(?!000|666|9\d\d)\d{3}([- ])(?!00)\d{2}\1(?!0000)\d{4}(?![\w-])")
_STREET_SUFFIX = (
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|"
    r"Terrace|Ter|Circle|Cir|Parkway|Pkwy|Highway|Hwy|Square|Sq|Crescent|Close|Gardens)"
)
_STREET = re.compile(
    r"(?<![\w.])\d{1,5}[A-Za-z]?\s+(?:[NSEW]\.?\s+)?(?:[A-Z][a-z]+\s+){1,3}" + _STREET_SUFFIX
    + r"\.?(?!\w)(?:,?\s+(?:Apt|Suite|Unit|#)\.?\s*\w+)?"
)  # fmt: skip


def _emit(
    out: list[Detection],
    start: int,
    end: int,
    etype: EntityType,
    score: float,
    text: str,
    suppressed: str | None = None,
) -> None:
    out.append(Detection(start, end, etype, score, "regex", text[start:end], suppressed))


def _detect_emails(text: str, out: list[Detection]) -> None:
    for m in _EMAIL.finditer(text):
        addr = m.group(0)
        local, domain = addr.rsplit("@", 1)
        reason = None
        if domain.rsplit(".", 1)[1] in _CODE_ATTRIBUTE_TLDS:
            continue  # code such as "weights@inputs.shape", not an address at all
        if re.search(r"://[^\s/@]*:$", text[max(0, m.start() - 80) : m.start()]):
            continue  # "scheme://user:password@host" is handled as URL_CREDENTIAL
        if _RESERVED_EMAIL_DOMAIN.search(domain):
            reason = "reserved_example_domain"
        elif local == "git" and domain.lower() in _GIT_FORGES:
            reason = "git_ssh_remote"
        elif _PLACEHOLDER_EMAIL.match(addr):
            reason = "placeholder_email"
        _emit(out, m.start(), m.end(), EntityType.EMAIL, 0.95, text, reason)


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _detect_phones(text: str, out: list[Detection]) -> None:
    for m in _PHONE_INTL.finditer(text):
        digits = _digits(m.group(0))
        if not 8 <= len(digits) <= 15:
            continue
        reason = None
        if re.match(r"\+1[ .-]?\(?\d{3}\)?[ .-]?555[ .-]?\d{4}$", m.group(0)):
            reason = "fictional_555_number"
        elif digits.startswith(("44207946", "447700900", "441632960")):
            reason = "ofcom_drama_number"  # UK numbers reserved for fiction
        _emit(out, m.start(), m.end(), EntityType.PHONE, 0.75, text, reason)
    for m in _PHONE_NANP.finditer(text):
        digits = _digits(m.group(0))
        if digits.startswith("1") and len(digits) == 11:
            digits = digits[1:]
        if len(digits) != 10:
            continue
        # Skip matches already covered by the international pattern.
        if any(d.entity_type is EntityType.PHONE and d.start <= m.start() < d.end for d in out):
            continue
        reason = None
        if digits[3:6] == "555" or digits[0:3] == "555":
            reason = "fictional_555_number"
        elif digits[0] in "01":
            reason = "invalid_nanp_area_code"
        _emit(out, m.start(), m.end(), EntityType.PHONE, 0.7, text, reason)
    for m in _PHONE_LOCAL.finditer(text):
        if any(d.entity_type is EntityType.PHONE and d.start <= m.start(1) < d.end for d in out):
            continue
        reason = "fictional_555_number" if m.group(1).startswith("555") else None
        _emit(out, m.start(1), m.end(1), EntityType.PHONE, 0.6, text, reason)


def _detect_cards(text: str, out: list[Detection]) -> None:
    for m in _CARD.finditer(text):
        raw = m.group(0)
        digits = _digits(raw)
        if not 13 <= len(digits) <= 19 or digits[0] not in "23456":
            continue
        # Mixed separators ("1234-5678 9012") are unusual for real cards but common in data.
        if "-" in raw and " " in raw:
            continue
        if not luhn_valid(digits):
            continue
        if any(
            d.entity_type is EntityType.IBAN and d.start < m.end() and m.start() < d.end
            for d in out
        ):
            continue  # the digit groups of an IBAN, which can pass Luhn by chance
        reason = "public_test_card" if digits in _TEST_CARDS else None
        _emit(out, m.start(), m.end(), EntityType.CREDIT_CARD, 0.9, text, reason)


def _detect_ibans(text: str, out: list[Detection]) -> None:
    for m in _IBAN.finditer(text):
        if not iban_valid(m.group(0)):
            continue
        compact = m.group(0).replace(" ", "")
        reason = "documented_example_iban" if compact in _EXAMPLE_IBANS else None
        _emit(out, m.start(), m.end(), EntityType.IBAN, 0.9, text, reason)


def _in_subscript(text: str, start: int) -> bool:
    """True when ``start`` opens a subscript or slice (``s[::-1]``, ``arr[100::5]``): code."""
    before = text[max(0, start - 2) : start]
    return before.endswith("[") and (len(before) < 2 or before[0].isalnum() or before[0] in "_)]")


def _detect_ips(text: str, out: list[Detection]) -> None:
    for m in _IPV4.finditer(text):
        try:
            addr4 = ipaddress.IPv4Address(m.group(0))
        except ValueError:
            continue
        reason = _ip_allowlist_reason(addr4)
        if reason is None and _VERSION_CONTEXT.search(text[max(0, m.start() - 12) : m.start()]):
            reason = "version_string"
        _emit(out, m.start(), m.end(), EntityType.IP_ADDRESS, 0.8, text, reason)
    for m in _IPV6.finditer(text):
        candidate = m.group(0)
        if candidate.count(":") < 2 or sum(ch.isdigit() for ch in candidate) == 0:
            continue
        # Short hex-ish tokens around "::" (C++ scopes such as "a1::b2") are not addresses.
        if not re.search(r"[0-9A-Fa-f]{3,4}", candidate) or _in_subscript(text, m.start()):
            continue
        try:
            addr6 = ipaddress.IPv6Address(candidate.split("%", 1)[0])
        except ValueError:
            continue
        reason = _ip_allowlist_reason(addr6)
        if reason is None and addr6.exploded.startswith("2001:0db8"):
            reason = "private_or_documentation_ip"
        _emit(out, m.start(), m.end(), EntityType.IP_ADDRESS, 0.8, text, reason)


def _detect_url_credentials(text: str, out: list[Detection]) -> None:
    for m in _URL_USERINFO.finditer(text):
        password = m.group(2)
        reason = "placeholder_credential" if _looks_placeholder(password) else None
        _emit(out, m.start(1), m.end(2), EntityType.URL_CREDENTIAL, 0.9, text, reason)
    for m in _URL_QUERY_SECRET.finditer(text):
        value = m.group(1)
        if len(value) < 8:
            continue
        reason = "placeholder_credential" if _looks_placeholder(value) else None
        _emit(out, m.start(1), m.end(1), EntityType.URL_CREDENTIAL, 0.8, text, reason)


def _detect_secrets(text: str, out: list[Detection]) -> None:
    reason: str | None
    for name, pattern in _API_KEYS:
        for m in pattern.finditer(text):
            reason = "placeholder_credential" if _PLACEHOLDER_VALUE.search(m.group(0)) else None
            score = 0.99 if name != "jwt" else 0.9
            _emit(out, m.start(), m.end(), EntityType.API_KEY, score, text, reason)
    for m in _SECRET_ASSIGNMENT.finditer(text):
        value = m.group(1).rstrip(".,;")
        start, end = m.start(1), m.start(1) + len(value)
        if len(value) < 8:
            continue
        quoted = text[m.start(1) - 1 : m.start(1)] in {"'", '"'}
        reason = None
        if _looks_placeholder(value):
            reason = "placeholder_credential"
        elif not quoted and (
            re.fullmatch(r"[A-Za-z_]+", value)  # api_key = get_key / token = tokenizer
            or re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", value)  # os.environ
            or text[end : end + 1] in {"(", "["}  # a call or subscript, not a literal
        ):
            reason = "identifier_not_secret"
        _emit(out, start, end, EntityType.SECRET, 0.85, text, reason)
    for m in _HIGH_ENTROPY.finditer(text):
        tok = m.group(0)
        has_classes = all(
            any(pred(c) for c in tok) for pred in (str.islower, str.isupper, str.isdigit)
        )
        if not has_classes or re.fullmatch(r"(?:0[xX])?[0-9a-fA-F]+", tok):
            continue  # hex digests (git SHAs, checksums) are not secrets
        if "/" in tok and re.fullmatch(r"[A-Za-z0-9_./-]+", tok) and tok.count("/") >= 2:
            continue  # file paths
        if shannon_entropy(tok) < 4.0:
            continue
        reason = "placeholder_credential" if _PLACEHOLDER_VALUE.search(tok) else None
        _emit(out, m.start(), m.end(), EntityType.HIGH_ENTROPY_TOKEN, 0.5, text, reason)


def _detect_ssns(text: str, out: list[Detection]) -> None:
    for m in _SSN.finditer(text):
        reason = "example_ssn" if m.group(0) in _EXAMPLE_SSNS else None
        _emit(out, m.start(), m.end(), EntityType.US_SSN, 0.85, text, reason)


def _detect_addresses(text: str, out: list[Detection]) -> None:
    for m in _STREET.finditer(text):
        reason = "famous_or_fictional_address" if _FAMOUS_ADDRESSES.match(m.group(0)) else None
        _emit(out, m.start(), m.end(), EntityType.STREET_ADDRESS, 0.6, text, reason)


_REGEX_DETECTORS: tuple[Callable[[str, list[Detection]], None], ...] = (
    _detect_emails,
    _detect_url_credentials,
    _detect_secrets,
    _detect_ibans,
    _detect_cards,
    _detect_ssns,
    _detect_phones,
    _detect_ips,
    _detect_addresses,
)


def detect_regex(text: str) -> list[Detection]:
    """Run the dependency-free regex layer over ``text``.

    Returns:
        All candidates in start order, including allowlisted ones (``suppressed`` set).
    """
    out: list[Detection] = []
    for detector in _REGEX_DETECTORS:
        detector(text, out)
    return sorted(out, key=lambda d: (d.start, -d.end))


# --------------------------------------------------------------------------------------------
# PERSON context heuristic (Presidio backend)
# --------------------------------------------------------------------------------------------

_RELATION = (
    r"(?:friend|best friend|bff|boss|manager|supervisor|coworker|co-worker|colleague|neighbou?r|"
    r"wife|husband|partner|spouse|fianc[eé]e?|boyfriend|girlfriend|ex|son|daughter|kid|child|"
    r"mom|mum|mother|dad|father|parent|brother|bro|sister|sis|sibling|cousin|aunt|uncle|niece|"
    r"nephew|grandma|grandpa|grandmother|grandfather|teacher|professor|student|tutor|doctor|"
    r"therapist|client|customer|patient|tenant|landlord|roommate|flatmate|classmate|employee|"
    r"employer|coach|lawyer|dentist|nanny|babysitter|in-law|stepmom|stepdad)"
)
_HONORIFIC = r"(?:(?:mr|mrs|ms|mx|miss|dr|prof|professor|coach)\.?\s+)?"
_PRIVATE_BEFORE = re.compile(
    r"(?:\b(?:my|our|his|her|their|your)\s+(?:[\w']+\s+){0,2}" + _RELATION
    + r"s?(?:'s)?,?\s*" + _HONORIFIC + r"$"
    r"|\b(?:my name is|i am|i'm|im|this is|call me|signed|regards|sincerely|cheers|thanks),?\s*$"
    r"|\bname\s*[:=]\s*[\"']?$"
    r"|(?:^|\n|[\"'“(])\s*(?:dear|hi|hello|hey|to|from|attn:?)\s+$"
    r"|\b(?:named|called)\s+$)",
    re.IGNORECASE,
)  # fmt: skip
_PRIVATE_AFTER = re.compile(
    r"^(?:'s)?\s*(?:\(|,)?\s*(?:my|our)\s+" + _RELATION
    + r"|^(?:'s)\s+(?:email|e-mail|phone|cell|mobile|number|address|home|birthday|date of birth|"
    r"dob|ssn|salary|diagnosis|medical|bank|account)",
    re.IGNORECASE,
)  # fmt: skip
_FICTIONAL_BEFORE_NAMED = re.compile(
    r"\b(?:character|protagonist|hero|heroine|villain|narrator|dog|cat|puppy|kitten|pet|robot|"
    r"dragon|wizard|knight|alien|unicorn|fox|bear|mouse|bunny|rabbit|princess|prince|detective|"
    r"girl|boy|man|woman|kid|pirate|ghost|ai|fairy|elf|troll|monster|story|poem|tale)\b"
    r"[^.!?\n]{0,25}\b(?:named|called)\s+$",
    re.IGNORECASE,
)


def private_person_context(
    text: str, start: int, end: int, other_pii: Sequence[Detection] = ()
) -> str | None:
    """Decide whether a PERSON span refers to a private individual.

    The heuristic looks for cues that the name belongs to someone the prompt author knows
    personally, as opposed to a public figure in a factual question or a fictional character:

    * a relationship word before the name ("my coworker Dana Reyes", "my boss, Tom") or right
      after it ("Priya, my roommate");
    * a self-introduction or sign-off ("my name is ...", "regards, ...") or a salutation at the
      start of a line ("Dear ...", "Hi ...");
    * "named"/"called" directly before the name, unless it introduces a fictional character or
      animal ("a dragon named Ember");
    * a possessive followed by personal attributes ("Sam's phone number");
    * any other active, non-name PII in the same record (a name next to an e-mail or phone
      number is contact data).

    Returns:
        A short cue label when the context looks private, else None (the span is suppressed).
    """
    before = text[max(0, start - 60) : start]
    after = text[end : end + 40]
    if any(d.entity_type not in (EntityType.PERSON, EntityType.LOCATION) for d in other_pii):
        return "co_occurs_with_pii"
    if _PRIVATE_BEFORE.search(before):
        if re.search(r"\b(?:named|called)\s+$", before, re.IGNORECASE):
            if _FICTIONAL_BEFORE_NAMED.search(before):
                return None
            return "named_cue"
        return "relationship_or_address_cue"
    if _PRIVATE_AFTER.search(after):
        return "relationship_or_attribute_cue"
    return None


# --------------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------------

Detector = Callable[[str], list[Detection]]
"""A function mapping text to candidate detections (active and suppressed)."""


def presidio_available() -> bool:
    """True if ``presidio_analyzer`` and the spaCy ``en_core_web_sm`` model are importable."""
    try:
        import importlib.util

        return (
            importlib.util.find_spec("presidio_analyzer") is not None
            and importlib.util.find_spec("en_core_web_sm") is not None
        )
    except (ImportError, ValueError):
        return False


class _PresidioDetector:
    """Regex layer plus Presidio NER for PERSON and LOCATION (loaded lazily, once)."""

    backend = "presidio"

    def __init__(self, score_threshold: float = 0.6) -> None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
        )
        self._engine = AnalyzerEngine(
            nlp_engine=provider.create_engine(), supported_languages=["en"]
        )
        self._threshold = score_threshold

    def __call__(self, text: str) -> list[Detection]:
        out = detect_regex(text)
        regex_active = [d for d in out if d.suppressed is None]
        results = self._engine.analyze(
            text=text,
            language="en",
            entities=["PERSON", "LOCATION"],
            score_threshold=self._threshold,
        )
        for r in results:
            span = text[r.start : r.end]
            etype = EntityType(r.entity_type)
            # spaCy occasionally tags placeholders we inserted or bare code identifiers.
            if re.fullmatch(r"<[A-Z_]+>|[\W\d_]+", span):
                continue
            reason: str | None = None
            if any(d.start < r.end and r.start < d.end for d in regex_active):
                reason = "overlaps_regex_detection"
            elif etype is EntityType.PERSON:
                if private_person_context(text, r.start, r.end, regex_active) is None:
                    reason = "no_private_context"
            out.append(Detection(r.start, r.end, etype, float(r.score), "presidio", span, reason))
        return sorted(out, key=lambda d: (d.start, -d.end))


def make_detector(backend: str = "regex") -> tuple[Detector, str]:
    """Build a detector.

    Args:
        backend: ``"regex"``, ``"presidio"`` (falls back if unavailable) or ``"auto"`` (Presidio
            when installed, else regex).

    Returns:
        ``(detector, backend_name)`` where ``backend_name`` is what actually runs.
    """
    if backend not in {"auto", "regex", "presidio"}:
        raise ValueError(f"unknown backend {backend!r}; expected auto, regex or presidio")
    if backend == "regex" or (backend == "auto" and not presidio_available()):
        return detect_regex, "regex"
    if not presidio_available():
        return detect_regex, "regex"
    try:
        return _PresidioDetector(), "presidio"
    except (ImportError, OSError, RuntimeError, ValueError):
        return detect_regex, "regex"


# --------------------------------------------------------------------------------------------
# Scrubbing
# --------------------------------------------------------------------------------------------

_SEVERITY = {Action.DROP_ROW: 3, Action.REDACT: 2, Action.FLAG_FOR_REVIEW: 1, Action.IGNORE: 0}


def _resolve_overlaps(
    items: Iterable[tuple[Detection, Action]],
) -> list[tuple[Detection, Action]]:
    """Keep a non-overlapping subset, preferring severe actions, then score, then length."""
    ranked = sorted(
        items, key=lambda da: (-_SEVERITY[da[1]], -da[0].score, -(da[0].end - da[0].start))
    )
    kept: list[tuple[Detection, Action]] = []
    for det, act in ranked:
        if all(det.end <= k.start or det.start >= k.end for k, _ in kept):
            kept.append((det, act))
    return sorted(kept, key=lambda da: da[0].start)


def scrub(
    text: str,
    detector: Detector = detect_regex,
    policy: Mapping[EntityType | str, Action | str] | None = None,
    backend: str = "regex",
) -> ScrubResult:
    """Detect PII in ``text`` and apply the per-entity policy.

    Args:
        text: The text to scrub.
        detector: A detector from :func:`make_detector` (default: the regex layer).
        policy: Overrides merged over :data:`DEFAULT_POLICY`; keys/values may be strings.
        backend: Backend name recorded in the result.

    Returns:
        The :class:`ScrubResult`. A row with any ``drop_row`` detection has ``scrubbed=None``.
    """
    merged: dict[EntityType, Action] = dict(DEFAULT_POLICY)
    for key, value in (policy or {}).items():
        merged[EntityType(key)] = Action(value)

    candidates = detector(text)
    suppressed = tuple(d for d in candidates if d.suppressed is not None)
    active = _resolve_overlaps(
        (d, merged[d.entity_type]) for d in candidates if d.suppressed is None
    )

    actions = {a for _, a in active}
    if Action.DROP_ROW in actions:
        return ScrubResult(text, None, RowStatus.DROPPED, backend, tuple(active), suppressed)

    pieces: list[str] = []
    cursor = 0
    for det, act in active:
        if act is Action.REDACT:
            pieces.append(text[cursor : det.start])
            pieces.append(det.entity_type.placeholder)
            cursor = det.end
    pieces.append(text[cursor:])
    scrubbed = "".join(pieces)

    if Action.FLAG_FOR_REVIEW in actions:
        status = RowStatus.FLAGGED
    elif Action.REDACT in actions:
        status = RowStatus.REDACTED
    else:
        status = RowStatus.CLEAN
    return ScrubResult(text, scrubbed, status, backend, tuple(active), suppressed)


# Strict ingestion deliberately redacts demonstration values too. The legacy policy API
# remains available for audits, but is not suitable for publishing text without review.
_NON_PII_REASONS = {
    "version_string",
    "identifier_not_secret",
    "git_ssh_remote",
    "no_private_context",
    "overlaps_regex_detection",
}
_STRICT_IPV6 = re.compile(r"(?<![\w:])[0-9A-Fa-f:.]*:[0-9A-Fa-f:.]+(?:%[\w]+)?(?![\w:])")
# Lower-case addresses ("18 larkspur lane") only with an unambiguous street suffix: with
# IGNORECASE on the full suffix list, "Way" / "Drive" / "Dr" / "St" match prose ("60 miles each
# way", "20 minutes to drive to work", "2 Cups Dr Pepper").
_STRICT_STREET = re.compile(
    r"(?<![\w.])\d{1,5}[a-z]?\s+(?:[nsew]\.?\s+)?(?:[a-z]+\s+){1,3}"
    r"(?:street|avenue|ave|road|rd|lane|ln|boulevard|blvd)\.?(?!\w)",
    re.IGNORECASE,
)
_STRICT_IBAN = re.compile(_IBAN.pattern, re.IGNORECASE)
_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s<>\"']+", re.IGNORECASE)
_URL_TOKEN = re.compile(
    r"[?&#](?:access_token|token|api[_-]?key|apikey|key|secret|client_secret|password|"
    r"passwd|pwd|auth|sig|signature|session|sessionid)=([^&#\s]+)",
    re.IGNORECASE,
)


def detect_strict(text: str, detector: Detector = detect_regex) -> list[Detection]:
    """Detect ingestion PII without demonstration-value exemptions; keep code heuristics."""
    found = [
        replace(d, suppressed=None) if d.suppressed not in _NON_PII_REASONS else d
        for d in detector(text)
        if not (
            d.entity_type is EntityType.STREET_ADDRESS
            and re.search(
                r"\b(?:across|the|over|under|along|on|from|near|of)\b", d.text, re.IGNORECASE
            )
        )
    ]
    for pattern, kind in (
        (_STRICT_IBAN, EntityType.IBAN),
        (_STRICT_STREET, EntityType.STREET_ADDRESS),
    ):
        for match in pattern.finditer(text):
            if kind is EntityType.IBAN and not iban_valid(match.group()):
                continue
            if kind is EntityType.STREET_ADDRESS and re.search(
                r"\b(?:across|the|over|under|along|on|from|near|of|by|to|via|per|each|a|an|down|"
                r"up|miles?|km|minutes?|hours?)\b",
                match.group(),
                re.IGNORECASE,
            ):
                continue
            end = match.end()
            if kind is EntityType.STREET_ADDRESS:
                end = match.start() + len(match.group().rstrip("."))
            # Prefer strict spans over legacy spans that include sentence punctuation.
            _emit(found, match.start(), end, kind, 0.99, text)
    for match in _CARD.finditer(text):
        digits = _digits(match.group())
        if 13 <= len(digits) <= 19 and digits[0] in "23456" and luhn_valid(digits):
            if not any(
                d.entity_type is EntityType.IBAN and d.start < match.end() and match.start() < d.end
                for d in found
            ):
                _emit(found, match.start(), match.end(), EntityType.CREDIT_CARD, 0.95, text)
    for match in re.finditer(
        r"\b(?:call|phone|tel|mobile|cell|contact)\b[:.]?\s+(\d{10,15})(?!\w)",
        text,
        re.IGNORECASE,
    ):
        _emit(found, match.start(1), match.end(1), EntityType.PHONE, 0.85, text)
    for match in _STRICT_IPV6.finditer(text):
        value = match.group().rstrip(".")
        try:
            ipaddress.IPv6Address(value.split("%", 1)[0])
        except ValueError:
            continue
        if _in_subscript(text, match.start()):
            continue
        # Short C++ scope-like names have no unambiguous address cue.
        unambiguous = value.startswith("::") or value.endswith("::") or value.count(":") >= 7
        if not unambiguous and not re.search(r"[a-fA-F0-9]{3,4}", value):
            continue
        _emit(found, match.start(), match.start() + len(value), EntityType.IP_ADDRESS, 0.95, text)
    for url in _URL.finditer(text):
        userinfo = re.match(r"[a-z][a-z0-9+.-]*://([^@/\s?#]+)@", url.group(), re.IGNORECASE)
        if userinfo:
            _emit(
                found,
                url.start() + userinfo.start(1),
                url.start() + userinfo.end(1),
                EntityType.URL_CREDENTIAL,
                1.0,
                text,
            )
        for match in _URL_TOKEN.finditer(url.group()):
            start = url.start() + match.start(1)
            _emit(found, start, url.start() + match.end(1), EntityType.URL_CREDENTIAL, 0.98, text)
    # Explicit arithmetic after a candidate is not a phone number.
    return [
        d
        for d in found
        if not (d.entity_type is EntityType.PHONE and re.match(r"\s*[+*/=]\s*\d", text[d.end :]))
    ]


def scrub_text(text: str, detector: Detector = detect_regex, backend: str = "regex") -> ScrubResult:
    """Deterministic typed replacement for ingestion, including cards and ambiguous secrets."""
    return scrub(
        text,
        lambda value: detect_strict(value, detector),
        {kind: Action.REDACT for kind in EntityType},
        backend,
    )


def scrub_record(
    record: Mapping[str, Any],
    detector: Detector = detect_regex,
    backend: str = "regex",
    skip: Iterable[str] = (),
) -> tuple[dict[str, Any], list[ScrubResult]]:
    """Scrub every string value recursively, leaving keys and non-text values unchanged.

    Top-level fields named in ``skip`` (ids, source and category labels) are kept verbatim.
    """
    results: list[ScrubResult] = []
    skipped = set(skip)

    def visit(value: Any) -> Any:
        if isinstance(value, str):
            result = scrub_text(value, detector, backend)
            results.append(result)
            return result.scrubbed
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    return {k: v if k in skipped else visit(v) for k, v in record.items()}, results


# --------------------------------------------------------------------------------------------
# Batch scan, report, review sample
# --------------------------------------------------------------------------------------------


def sample_indices(n: int, sample_size: int, seed: int) -> list[int]:
    """Deterministic sorted random sample of ``min(n, sample_size)`` indices from ``range(n)``."""
    k = min(n, sample_size)
    return sorted(random.Random(seed).sample(range(n), k))


def scan_records(
    records: Sequence[Mapping[str, Any]],
    field_name: str,
    detector: Detector,
    backend: str,
    policy: Mapping[EntityType | str, Action | str] | None = None,
) -> list[ScrubResult]:
    """Scrub ``record[field_name]`` for every record.

    Raises:
        KeyError: If a record lacks the field or it is not a string.
    """
    results: list[ScrubResult] = []
    for i, rec in enumerate(records):
        value = rec.get(field_name)
        if not isinstance(value, str):
            raise KeyError(f"record {i} has no string field {field_name!r}")
        results.append(scrub(value, detector, policy, backend))
    return results


def _record_id(rec: Mapping[str, Any], index: int) -> str:
    return str(rec.get("id", f"row-{index}"))


def render_report(
    results: Sequence[ScrubResult],
    *,
    source: str,
    field_name: str,
    backend: str,
    sample_path: str | None,
    sample_size: int,
    seed: int,
    input_hash: str = "unavailable",
    git_sha: str = "unavailable",
    git_dirty: bool = False,
) -> str:
    """Render the markdown scrub report (no raw PII: entities appear only as placeholders)."""
    status_counts = Counter(r.status for r in results)
    type_action: Counter[tuple[str, str]] = Counter()
    suppressed: Counter[tuple[str, str, str]] = Counter()
    for r in results:
        for d, a in r.detections:
            type_action[(d.entity_type.value, a.value)] += 1
        for d in r.suppressed:
            suppressed[(d.entity_type.value, d.backend, d.suppressed or "")] += 1

    n_ignored = sum(n for (_, act), n in type_action.items() if act == Action.IGNORE.value)
    lines = [
        "# PII scrub report",
        "",
        f"- Source: `{source}` (field `{field_name}`)",
        f"- Backend: `{backend}`",
        f"- Seed: {seed}",
        f"- Git SHA: `{git_sha}`",
        f"- Git dirty: {str(git_dirty).lower()}",
        f"- Input SHA256: `{input_hash}`",
        f"- Rows touched: {sum(r.status is not RowStatus.CLEAN for r in results)}",
        "- Human review: pending; no 500-row review has been performed by this command.",
        f"- Records scanned: {len(results)}",
        f"- Actionable detections (redact / drop_row / flag_for_review): "
        f"{sum(type_action.values()) - n_ignored}",
        f"- Recorded with action `ignore` (not PII under the policy): {n_ignored}",
        f"- Allowlisted / suppressed candidates: {sum(suppressed.values())}",
    ]
    if sample_path:
        k = min(len(results), sample_size)
        lines.append(f"- Manual review sample: `{sample_path}` ({k} rows, seed {seed})")
    lines += ["", "## Rows by outcome", "", "| Status | Rows |", "| :--- | ---: |"]
    for status in RowStatus:
        lines.append(f"| {status.value} | {status_counts.get(status, 0)} |")

    lines += ["", "## Active detections by entity type and action", ""]
    lines += ["| Entity type | Action | Count |", "| :--- | :--- | ---: |"]
    for kind in EntityType:
        actions = [
            (act, count) for (name, act), count in sorted(type_action.items()) if name == kind.value
        ]
        for act, count in actions or [(Action.REDACT.value, 0)]:
            lines.append(f"| {kind.value} | {act} | {count} |")

    lines += [
        "",
        "## Manual review",
        "",
        "Sample the scrubbed output with the command below (substitute local paths):",
        "",
        "```sh",
        "python -m terser.data.pii sample --in <scrubbed.jsonl> --out <review.csv> "
        "--text-field <field> --sample-size 500 --seed " + str(seed),
        "```",
        "",
        "The sampler requires at least 500 rows and leaves reviewer columns blank.",
        "Read every scrubbed field, including row_json, for all 500 rows; mark residual PII",
        "as 0 or 1 and record reviewer, date, false positives and total residual PII here.",
        "Pass requires 500 reviewed rows and zero residual PII. Keep 2.1.3 pending until then.",
    ]

    lines += ["", "## Suppressed candidates (allowlist / context heuristic)", ""]
    if suppressed:
        lines += ["| Entity type | Backend | Reason | Count |", "| :--- | :--- | :--- | ---: |"]
        for (etype, be, reason), n in sorted(suppressed.items()):
            lines.append(f"| {etype} | {be} | {reason} | {n} |")
    else:
        lines.append("None.")

    lines += ["", "## Actionable detections", ""]
    lines.append("See aggregate counts above; raw text and record identifiers are omitted.")
    return "\n".join(lines) + "\n"


SAMPLE_COLUMNS = (
    "id",
    "original",
    "scrubbed",
    "detections",
    "reviewer_pii_found",
    "reviewer_notes",
)


def write_sample_csv(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    results: Sequence[ScrubResult],
    sample_size: int,
    seed: int,
) -> list[int]:
    """Write the manual-review sample CSV and return the sampled indices.

    The reviewer columns are left blank for a human to fill in.
    """
    idx = sample_indices(len(results), sample_size, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(SAMPLE_COLUMNS)
        for i in idx:
            res = results[i]
            writer.writerow(
                [
                    _record_id(records[i], i),
                    res.original,
                    "" if res.scrubbed is None else res.scrubbed,
                    json.dumps(res.detections_json(), ensure_ascii=False),
                    "",
                    "",
                ]
            )
    return idx


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{lineno}: expected a JSON object")
                records.append(record)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON") from exc
    return records


def _load_policy(path: str | None) -> dict[str, str] | None:
    if path is None:
        return None
    raw = Path(path).read_text(encoding="utf-8")
    if path.endswith((".yaml", ".yml")):
        import yaml

        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"policy file {path} must contain a mapping of entity type to action")
    return {str(k): str(v) for k, v in data.items()}


def _git_provenance() -> tuple[str, bool]:
    """Read local revision metadata; git is optional outside a checkout."""
    root = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", False


def _distinct_paths(paths: Sequence[Path]) -> None:
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("input, output and report paths must be distinct")


def _strict_cli(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m terser.data.pii")
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--text-field", default="prompt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("regex", "presidio"), default="regex")
    args = parser.parse_args(argv)
    _distinct_paths([args.inp, args.out, args.report])
    records = _read_jsonl(args.inp)
    detector, backend = make_detector(args.backend)
    # Validate the selected field before writing any output. All other string values are scanned.
    for i, record in enumerate(records):
        if not isinstance(record.get(args.text_field), str):
            raise ValueError(f"row {i + 1} lacks string field {args.text_field!r}")
    output: list[dict[str, Any]] = []
    results: list[ScrubResult] = []
    for record in records:
        row, field_results = scrub_record(record, detector, backend)
        detections = tuple(item for result in field_results for item in result.detections)
        suppressed = tuple(item for result in field_results for item in result.suppressed)
        status = RowStatus.REDACTED if detections else RowStatus.CLEAN
        result = ScrubResult("", str(row[args.text_field]), status, backend, detections, suppressed)
        row["pii"] = {"status": status.value, "backend": backend, "counts": result.counts}
        results.append(result)
        output.append(row)
    input_hash = hashlib.sha256(args.inp.read_bytes()).hexdigest()
    sha, dirty = _git_provenance()
    report = render_report(
        results,
        source=args.inp.as_posix(),
        field_name=args.text_field,
        backend=backend,
        sample_path=None,
        sample_size=500,
        seed=args.seed,
        input_hash=input_hash,
        git_sha=sha,
        git_dirty=dirty,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
        encoding="utf-8",
        newline="\n",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8", newline="\n")
    counts = Counter(d.entity_type.value for result in results for d, _ in result.detections)
    print(
        f"backend={backend} records={len(records)} "
        f"rows_touched={sum(result.status is RowStatus.REDACTED for result in results)} "
        f"seed={args.seed}"
    )
    print(
        "counts="
        + json.dumps({kind.value: counts[kind.value] for kind in EntityType}, sort_keys=True)
    )
    print(f"input_sha256={input_hash}")
    print(f"git_sha={sha} git_dirty={str(dirty).lower()}")
    print("human_review=pending")
    return 0


def _sample_cli(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m terser.data.pii sample")
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--text-field", default="prompt")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    _distinct_paths([args.inp, args.out])
    records = _read_jsonl(args.inp)
    if args.sample_size <= 0 or len(records) < args.sample_size:
        raise ValueError(
            f"sample requires at least {args.sample_size} rows and a positive sample size"
        )
    for i, record in enumerate(records):
        if not isinstance(record.get(args.text_field), str):
            raise ValueError(f"row {i + 1} lacks string field {args.text_field!r}")
    indices = sample_indices(len(records), args.sample_size, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["id", "scrubbed", "row_json", "reviewer_pii_found", "reviewer_notes"])
        for i in indices:
            record = records[i]
            writer.writerow(
                [
                    _record_id(record, i),
                    record[args.text_field],
                    json.dumps(record, ensure_ascii=False),
                    "",
                    "",
                ]
            )
    print(
        f"sample_rows={len(indices)} population={len(records)} seed={args.seed} "
        "human_review=pending"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``python -m terser.data.pii scan ...``)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "sample":
        return _sample_cli(argv[1:])
    if not argv or argv[0] != "scan":
        return _strict_cli(argv)
    parser = argparse.ArgumentParser(prog="python -m terser.data.pii", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="scrub a JSONL file and write report + review sample")
    scan.add_argument("--in", dest="inp", required=True, help="input JSONL")
    scan.add_argument("--field", default="prompt", help="text field to scrub (default: prompt)")
    scan.add_argument("--out", required=True, help="output JSONL (dropped rows omitted)")
    scan.add_argument("--report", required=True, help="markdown report path")
    scan.add_argument("--sample", help="manual-review sample CSV path")
    scan.add_argument("--sample-size", type=int, default=500)
    scan.add_argument("--seed", type=int, default=0)
    scan.add_argument("--backend", choices=("auto", "regex", "presidio"), default="regex")
    scan.add_argument("--policy", help="JSON/YAML mapping entity type -> action (overrides)")
    args = parser.parse_args(argv)

    in_path = Path(args.inp)
    records = _read_jsonl(in_path)
    detector, backend = make_detector(args.backend)
    policy = _load_policy(args.policy)
    results = scan_records(records, args.field, detector, backend, policy)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec, res in zip(records, results, strict=True):
            if res.scrubbed is None:
                continue
            row = dict(rec)
            row[args.field] = res.scrubbed
            row["pii"] = {
                "status": res.status.value,
                "backend": backend,
                "detections": res.detections_json(),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.sample:
        write_sample_csv(Path(args.sample), records, results, args.sample_size, args.seed)

    report = render_report(
        results,
        source=in_path.as_posix(),
        field_name=args.field,
        backend=backend,
        sample_path=Path(args.sample).as_posix() if args.sample else None,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")

    counts = Counter(r.status.value for r in results)
    print(
        f"backend={backend} records={len(results)} "
        + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
