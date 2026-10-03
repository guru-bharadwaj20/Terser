"""Strict ingestion regressions for plan row 2.1.3; synthetic fixtures only."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from terser.data import pii
from terser.data.collect import collect


@pytest.mark.parametrize(
    ("text", "expected", "kind"),
    [
        ("mail alice@example.com", "mail <EMAIL>", "EMAIL"),
        ("call (212) 867-5309", "call <PHONE>", "PHONE"),
        ("host 192.168.1.9", "host <IP_ADDRESS>", "IP_ADDRESS"),
        ("host ::dead:beef", "host <IP_ADDRESS>", "IP_ADDRESS"),
        ("host ::ffff:192.0.2.128", "host <IP_ADDRESS>", "IP_ADDRESS"),
        ("card 4539 1488 0343 6467", "card <CREDIT_CARD>", "CREDIT_CARD"),
        ("card 4242 4242 4242 4242", "card <CREDIT_CARD>", "CREDIT_CARD"),
        ("pay DE44 5001 0517 5407 3249 31", "pay <IBAN>", "IBAN"),
        ("pay de44500105175407324931", "pay <IBAN>", "IBAN"),
        ("ship to 18 larkspur lane", "ship to <STREET_ADDRESS>", "STREET_ADDRESS"),
        (
            "https://user:password@host.test/a",
            "https://<URL_CREDENTIAL>@host.test/a",
            "URL_CREDENTIAL",
        ),
        (
            "https://host.test/?token=abc",
            "https://host.test/?token=<URL_CREDENTIAL>",
            "URL_CREDENTIAL",
        ),
        (
            "https://host.test/#access_token=Ab19%2Fxy",
            "https://host.test/#access_token=<URL_CREDENTIAL>",
            "URL_CREDENTIAL",
        ),
        ("AK" + "IA" + "Q3EGRTZ7WXN4P2LM", "<API_KEY>", "API_KEY"),
        ("gh" + "p_" + "aB3dE5fG7hI9jK1lM3nO5pQ7rS9tU1vW3xY5", "<API_KEY>", "API_KEY"),
        ("sk" + "-proj-" + "Ab12Cd34Ef56Gh78Ij90Kl12", "<API_KEY>", "API_KEY"),
        (
            "Q2hhbmdlZDEyM0FiY2RlZmdoaWprbG1ub3BxcnN0dXZ3eHl6",
            "<HIGH_ENTROPY_TOKEN>",
            "HIGH_ENTROPY_TOKEN",
        ),
    ],
)
def test_strict_redaction(text: str, expected: str, kind: str) -> None:
    result = pii.scrub_text(text)
    assert result.scrubbed == expected
    assert result.counts == {kind: 1}
    assert pii.scrub_text(result.scrubbed).scrubbed == expected


@pytest.mark.parametrize(
    "text",
    [
        "upgrade to version 1.2.3.4",
        "pip install pkg==1.2.3.4",
        "python 3.11.4",
        "999.1.1.1",
        "std::vector<int> and a1::b2",
        "12:30:45",
        "compute +12345678 + 23456789",
        "compute 212 - 867 - 5309 = -5988",
        "x = 0xA94A8FE5CCB19BA61C4C0873D391E987982FBBD3",
        "card 4539 1488 0343 6468",
        "pay DE44500105175407324932",
        "chapter 3 covers loops",
        "https://host.test/?page=123",
        "token = tokenizer(text)",
        "y = weights@inputs.shape",
    ],
)
def test_strict_tricky_negatives(text: str) -> None:
    result = pii.scrub_text(text)
    assert result.scrubbed == text
    assert result.counts == {}


def test_collect_scrubs_reference_and_payload_before_interim() -> None:
    rows, counts = collect(
        [
            {
                "source_id": "s:1",
                "source": "s",
                "category_hint": "fact",
                "instruction": "Explain this contact record please.",
                "input": "email a.b@example.com",
                "reference": "Call (212) 867-5309 or visit 18 Larkspur Lane.",
            }
        ],
        per_category=1,
    )
    assert len(rows) == 1
    assert rows[0]["input"] == "email <EMAIL>"
    assert rows[0]["reference"] == "Call <PHONE> or visit <STREET_ADDRESS>."
    assert counts["pii:EMAIL"] == counts["pii:PHONE"] == counts["pii:STREET_ADDRESS"] == 1


def test_direct_cli_scrubs_all_strings_and_records_provenance(tmp_path: Path) -> None:
    source, output, report = [tmp_path / name for name in ("in.jsonl", "out.jsonl", "report.md")]
    source.write_text(
        json.dumps(
            {
                "prompt": "mail a.b@example.com",
                "reference": "18 Larkspur Lane",
                "nested": ["(212) 867-5309", 42],
            }
        )
        + "\n"
    )
    assert (
        pii.main(
            [
                "--in",
                str(source),
                "--out",
                str(output),
                "--report",
                str(report),
                "--text-field",
                "prompt",
                "--seed",
                "7",
            ]
        )
        == 0
    )
    row = json.loads(output.read_text())
    assert row["prompt"] == "mail <EMAIL>"
    assert row["reference"] == "<STREET_ADDRESS>"
    assert row["nested"] == ["<PHONE>", 42]
    md = report.read_text()
    assert hashlib.sha256(source.read_bytes()).hexdigest() in md
    assert "Git SHA:" in md and "Seed: 7" in md and "Rows touched: 1" in md
    assert "| EMAIL | redact | 1 |" in md and "| PHONE | redact | 1 |" in md
    assert "a.b@" not in md and "Larkspur" not in md
    assert "Human review: pending" in md


def test_report_never_leaks_adjacent_entities() -> None:
    records = [{"id": "private.person@example.com", "prompt": "a.b@gmail.com c.d@gmail.com"}]
    results = pii.scan_records(records, "prompt", pii.detect_regex, "regex")
    md = pii.render_report(
        records,
        results,
        source="input.jsonl",
        field_name="prompt",
        backend="regex",
        sample_path=None,
        sample_size=500,
        seed=0,
    )
    assert "gmail.com" not in md and "private.person" not in md


def test_seeded_sample_requires_500_rows_and_retains_review_columns(tmp_path: Path) -> None:
    source, first, second = [tmp_path / name for name in ("in.jsonl", "a.csv", "b.csv")]
    source.write_text(
        "".join(json.dumps({"id": str(i), "prompt": f"clean row {i}"}) + "\n" for i in range(600))
    )
    for output in (first, second):
        assert (
            pii.main(
                [
                    "sample",
                    "--in",
                    str(source),
                    "--out",
                    str(output),
                    "--text-field",
                    "prompt",
                    "--seed",
                    "7",
                    "--sample-size",
                    "500",
                ]
            )
            == 0
        )
    assert first.read_bytes() == second.read_bytes()
    with first.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 500 and len({r["id"] for r in rows}) == 500
    assert all(r["reviewer_pii_found"] == "" for r in rows)
    assert rows[0]["scrubbed"].startswith("clean row ")
    source.write_text('{"prompt":"too few"}\n')
    with pytest.raises(ValueError, match="500"):
        pii.main(["sample", "--in", str(source), "--out", str(first)])


def test_default_backend_does_not_probe_optional_ner(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_probe() -> bool:
        raise AssertionError("default backend must not probe optional dependencies")

    monkeypatch.setattr(pii, "presidio_available", forbidden_probe)
    assert pii.make_detector()[1] == "regex"


def test_optional_ner_missing_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pii, "presidio_available", lambda: False)
    assert pii.make_detector("presidio")[1] == "regex"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Call 9876543210 today", "Call <PHONE> today"),
        ("card 4539-1488 0343-6467", "card <CREDIT_CARD>"),
        ("IBAN DE44 5001 0517 5407 3249 31 NEXT", "IBAN <IBAN> NEXT"),
        ("Visit 18 Larkspur Lane.", "Visit <STREET_ADDRESS>."),
    ],
)
def test_strict_detector_boundaries(text: str, expected: str) -> None:
    assert pii.scrub_text(text).scrubbed == expected


@pytest.mark.parametrize("payload", ["[]\n", "null\n", "{bad}\n", '{"other":"text"}\n'])
def test_cli_rejects_invalid_input_before_writing(tmp_path: Path, payload: str) -> None:
    source, output, report = [tmp_path / name for name in ("in.jsonl", "out.jsonl", "r.md")]
    source.write_text(payload)
    with pytest.raises(ValueError):
        pii.main(["--in", str(source), "--out", str(output), "--report", str(report)])
    assert not output.exists() and not report.exists()


def test_cli_refuses_input_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "in.jsonl"
    source.write_text('{"prompt":"clean"}\n')
    with pytest.raises(ValueError, match="distinct"):
        pii.main(["--in", str(source), "--out", str(source), "--report", str(tmp_path / "r.md")])
    assert source.read_text() == '{"prompt":"clean"}\n'


def test_street_prepositions_are_not_addresses() -> None:
    text = "Please park on level 3 across the street."
    assert pii.scrub_text(text).scrubbed == text


def test_report_lists_zero_counts() -> None:
    md = pii.render_report(
        [{"prompt": "clean"}],
        [pii.scrub_text("clean")],
        source="input",
        field_name="prompt",
        backend="regex",
        sample_path=None,
        sample_size=500,
        seed=0,
    )
    assert "| EMAIL | redact | 0 |" in md


def test_optional_ner_broken_installation_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pii, "presidio_available", lambda: True)

    def unavailable() -> None:
        raise OSError("local language resource missing")

    monkeypatch.setattr(pii, "_PresidioDetector", unavailable)
    assert pii.make_detector("presidio")[1] == "regex"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://dana@host.test/path", "https://<URL_CREDENTIAL>@host.test/path"),
        ("https://dana:p%40ssword@host.test/path", "https://<URL_CREDENTIAL>@host.test/path"),
        ("Park on level 3 Across The Street.", "Park on level 3 Across The Street."),
    ],
)
def test_url_userinfo_and_capitalized_prose(text: str, expected: str) -> None:
    assert pii.scrub_text(text).scrubbed == expected


def test_real_slice_uses_strict_redaction_before_export() -> None:
    from terser.data.real_slice import SliceSummary, build_records

    summary = SliceSummary()
    rows = build_records(
        [
            {
                "conversation_hash": "synthetic-1",
                "language": "English",
                "toxic": False,
                "redacted": False,
                "conversation": [
                    {
                        "role": "user",
                        "content": "Write a poem and deliver it to 18 Larkspur Lane please.",
                    }
                ],
            }
        ],
        summary,
    )
    assert len(rows) == 1
    assert rows[0]["prompt"] == "Write a poem and deliver it to <STREET_ADDRESS> please."
    assert summary.pii_scrubbed == 1


@pytest.mark.parametrize("address", ["::", "::2", "::a", "1::", "1:2:3:4:5:6:7:8"])
def test_short_unambiguous_ipv6(address: str) -> None:
    result = pii.scrub_text("host " + address)
    assert result.scrubbed == "host <IP_ADDRESS>"
    assert result.counts == {"IP_ADDRESS": 1}


def test_hex_literal_with_prefix_is_not_a_generic_secret() -> None:
    text = "value = 0x" + "0123456789ABCDEF" * 4
    assert pii.scrub_text(text).scrubbed == text


def test_explicit_secret_assignment_redacts_hex() -> None:
    text = 'api_key = "' + "0123456789abcdef" * 4 + '"'
    assert pii.scrub_text(text).scrubbed == 'api_key = "<SECRET>"'
