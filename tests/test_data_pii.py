"""Tests for ``terser.data.pii`` (plan row 2.1.3, policy in ``docs/pii_policy.md``).

Credential-shaped fixtures are assembled at runtime (``"gh" + "p_" + ...``) so that secret
scanners on the repository do not mistake the test file for a leak.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terser.data import pii
from terser.data.pii import (
    DEFAULT_POLICY,
    Action,
    EntityType,
    RowStatus,
    detect_regex,
    iban_valid,
    luhn_valid,
    private_person_context,
    scrub,
)


def active(text: str) -> list[tuple[EntityType, str]]:
    """Unsuppressed regex candidates as (type, matched text)."""
    return [(d.entity_type, d.text) for d in detect_regex(text) if d.suppressed is None]


def suppressed(text: str) -> list[tuple[EntityType, str | None]]:
    """Allowlisted regex candidates as (type, reason)."""
    return [(d.entity_type, d.suppressed) for d in detect_regex(text) if d.suppressed]


def types(text: str) -> set[EntityType]:
    return {t for t, _ in active(text)}


# --------------------------------------------------------------------------------------------
# Validators
# --------------------------------------------------------------------------------------------


def test_luhn() -> None:
    assert luhn_valid("4539148803436467")
    assert not luhn_valid("4539148803436468")
    assert not luhn_valid("12ab")
    assert not luhn_valid("7")


def test_iban_mod97_and_length() -> None:
    assert iban_valid("DE44 5001 0517 5407 3249 31")
    assert not iban_valid("DE44 5001 0517 5407 3249 32")  # checksum
    assert not iban_valid("DE44500105175407324")  # wrong length for DE


def test_shannon_entropy() -> None:
    assert pii.shannon_entropy("") == 0.0
    assert pii.shannon_entropy("aaaa") == 0.0
    assert pii.shannon_entropy("abcd") == pytest.approx(2.0)


# --------------------------------------------------------------------------------------------
# Regex layer: one positive and one negative per entity type
# --------------------------------------------------------------------------------------------


class TestEmail:
    def test_real_address_detected(self) -> None:
        assert active("mail me at dana.reyes@gmail.com please") == [
            (EntityType.EMAIL, "dana.reyes@gmail.com")
        ]

    @pytest.mark.parametrize(
        "text",
        ["alice@example.com", "bob@mail.example.org", "ops@staging.test", "x@foo.invalid"],
    )
    def test_rfc2606_domains_allowlisted(self, text: str) -> None:
        assert active(text) == []
        assert suppressed(text) == [(EntityType.EMAIL, "reserved_example_domain")]

    def test_placeholder_and_git_remote(self) -> None:
        assert suppressed("your.email@yourdomain.com") == [(EntityType.EMAIL, "placeholder_email")]
        assert suppressed("git clone git@github.com:org/repo.git") == [
            (EntityType.EMAIL, "git_ssh_remote")
        ]

    def test_code_is_not_an_email(self) -> None:
        assert detect_regex("y = weights@inputs.shape") == []
        assert detect_regex("@pytest.fixture\ndef client(): ...") == []


class TestPhone:
    @pytest.mark.parametrize(
        ("text", "number"),
        [
            ("call +44 20 7123 4567 today", "+44 20 7123 4567"),
            ("Tel +49 30 12345678", "+49 30 12345678"),
            ("my number is (212) 867-5309", "(212) 867-5309"),
            ("text me on 212.867.5309", "212.867.5309"),
            ("call 867-5309 after six", "867-5309"),
        ],
    )
    def test_detected(self, text: str, number: str) -> None:
        assert active(text) == [(EntityType.PHONE, number)]

    @pytest.mark.parametrize(
        ("text", "reason"),
        [
            ("like 555-123-4567", "fictional_555_number"),
            ("call 555-0142", "fictional_555_number"),
            ("ring +44 20 7946 0958", "ofcom_drama_number"),
        ],
    )
    def test_fictional_numbers_allowlisted(self, text: str, reason: str) -> None:
        assert active(text) == []
        assert suppressed(text) == [(EntityType.PHONE, reason)]

    @pytest.mark.parametrize(
        "text",
        [
            "between 250-1000 items",
            "released 2024-01-15",
            "pip install numpy==1.26.4",
            "the range 100-2000 feet",
            "ISBN 978-0-306-40615-7",
        ],
    )
    def test_negatives(self, text: str) -> None:
        assert EntityType.PHONE not in types(text)


class TestCreditCard:
    def test_luhn_valid_card_detected(self) -> None:
        assert active("card 4539 1488 0343 6467 exp 04/27") == [
            (EntityType.CREDIT_CARD, "4539 1488 0343 6467")
        ]

    def test_luhn_failure_not_detected(self) -> None:
        assert detect_regex("card 4539 1488 0343 6468 exp 04/27") == []

    def test_public_test_card_allowlisted(self) -> None:
        assert suppressed("use 4242 4242 4242 4242 in Stripe test mode") == [
            (EntityType.CREDIT_CARD, "public_test_card")
        ]

    def test_iban_digits_are_not_a_card(self) -> None:
        # The digit groups of this IBAN happen to pass Luhn.
        assert active("IBAN DE44 5001 0517 5407 3249 31") == [
            (EntityType.IBAN, "DE44 5001 0517 5407 3249 31")
        ]


class TestIban:
    def test_valid_iban_detected(self) -> None:
        assert types("pay to GB33BUKB20201555555555 by Friday") == {EntityType.IBAN}

    def test_invalid_checksum_ignored(self) -> None:
        assert detect_regex("pay to GB34BUKB20201555555555 by Friday") == []

    def test_documented_example_allowlisted(self) -> None:
        assert suppressed("e.g. DE89 3704 0044 0532 0130 00") == [
            (EntityType.IBAN, "documented_example_iban")
        ]


class TestIpAddress:
    def test_public_ipv4_and_ipv6_detected(self) -> None:
        text = "blocked 93.184.216.34 and 2a00:1450:4001:82a::200e"
        assert active(text) == [
            (EntityType.IP_ADDRESS, "93.184.216.34"),
            (EntityType.IP_ADDRESS, "2a00:1450:4001:82a::200e"),
        ]

    @pytest.mark.parametrize(
        ("text", "reason"),
        [
            ("app.run(host='127.0.0.1')", "loopback_ip"),
            ("bind 0.0.0.0:8080", "unspecified_ip"),
            ("router at 192.168.1.1", "private_or_documentation_ip"),
            ("see 203.0.113.9 in RFC 5737", "private_or_documentation_ip"),
            ("dns 8.8.8.8", "public_dns_resolver"),
            ("link-local fe80::1", "link_local_ip"),
            ("doc range 2001:db8::1", "private_or_documentation_ip"),
            ("upgrade to version 1.2.3.4", "version_string"),
        ],
    )
    def test_code_and_non_personal_ips_allowlisted(self, text: str, reason: str) -> None:
        assert active(text) == []
        assert suppressed(text) == [(EntityType.IP_ADDRESS, reason)]

    @pytest.mark.parametrize(
        "text",
        [
            "python 3.11.4 and numpy 1.26",
            "section 1.2.3 of the spec",
            "at 12:30:45 sharp",
            "mac 00:1A:2B:3C:4D:5E",
            "std::vector<int> and a1::b2",
            "999.1.1.1 is not an address",
        ],
    )
    def test_version_strings_and_code_not_ips(self, text: str) -> None:
        assert detect_regex(text) == []


class TestCredentials:
    def test_url_userinfo_detected_once(self) -> None:
        assert active("postgres://admin:S3cr3tP4ss@db.internal:5432/app") == [
            (EntityType.URL_CREDENTIAL, "admin:S3cr3tP4ss")
        ]

    def test_url_placeholder_password_allowlisted(self) -> None:
        assert suppressed("postgres://user:password@localhost/app") == [
            (EntityType.URL_CREDENTIAL, "placeholder_credential")
        ]

    def test_query_token_detected(self) -> None:
        found = active("https://api.x.com/v1?api_key=9fK2mQ7xLp4Zr8Tt&x=1")
        assert (EntityType.URL_CREDENTIAL, "9fK2mQ7xLp4Zr8Tt") in found
        assert all(t != "9fK2mQ7xLp4Zr8Tt&x=1" for _, t in found)

    @pytest.mark.parametrize(
        "token",
        [
            "AK" + "IA" + "Q3EGRTZ7WXN4P2LM",
            "gh" + "p_" + "aB3dE5fG7hI9jK1lM3nO5pQ7rS9tU1vW3xY5",
            "xo" + "xb-" + "2048151623-Zq8wLm3kRt",
            "sk" + "-proj-" + "Ab12Cd34Ef56Gh78Ij90Kl12",
            "AI" + "za" + "SyB2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R",
            "sk" + "_live_" + "4eC39HqLyjWDarjtT1zdp7dc",
        ],
    )
    def test_api_keys_detected(self, token: str) -> None:
        assert EntityType.API_KEY in types(f"my key is {token} ok")

    def test_documented_aws_example_key_allowlisted(self) -> None:
        text = "AK" + "IAIOSFODNN7" + "EXAMPLE"
        assert suppressed(text) == [(EntityType.API_KEY, "placeholder_credential")]

    def test_private_key_header(self) -> None:
        header = "-----BEGIN RSA " + "PRIVATE" + " KEY-----"  # split so scanners skip this file
        assert EntityType.API_KEY in types(header + "\nMIIE...")

    def test_quoted_secret_assignment_detected(self) -> None:
        assert active('api_key = "Zx81Kq03LmP9"') == [(EntityType.SECRET, "Zx81Kq03LmP9")]

    @pytest.mark.parametrize(
        ("text", "reason"),
        [
            ("api_key = os.environ.get('KEY')", "identifier_not_secret"),
            ("token = tokenizer(text)", "identifier_not_secret"),
            ("password = get_password_from_env()", "identifier_not_secret"),
            ("api_key = 'YOUR_API_KEY'", "placeholder_credential"),
            ('password: "changeme"', "placeholder_credential"),
        ],
    )
    def test_code_and_placeholders_not_secrets(self, text: str, reason: str) -> None:
        assert active(text) == []
        assert (EntityType.SECRET, reason) in suppressed(text)

    def test_high_entropy_token_flagged_but_hex_digest_not(self) -> None:
        blob = "Q2hhbmdlZDEyM0FiY2RlZmdoaWprbG1ub3BxcnN0dXZ3eHl6"
        assert types(f"blob {blob}") == {EntityType.HIGH_ENTROPY_TOKEN}
        assert detect_regex("commit a94a8fe5ccb19ba61c4c0873d391e987982fbbd3") == []
        assert detect_regex("see src/terser/data/pii/some_long_module_name_here.py") == []


class TestSsnAndAddress:
    def test_ssn_detected(self) -> None:
        assert active("SSN 219-09-9999") == [(EntityType.US_SSN, "219-09-9999")]

    def test_ssn_negatives(self) -> None:
        assert suppressed("e.g. 123-45-6789") == [(EntityType.US_SSN, "example_ssn")]
        assert detect_regex("000-12-3456 and 666-12-3456 and 900-12-3456") == []

    def test_street_address_detected(self) -> None:
        assert active("I live at 18 Larkspur Lane, Apt 4") == [
            (EntityType.STREET_ADDRESS, "18 Larkspur Lane, Apt 4")
        ]

    def test_famous_and_fictional_addresses_allowlisted(self) -> None:
        for text in ("221B Baker Street", "1600 Pennsylvania Avenue", "42 Wallaby Way"):
            assert suppressed(text) == [
                (EntityType.STREET_ADDRESS, "famous_or_fictional_address")
            ], text

    def test_address_negatives(self) -> None:
        assert detect_regex("chapter 3 covers loops; the 3 Musketeers") == []
        assert detect_regex("take 2 lanes on the highway") == []


def test_plain_prompts_have_no_candidates() -> None:
    prompts = [
        "write a python function that checks if a string is a palindrome",
        "what's the capital of Australia? i thought it was sydney lol",
        "SELECT id, email FROM users WHERE created_at > NOW() - INTERVAL '7 days';",
        "convert 90 km/h to m/s, and 1 mile = 1.609 km",
    ]
    for p in prompts:
        assert detect_regex(p) == [], p


# --------------------------------------------------------------------------------------------
# Person context heuristic
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "name", "private"),
    [
        ("contact my colleague Arvid, who covers", "Arvid", True),
        ("one of my coworkers, Delphine, is retiring", "Delphine", True),
        ("a thank you for my kid's teacher, Mr. Okafor.", "Okafor", True),
        ("Priya, my roommate, snores", "Priya", True),
        ('"Hi Gwendolyn, I wanted to', "Gwendolyn", True),
        ("hey!! im Jonah, i love dogs", "Jonah", True),
        ("name: Wrenna Holt age: 27", "Wrenna Holt", True),
        ("I met a guy named Tom", "Tom", True),
        ("who wrote Frankenstein? Mary Shelley?", "Mary Shelley", False),
        ("is it Adam Smith or John Maynard Keynes?", "Adam Smith", False),
        ("a story about a dragon named Ember", "Ember", False),
        ("Alice is taller than Bob. Bob is taller than Carol.", "Bob", False),
    ],
)
def test_private_person_context(text: str, name: str, private: bool) -> None:
    start = text.index(name)
    assert (private_person_context(text, start, start + len(name)) is not None) is private


def test_person_next_to_contact_data_is_private() -> None:
    text = "ask Fenwick at fenwick@caterers.co.uk"
    others = [d for d in detect_regex(text) if d.suppressed is None]
    assert private_person_context(text, 4, 11, others) == "co_occurs_with_pii"


# --------------------------------------------------------------------------------------------
# Policy actions
# --------------------------------------------------------------------------------------------


def test_default_policy_covers_every_entity_type() -> None:
    assert set(DEFAULT_POLICY) == set(EntityType)


def test_redact_replaces_with_typed_placeholder() -> None:
    res = scrub("mail dana.reyes@gmail.com or call (212) 867-5309")
    assert res.status is RowStatus.REDACTED
    assert res.scrubbed == "mail <EMAIL> or call <PHONE>"
    assert [a for _, a in res.detections] == [Action.REDACT, Action.REDACT]
    assert res.backend == "regex"


def test_drop_row() -> None:
    res = scrub("my card is 4539 1488 0343 6467 and email a.b@gmail.com")
    assert res.status is RowStatus.DROPPED
    assert res.scrubbed is None


def test_flag_for_review_keeps_text() -> None:
    text = "ship it to 18 Larkspur Lane please"
    res = scrub(text)
    assert res.status is RowStatus.FLAGGED
    assert res.scrubbed == text


def test_clean_and_suppressed_rows() -> None:
    res = scrub("use alice@example.com and 127.0.0.1")
    assert res.status is RowStatus.CLEAN
    assert res.detections == ()
    assert {d.suppressed for d in res.suppressed} == {"reserved_example_domain", "loopback_ip"}


def test_policy_overrides_accept_strings() -> None:
    text = "mail dana.reyes@gmail.com"
    assert scrub(text, policy={"EMAIL": "drop_row"}).status is RowStatus.DROPPED
    flagged = scrub(text, policy={EntityType.EMAIL: Action.FLAG_FOR_REVIEW})
    assert flagged.status is RowStatus.FLAGGED and flagged.scrubbed == text
    ignored = scrub(text, policy={"EMAIL": "ignore"})
    assert ignored.status is RowStatus.CLEAN and ignored.scrubbed == text
    with pytest.raises(ValueError):
        scrub(text, policy={"EMAIL": "shred"})


def test_detections_json_has_no_matched_text() -> None:
    res = scrub("mail dana.reyes@gmail.com")
    dumped = json.dumps(res.detections_json())
    assert "dana" not in dumped
    assert res.detections_json()[0]["action"] == "redact"


def test_make_detector_validates_backend() -> None:
    detector, name = pii.make_detector("regex")
    assert name == "regex" and detector is detect_regex
    with pytest.raises(ValueError):
        pii.make_detector("spacy")


def test_presidio_backend_falls_back_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pii, "presidio_available", lambda: False)
    assert pii.make_detector("auto")[1] == "regex"
    assert pii.make_detector("presidio")[1] == "regex"


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


RECORDS = [
    {"id": "r1", "prompt": "write a haiku about autumn"},
    {"id": "r2", "prompt": "email dana.reyes@gmail.com about the offsite"},
    {"id": "r3", "prompt": "charge 4539 1488 0343 6467 for the order"},
    {"id": "r4", "prompt": "deliver to 18 Larkspur Lane"},
    {"id": "r5", "prompt": "docs use alice@example.com"},
]


def _run_cli(tmp_path: Path, name: str) -> tuple[Path, Path]:
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n", encoding="utf-8")
    out, report = tmp_path / f"{name}.jsonl", tmp_path / f"{name}.md"
    code = pii.main(
        [
            "scan", "--in", str(src), "--field", "prompt", "--out", str(out),
            "--report", str(report), "--seed", "0", "--backend", "regex",
        ]
    )  # fmt: skip
    assert code == 0
    return out, report


def test_cli_scan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out, report = _run_cli(tmp_path, "a")
    assert "backend=regex records=5" in capsys.readouterr().out

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    by_id = {r["id"]: r for r in rows}
    assert set(by_id) == {"r1", "r2", "r4", "r5"}  # r3 dropped (card)
    assert by_id["r2"]["prompt"] == "email <EMAIL> about the offsite"
    assert by_id["r2"]["pii"]["status"] == "redacted"
    assert by_id["r2"]["pii"]["backend"] == "regex"
    assert by_id["r4"]["pii"]["status"] == "flagged"
    assert by_id["r5"]["pii"]["status"] == "clean"
    assert b"\r\n" not in out.read_bytes()

    md = report.read_text(encoding="utf-8")
    assert "- Backend: `regex`" in md
    assert "| dropped | 1 |" in md and "| redacted | 1 |" in md and "| flagged | 1 |" in md
    assert "reserved_example_domain" in md
    assert "dana.reyes" not in md and "4539" not in md  # no raw PII in the report


def test_cli_scan_is_deterministic(tmp_path: Path) -> None:
    first, _ = _run_cli(tmp_path, "a")
    second, _ = _run_cli(tmp_path, "b")
    assert first.read_bytes() == second.read_bytes()


def test_cli_policy_file_and_missing_field(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(RECORDS[1]) + "\n", encoding="utf-8")
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"EMAIL": "drop_row"}), encoding="utf-8")
    out = tmp_path / "out.jsonl"
    args = ["scan", "--in", str(src), "--out", str(out), "--report", str(tmp_path / "r.md")]
    assert pii.main([*args, "--backend", "regex", "--policy", str(policy)]) == 0
    assert out.read_text(encoding="utf-8") == ""
    with pytest.raises(KeyError):
        pii.main([*args, "--backend", "regex", "--field", "missing"])


# --------------------------------------------------------------------------------------------
# Presidio backend (skipped when presidio-analyzer / en_core_web_sm are not installed)
# --------------------------------------------------------------------------------------------

requires_presidio = pytest.mark.skipif(
    not pii.presidio_available(), reason="presidio-analyzer / en_core_web_sm not installed"
)


@pytest.fixture(scope="module")
def presidio_detector() -> pii.Detector:
    detector, name = pii.make_detector("presidio")
    assert name == "presidio"
    return detector


@requires_presidio
@pytest.mark.slow
def test_presidio_private_person_flagged(presidio_detector: pii.Detector) -> None:
    text = "Please forward this to my colleague Arvid Lindqvist before Friday."
    res = scrub(text, presidio_detector, backend="presidio")
    persons = [(d, a) for d, a in res.detections if d.entity_type is EntityType.PERSON]
    assert persons and all(d.backend == "presidio" for d, _ in persons)
    assert res.status is RowStatus.FLAGGED
    assert res.scrubbed == text


@requires_presidio
@pytest.mark.slow
def test_presidio_public_figure_suppressed(presidio_detector: pii.Detector) -> None:
    res = scrub(
        "Who was the father of modern economics, Adam Smith or John Maynard Keynes?",
        presidio_detector,
        backend="presidio",
    )
    assert res.status is RowStatus.CLEAN
    assert res.backend == "presidio"
    assert {d.suppressed for d in res.suppressed if d.entity_type is EntityType.PERSON} == {
        "no_private_context"
    }


@requires_presidio
@pytest.mark.slow
def test_presidio_keeps_regex_layer(presidio_detector: pii.Detector) -> None:
    res = scrub("mail dana.reyes@gmail.com", presidio_detector, backend="presidio")
    assert res.scrubbed == "mail <EMAIL>"
    assert {d.backend for d, _ in res.detections} == {"regex"}
