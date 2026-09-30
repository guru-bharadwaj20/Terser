"""Code extraction and sandboxed unit-test execution for the ``code`` category (plan row 2.3.1).

Two jobs:

1. **Extraction.** Pull the code a model wrote out of its free-form answer: fenced blocks (with
   language tags), unclosed fences from truncated answers, or bare code with prose around it.
   :func:`extract_code` is language aware: for Python it keeps tagged and untagged blocks that
   parse, picks the block defining the entry point, and adds top-level helpers and imports that
   other blocks define (so "helper in block 1, solution in block 2" still runs, while example-usage
   blocks are not executed).
2. **Execution.** :func:`run_code_tests` runs the extracted code against HumanEval-style
   (``check(candidate)``) or MBPP-style (list of ``assert`` statements) tests in a separate
   Python process and reports pass/fail per test plus the tail of stderr.

Sandbox (best effort)
---------------------
Each run gets:

* a **separate process** started with ``-I`` (isolated mode: ignores ``PYTHON*`` variables, user
  site-packages and the script directory) and ``-B``, using the base interpreter rather than the
  project virtualenv, so the project's packages and ``.env`` are not importable;
* a fresh **temporary directory** as working directory, ``HOME`` and ``TEMP``, deleted afterwards;
* a **scrubbed environment** (no API keys, tokens or proxies from the parent; proxies point at a
  closed port);
* a wall-clock **timeout**, after which the process (POSIX: its whole session) is killed;
* a **memory limit**: ``RLIMIT_AS`` on Linux (plus ``RLIMIT_CPU``, ``RLIMIT_FSIZE``,
  ``RLIMIT_CORE``); on Windows a Job Object with a process and job commit limit, an active-process
  limit of 1 (no child processes) and kill-on-close. On macOS ``RLIMIT_AS`` is often not enforced;
* **no network** (best effort): ``socket.socket``, ``socket.create_connection`` and
  ``socket.getaddrinfo`` raise inside the child;
* a HumanEval-style **reliability guard** that disables process creation (``subprocess.Popen``,
  ``os.system``, ``os.fork``, ``os.exec*``...) and destructive file operations (``os.remove``,
  ``shutil.rmtree``...).

Windows limitations: there is no ``resource`` module, so CPU-time and file-size limits are not
applied (the wall-clock timeout bounds both); the Job Object is attached by the child to itself,
so the interpreter's own start-up runs before the limit applies; the memory limit counts committed
memory, so a very small limit can make the interpreter fail to start; the network guard and the
reliability guard are Python-level monkeypatches that ``ctypes`` or re-imported C modules bypass.

**This is not a security boundary.** It stops honest mistakes by a model (infinite loops, runaway
allocation, stray ``rm -rf``-style calls, accidental network use) from harming the labelling run.
It does not contain deliberately hostile code: the child runs as the same user, can read any file
that user can read, and can undo every Python-level guard. Container / VM isolation with seccomp
and no network namespace (the Phase 11 hardening work) is out of scope for this module; do not run
untrusted third-party code through it.
"""

from __future__ import annotations

import ast
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "CodeBlock",
    "CodeRunResult",
    "CodeTests",
    "SandboxLimits",
    "default_python",
    "extract_code",
    "extract_code_blocks",
    "pass_rate",
    "run_code_tests",
    "run_output_tests",
]

RunStatus = Literal["passed", "failed", "error", "timeout", "no_code"]

PYTHON_ALIASES: frozenset[str] = frozenset({"python", "py", "python3", "py3", "ipython", "pycon"})
"""Fence language tags treated as Python."""

_LANGUAGE_ALIASES: dict[str, frozenset[str]] = {
    "python": PYTHON_ALIASES,
    "javascript": frozenset({"javascript", "js", "node", "jsx", "mjs"}),
    "typescript": frozenset({"typescript", "ts", "tsx"}),
    "java": frozenset({"java"}),
    "sql": frozenset({"sql", "postgresql", "postgres", "mysql", "sqlite", "tsql", "plsql"}),
    "bash": frozenset({"bash", "sh", "shell", "zsh", "console"}),
    "cpp": frozenset({"cpp", "c++", "cc", "cxx", "hpp"}),
    "c": frozenset({"c", "h"}),
    "csharp": frozenset({"csharp", "cs", "c#"}),
    "go": frozenset({"go", "golang"}),
    "rust": frozenset({"rust", "rs"}),
}

_OPEN_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*([^\s`]*)[^\n]*$", re.MULTILINE)
_CODE_START_RE = re.compile(r"^(?:def |async def |class |import |from \S+ import |@\w)")
_STDERR_TAIL_CHARS = 2000


def _canonical_language(tag: str | None) -> str | None:
    """Map a fence tag (``py``, ``Python3``, ``js``...) to a canonical language name."""
    if not tag:
        return None
    key = tag.strip().lower().lstrip("{.").rstrip("}")
    for name, aliases in _LANGUAGE_ALIASES.items():
        if key in aliases:
            return name
    return key


@dataclass(frozen=True)
class CodeBlock:
    """One fenced code block. ``language`` is canonical (``"python"``) or None if untagged."""

    language: str | None
    code: str
    closed: bool = True


def extract_code_blocks(text: str) -> list[CodeBlock]:
    """Return all fenced code blocks (backtick or tilde fences) in document order.

    A final fence that is opened but never closed (a truncated answer) yields a block with
    ``closed=False`` running to the end of the text.
    """
    if not isinstance(text, str):
        return []
    blocks: list[CodeBlock] = []
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        match = _OPEN_FENCE_RE.match(lines[i].rstrip("\r\n"))
        if not match:
            i += 1
            continue
        fence, tag = match.group(1), match.group(2)
        close_re = re.compile(rf"^[ \t]*{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$")
        body: list[str] = []
        j = i + 1
        while j < len(lines) and not close_re.match(lines[j].rstrip("\r\n")):
            body.append(lines[j])
            j += 1
        closed = j < len(lines)
        blocks.append(CodeBlock(_canonical_language(tag), "".join(body).rstrip() + "\n", closed))
        i = j + 1
    return blocks


def _parse_python(code: str) -> ast.Module | None:
    try:
        return ast.parse(code)
    except (SyntaxError, ValueError):
        return None


def _looks_like_python_module(tree: ast.Module) -> bool:
    """True if the module contains at least one definition, import or assignment."""
    kinds = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Import,
        ast.ImportFrom,
        ast.Assign,
        ast.AnnAssign,
    )
    return any(isinstance(node, kinds) for node in tree.body)


def _top_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _bare_python(text: str) -> str | None:
    """Find Python in unfenced text: the whole text, or the longest parseable run of lines."""
    tree = _parse_python(text)
    if tree is not None and _looks_like_python_module(tree):
        return text.strip() + "\n"
    lines = text.splitlines()
    starts = [k for k, line in enumerate(lines) if _CODE_START_RE.match(line)]
    for start in starts:
        for end in range(len(lines), start, -1):
            chunk = "\n".join(lines[start:end]).strip()
            tree = _parse_python(chunk)
            if tree is not None and _looks_like_python_module(tree):
                return chunk + "\n"
    return None


def _supplement(selected: str, others: Iterable[str]) -> str:
    """Prepend imports and top-level defs from ``others`` that ``selected`` does not define."""
    tree = _parse_python(selected)
    if tree is None:
        return selected
    defined = _top_level_names(tree)
    extra: list[str] = []
    for other in others:
        other_tree = _parse_python(other)
        if other_tree is None:
            continue
        for node in other_tree.body:
            keep = isinstance(node, ast.Import | ast.ImportFrom)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                keep = node.name not in defined
                if keep:
                    defined.add(node.name)
            if keep:
                segment = ast.get_source_segment(other, node, padded=False)
                if segment:
                    decorators = "".join(
                        f"@{ast.get_source_segment(other, d)}\n"
                        for d in getattr(node, "decorator_list", [])
                    )
                    extra.append(decorators + segment)
    if not extra:
        return selected
    return "\n\n".join(dict.fromkeys(extra)) + "\n\n\n" + selected


def extract_code(text: str, language: str = "python", entry_point: str | None = None) -> str | None:
    """Extract the code for ``language`` from a model answer, or None if there is none.

    Args:
        text: Free-form model output.
        language: Target language (canonical name or alias, e.g. ``"py"``).
        entry_point: For Python, the function/class the tests call. The last block defining it
            is chosen; otherwise the last block with a definition, otherwise the last block.

    Returns:
        The code (Python: selected block plus imports/helpers from the other Python blocks), or
        None. For Python, a block that does not parse is still returned when nothing parses, so
        the runner reports the syntax error instead of "no code".
    """
    if not isinstance(text, str) or not text.strip():
        return None
    lang = _canonical_language(language) or "python"
    blocks = extract_code_blocks(text)
    if lang != "python":
        tagged = [b.code for b in blocks if b.language == lang]
        untagged = [b.code for b in blocks if b.language is None]
        pool = tagged or untagged
        return pool[-1] if pool else None

    candidates = [
        b.code
        for b in blocks
        if b.language == "python" or (b.language is None and _parse_python(b.code) is not None)
    ]
    if not candidates:
        if any(b.language not in (None, "python") for b in blocks):
            return None
        return _bare_python(text)

    parsed = [(c, _parse_python(c)) for c in candidates]
    good = [(c, t) for c, t in parsed if t is not None]
    if not good:
        return candidates[-1]
    selected: str | None = None
    if entry_point:
        defining = [c for c, t in good if entry_point in _top_level_names(t)]
        if defining:
            selected = defining[-1]
    if selected is None:
        with_defs = [
            c
            for c, t in good
            if any(
                isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) for n in t.body
            )
        ]
        selected = with_defs[-1] if with_defs else good[-1][0]
    others = [c for c, _ in good if c is not selected]
    return _supplement(selected, others)


# --------------------------------------------------------------------------------------------
# Tests and results
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeTests:
    """Unit tests for one problem.

    ``style="humaneval"``: ``tests`` holds one source string defining ``check(candidate)``, called
    with the object named ``entry_point``. ``style="mbpp"``: ``tests`` holds one statement per test
    case (usually an ``assert``); each is run and scored separately. ``setup`` runs before the
    candidate code (MBPP ``test_setup_code`` / ``test_imports``).
    """

    style: Literal["humaneval", "mbpp"]
    tests: tuple[str, ...]
    entry_point: str | None = None
    setup: str = ""

    def __post_init__(self) -> None:
        if self.style not in ("humaneval", "mbpp"):
            raise ValueError(f"unknown test style {self.style!r}")
        if not self.tests or not all(isinstance(t, str) and t.strip() for t in self.tests):
            raise ValueError("tests must be a non-empty sequence of non-empty strings")
        if self.style == "humaneval":
            if len(self.tests) != 1:
                raise ValueError("humaneval tests take exactly one source string")
            if not self.entry_point:
                raise ValueError("humaneval tests need an entry_point")

    @classmethod
    def humaneval(cls, test: str, entry_point: str, setup: str = "") -> CodeTests:
        """HumanEval-style tests: ``test`` defines ``check(candidate)``."""
        return cls("humaneval", (test,), entry_point, setup)

    @classmethod
    def mbpp(
        cls, test_list: Sequence[str], setup: str = "", entry_point: str | None = None
    ) -> CodeTests:
        """MBPP-style tests: one assert statement per test case."""
        return cls("mbpp", tuple(test_list), entry_point, setup)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> CodeTests:
        """Build from a dataset record.

        Recognises HumanEval (``test`` + ``entry_point``) and MBPP (``test_list`` with optional
        ``test_setup_code`` / ``test_imports`` and ``entry_point``).
        """
        if "test_list" in record:
            setup_parts = [str(record.get("test_setup_code") or "")]
            setup_parts += [str(line) for line in record.get("test_imports") or []]
            setup = "\n".join(p for p in setup_parts if p.strip())
            return cls.mbpp(list(record["test_list"]), setup, record.get("entry_point"))
        if "test" in record and "entry_point" in record:
            return cls.humaneval(str(record["test"]), str(record["entry_point"]))
        raise ValueError("record has neither 'test_list' (MBPP) nor 'test' + 'entry_point'")

    @property
    def n_tests(self) -> int:
        """Number of scored test cases (HumanEval: 1)."""
        return len(self.tests)


@dataclass(frozen=True)
class SandboxLimits:
    """Resource limits for one run. ``memory_mb=None`` disables the memory limit."""

    timeout_s: float = 10.0
    memory_mb: int | None = 512
    max_output_bytes: int = 256_000
    block_network: bool = True

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if self.memory_mb is not None and self.memory_mb < 64:
            raise ValueError("memory_mb must be >= 64 (the interpreter itself needs memory)")
        if self.max_output_bytes < 0:
            raise ValueError("max_output_bytes must be >= 0")


@dataclass(frozen=True)
class CodeRunResult:
    """Outcome of running one piece of code against its tests."""

    status: RunStatus
    n_tests: int
    n_passed: int
    returncode: int | None = None
    stderr_tail: str = ""
    duration_s: float = 0.0
    per_test: tuple[bool, ...] = ()
    limits_applied: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        """True iff every test passed."""
        return self.status == "passed"

    @property
    def pass_rate(self) -> float:
        """Share of test cases passed (0.0 when there are no tests)."""
        return self.n_passed / self.n_tests if self.n_tests else 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form (includes ``passed`` and ``pass_rate``)."""
        data = asdict(self)
        data["per_test"] = list(self.per_test)
        data["limits_applied"] = list(self.limits_applied)
        data["passed"] = self.passed
        data["pass_rate"] = self.pass_rate
        return data


def pass_rate(results: Iterable[CodeRunResult]) -> float:
    """Share of runs in which every test passed (pass@1 over samples/problems); 0.0 if empty."""
    items = list(results)
    return sum(r.passed for r in items) / len(items) if items else 0.0


# --------------------------------------------------------------------------------------------
# Sandbox
# --------------------------------------------------------------------------------------------

_HARNESS = r"""
import json, os, sys, traceback

_payload_path, _result_path = sys.argv[1], sys.argv[2]
with open(_payload_path, encoding="utf-8") as _f:
    _payload = json.load(_f)
_out = open(_result_path, "w", encoding="utf-8")
_dumps, _write, _flush, _exit = json.dumps, _out.write, _out.flush, os._exit
_applied = []


def _emit(obj):
    _write(_dumps(obj) + "\n")
    _flush()


def _err_summary():
    return traceback.format_exc(limit=5)[-1500:]


def _limit_windows(memory_mb):
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = EXTENDED()
    # KILL_ON_JOB_CLOSE | DIE_ON_UNHANDLED_EXCEPTION | ACTIVE_PROCESS
    flags = 0x2000 | 0x400 | 0x8
    info.BasicLimitInformation.ActiveProcessLimit = 1
    if memory_mb:
        flags |= 0x100 | 0x200  # PROCESS_MEMORY | JOB_MEMORY
        info.ProcessMemoryLimit = info.JobMemoryLimit = memory_mb * 1024 * 1024
    info.BasicLimitInformation.LimitFlags = flags
    ok = k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
    if ok and k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        _applied.append("windows_job_object:no_child_processes")
        if memory_mb:
            _applied.append("memory:windows_job_object")
    return job


def _limit_posix(memory_mb, timeout_s, max_output):
    import resource

    def _set(name, value):
        limit = getattr(resource, name, None)
        if limit is None:
            return False
        try:
            soft, hard = resource.getrlimit(limit)
            if hard != resource.RLIM_INFINITY:
                value = min(value, hard)
            resource.setrlimit(limit, (value, hard))
            return True
        except (ValueError, OSError):
            return False

    if memory_mb and _set("RLIMIT_AS", memory_mb * 1024 * 1024):
        _applied.append("memory:rlimit_as")
    if _set("RLIMIT_CPU", int(timeout_s) + 1):
        _applied.append("cpu:rlimit_cpu")
    if _set("RLIMIT_FSIZE", max(max_output, 1_048_576) * 4):
        _applied.append("fsize:rlimit_fsize")
    _set("RLIMIT_CORE", 0)


_job = None
try:
    if os.name == "nt":
        _job = _limit_windows(_payload["memory_mb"])
    else:
        _limit_posix(_payload["memory_mb"], _payload["timeout_s"], _payload["max_output_bytes"])
except Exception:
    pass


class _Capped:
    def __init__(self, stream, limit):
        self._stream, self._limit, self._n = stream, limit, 0

    def write(self, s):
        room = self._limit - self._n
        if room > 0:
            chunk = s[:room]
            self._n += len(chunk)
            self._stream.write(chunk)
        return len(s)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


sys.stdout = _Capped(sys.stdout, _payload["max_output_bytes"])
sys.stderr = _Capped(sys.stderr, _payload["max_output_bytes"])

if _payload["block_network"]:
    import socket

    def _no_network(*args, **kwargs):
        raise OSError("network access is disabled in the Terser code sandbox")

    class _NoSocket(socket.socket):
        def __init__(self, *args, **kwargs):
            _no_network()

    socket.socket = _NoSocket
    socket.create_connection = _no_network
    socket.getaddrinfo = _no_network
    _applied.append("network:socket_guard")

import builtins, shutil, subprocess


def _disabled(*args, **kwargs):
    raise PermissionError("disabled in the Terser code sandbox")


for _mod, _names in (
    (os, ("system", "popen", "fork", "forkpty", "kill", "killpg", "execv", "execve", "execvp",
          "execvpe", "execl", "execle", "execlp", "execlpe", "spawnv", "spawnve", "spawnvp",
          "spawnvpe", "spawnl", "spawnle", "startfile", "remove", "unlink", "rmdir",
          "removedirs", "rename", "renames", "replace", "truncate", "chmod", "chown", "fchmod",
          "fchown", "lchown", "chroot", "setuid", "putenv", "unsetenv", "fchdir")),
    (shutil, ("rmtree", "move", "chown")),
    (subprocess, ("Popen", "run", "call", "check_call", "check_output", "getoutput",
                  "getstatusoutput")),
):
    for _name in _names:
        if hasattr(_mod, _name):
            setattr(_mod, _name, _disabled)
builtins.help = _disabled
_applied.append("reliability_guard")

_emit({"event": "start", "applied": _applied})
_ns = {"__name__": "__candidate__", "__builtins__": builtins}
try:
    if _payload["setup"].strip():
        exec(compile(_payload["setup"], "<setup>", "exec"), _ns)
    exec(compile(_payload["code"], "candidate.py", "exec"), _ns)
except BaseException:
    _summary = _err_summary()
    try:
        sys.__stderr__.write(_summary)
        sys.__stderr__.flush()
    except Exception:
        pass
    _emit({"event": "load_error", "error": _summary})
    _exit(3)

if _payload["style"] == "humaneval":
    try:
        exec(compile(_payload["tests"][0], "<test>", "exec"), _ns)
        _ns["check"](_ns[_payload["entry_point"]])
        _emit({"event": "test", "index": 0, "ok": True})
    except BaseException:
        _summary = _err_summary()
        sys.__stderr__.write(_summary)
        _emit({"event": "test", "index": 0, "ok": False, "error": _summary})
else:
    for _i, _test in enumerate(_payload["tests"]):
        try:
            exec(compile(_test, "<test %d>" % _i, "exec"), _ns)
            _emit({"event": "test", "index": _i, "ok": True})
        except BaseException:
            _summary = _err_summary()
            sys.__stderr__.write(_summary)
            _emit({"event": "test", "index": _i, "ok": False, "error": _summary})

_emit({"event": "done"})
try:
    sys.__stdout__.flush()
    sys.__stderr__.flush()
except Exception:
    pass
_exit(0)
"""


def default_python() -> str:
    """Interpreter for sandboxed runs: the base (non-virtualenv) interpreter when available.

    The base interpreter avoids the Windows venv launcher (an extra process) and keeps the project
    virtualenv's packages out of reach of candidate code.
    """
    base = getattr(sys, "_base_executable", None)
    if isinstance(base, str) and base and Path(base).is_file():
        return base
    return sys.executable


def _scrubbed_env(workdir: str) -> dict[str, str]:
    """A minimal environment: no credentials, proxies disabled, home/temp inside ``workdir``."""
    closed_proxy = "http://127.0.0.1:9"
    env = {
        "HOME": workdir,
        "TMPDIR": workdir,
        "TEMP": workdir,
        "TMP": workdir,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "HTTP_PROXY": closed_proxy,
        "HTTPS_PROXY": closed_proxy,
        "ALL_PROXY": closed_proxy,
        "http_proxy": closed_proxy,
        "https_proxy": closed_proxy,
        "all_proxy": closed_proxy,
        "NO_PROXY": "",
        "no_proxy": "",
    }
    if os.name == "nt":
        system_root = os.environ.get("SYSTEMROOT") or os.environ.get("WINDIR") or r"C:\Windows"
        env.update(
            {
                "SYSTEMROOT": system_root,
                "WINDIR": system_root,
                "PATH": str(Path(system_root) / "System32"),
                "USERPROFILE": workdir,
                "APPDATA": workdir,
                "LOCALAPPDATA": workdir,
            }
        )
        env.pop("LANG")
        env.pop("LC_ALL")
    else:
        env["PATH"] = os.defpath
    return env


def _kill(proc: subprocess.Popen[bytes]) -> None:
    """Kill the child and (POSIX) its whole session; Windows children die with the Job Object."""
    try:
        if sys.platform == "win32":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _tail(path: Path, chars: int = _STDERR_TAIL_CHARS) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - chars * 4))
            data = fh.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")[-chars:]


def run_code_tests(
    code: str | None,
    tests: CodeTests,
    limits: SandboxLimits | None = None,
    python: str | None = None,
) -> CodeRunResult:
    """Run ``code`` against ``tests`` in a sandboxed subprocess (see the module docstring).

    Args:
        code: Python source (e.g. from :func:`extract_code`). None/blank gives ``no_code``.
        tests: The problem's unit tests.
        limits: Timeout, memory and output limits; defaults to :class:`SandboxLimits()`.
        python: Interpreter path; defaults to :func:`default_python`.

    Returns:
        A :class:`CodeRunResult`. ``status`` is ``passed`` (all tests pass), ``failed`` (some
        test failed or raised), ``error`` (the code did not load or the process died),
        ``timeout`` (killed; tests that finished before the kill still count) or ``no_code``.
    """
    limits = limits or SandboxLimits()
    n_tests = tests.n_tests
    if code is None or not code.strip():
        return CodeRunResult("no_code", n_tests, 0, per_test=(False,) * n_tests)

    payload = {
        "code": code,
        "setup": tests.setup,
        "style": tests.style,
        "tests": list(tests.tests),
        "entry_point": tests.entry_point,
        "memory_mb": limits.memory_mb,
        "timeout_s": limits.timeout_s,
        "max_output_bytes": limits.max_output_bytes,
        "block_network": limits.block_network,
    }
    with tempfile.TemporaryDirectory(prefix="terser-sandbox-", ignore_cleanup_errors=True) as tmp:
        workdir = Path(tmp)
        harness = workdir / "_harness.py"
        payload_path = workdir / "_payload.json"
        result_path = workdir / "_result.jsonl"
        stdout_path = workdir / "_stdout.txt"
        stderr_path = workdir / "_stderr.txt"
        harness.write_text(_HARNESS, encoding="utf-8")
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        cmd = [
            python or default_python(),
            "-I",
            "-B",
            "-X",
            "utf8",
            str(harness),
            str(payload_path),
            str(result_path),
        ]
        popen_kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            popen_kwargs["start_new_session"] = True
        timed_out = False
        start = time.monotonic()
        with stdout_path.open("wb") as out_fh, stderr_path.open("wb") as err_fh:
            proc = subprocess.Popen(
                cmd,
                cwd=tmp,
                env=_scrubbed_env(tmp),
                stdin=subprocess.DEVNULL,
                stdout=out_fh,
                stderr=err_fh,
                **popen_kwargs,
            )
            try:
                returncode: int | None = proc.wait(timeout=limits.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill(proc)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                returncode = proc.returncode
        duration = time.monotonic() - start
        events = _read_events(result_path)
        stderr_tail = _tail(stderr_path)

    applied: tuple[str, ...] = ()
    per_test = [False] * n_tests
    loaded_error = False
    done = False
    errors: list[str] = []
    for event in events:
        kind = event.get("event")
        if kind == "start":
            applied = tuple(str(a) for a in event.get("applied", []))
        elif kind == "load_error":
            loaded_error = True
            errors.append(str(event.get("error", "")))
        elif kind == "test":
            index = event.get("index")
            if isinstance(index, int) and 0 <= index < n_tests:
                per_test[index] = bool(event.get("ok"))
            if not event.get("ok") and event.get("error"):
                errors.append(str(event["error"]))
        elif kind == "done":
            done = True
    if errors and errors[-1].strip() not in stderr_tail.replace("\r\n", "\n"):
        # Candidate output can crowd the real traceback out of the captured stderr.
        stderr_tail = (stderr_tail + "\n" + errors[-1])[-_STDERR_TAIL_CHARS:]
    n_passed = sum(per_test)
    status: RunStatus
    if timed_out:
        status = "timeout"
        stderr_tail = (stderr_tail + f"\n[terser sandbox] killed after {limits.timeout_s:g}s")[
            -_STDERR_TAIL_CHARS:
        ]
    elif loaded_error or not done:
        status = "error"
    else:
        status = "passed" if n_passed == n_tests else "failed"
    return CodeRunResult(
        status=status,
        n_tests=n_tests,
        n_passed=n_passed,
        returncode=returncode,
        stderr_tail=stderr_tail,
        duration_s=round(duration, 4),
        per_test=tuple(per_test),
        limits_applied=applied,
    )


def _read_events(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def run_output_tests(
    output: str,
    tests: CodeTests,
    language: str = "python",
    limits: SandboxLimits | None = None,
    python: str | None = None,
) -> CodeRunResult | None:
    """Extract code from a model ``output`` and run it; None if ``language`` is not executable.

    Only Python is executed. Other languages (SQL, Java...) return None so callers can fall back
    to the judge.
    """
    if (_canonical_language(language) or "python") != "python":
        return None
    code = extract_code(output, "python", tests.entry_point)
    return run_code_tests(code, tests, limits, python)
