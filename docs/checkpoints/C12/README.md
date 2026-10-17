# C12 evidence

[test_access_log.jsonl](test_access_log.jsonl) is the row 12.1 access log of the frozen test
split, written only by `terser.phase12.unlock` (one JSON line per unlock or read).

Two caveats when reading it against the git history:

- **Timestamps vs commit dates.** Each entry's `when` / `when_ist` is the wall-clock time of the
  access. The commits that carry the log are dated by this repository's commit-date convention,
  not the wall clock, so some commits (e.g. `e8bf7bd`, `9be54b6`, `55c6f99`) show dates earlier
  than the entries they contain. History is not rewritten; the log's own timestamps are the
  authoritative order of accesses.
- **`git_dirty` before the fix.** Until `git_state` learned to exclude the log file itself,
  appending one entry made the tree dirty for the next entry in the same run (line 12 `false`,
  line 13 `true` 0.1 s later). `git_dirty` on the existing entries therefore over-reports; the
  `git_sha` is unaffected.
