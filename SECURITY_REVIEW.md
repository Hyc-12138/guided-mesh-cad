# Public content review

Review date: 2026-10-18. Release: 0.3.0.dev17+public.1.

Scope:

- All 34 tracked files from the previously published main commit.
- All 48 distinct historical file blobs and email metadata in its 13 commits.
- Every file in this release source tree, source ZIP, and wheel, including archive member names, package metadata, license files and wheel RECORD.

Checks covered local absolute paths, known private identifiers, personal contact details, credentials, internal integration references, external fixture dependencies, unexpected binaries, and archive metadata. Matches in source code were reviewed; Python matrix multiplication operators were distinguished from email strings. Generic tests and examples construct their inputs locally.

No sensitive residuals were identified within this scope. Historical commit email addresses use GitHub noreply identities. LICENSE is the unmodified Apache 2.0 text; NOTICE uses project attribution.

This was a maintainer self-review with automated per-file scans and review of relevant source changes, not an independent security audit or a guarantee against every possible secret. Runtime outputs and user-provided model data are outside this publication review. Full internal evidence is retained locally; it is not included in release artifacts.
