# Security policy

## Reporting a vulnerability

If you believe you have found a security vulnerability in PTWM
— including data-corruption bugs that affect compressed output,
unchecked deserialization paths, or any issue where a maliciously
crafted `.ptwm` blob could cause unsafe behaviour in a consumer — please
report it privately.

**Contact:** kurt@computer.org

Please include:

* A description of the issue.
* A minimal reproducer (Python code, CLI command, or a malformed
  `.ptwm` blob) if you have one.
* The affected PTWM version (`ptwm --version` or the
  `__version__` attribute), and the platform.

We will acknowledge receipt within seven (7) days and aim to provide
an initial assessment within fourteen (14) days. We work under a
**90-day coordinated-disclosure window** by default; if a fix is
not ready before that point, we will discuss an extension with you
in good faith.

## Patent concerns

If you believe a PTWM component infringes an issued patent,
please use the same private contact above and include the patent
number(s) and the specific code paths or features you are concerned
about.

## Supported versions

Only the latest released minor version on `main` receives security
fixes. Older versions are out of scope.

## Scope

In scope:

* Code under `crates/ptwm-core/`, `crates/ptwm-py/`, and
  `python/ptwm/`.
* The `ptwm` CLI entry point.
* The official PyPI / crates.io artefacts.

Out of scope:

* Issues in third-party dependencies — please report those upstream.
  We will track and bump dependencies once an upstream fix lands.
* Denial-of-service from arbitrarily large inputs (PTWM does
  not bound input size by policy; consumers are expected to enforce
  their own limits).
