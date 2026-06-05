# Security Policy

## Supported Versions

Universe Selector is alpha-stage software. Security fixes are applied to the
latest released version only.

## Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub issues.

Instead, use GitHub's private vulnerability reporting on this repository: open
the **Security** tab and choose **Report a vulnerability**. (Maintainers: enable
this under Settings → Code security → Private vulnerability reporting.)

Please include:

- a description of the issue and its impact,
- steps to reproduce or a proof of concept,
- the affected version or commit.

You can expect an initial response within a reasonable time. Please give us a
chance to address the issue before any public disclosure.

## Scope

This tool fetches data from third-party providers and runs locally against a
local DuckDB database. Reports of issues such as code execution, path traversal,
or unsafe handling of provider data are in scope. The data quality or accuracy
of third-party market data is out of scope; see the Disclaimer and Data Sources
sections in the [README](README.md).
