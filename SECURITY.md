# Security Policy

## Supported Versions

As an active research project, security updates are provided for the current development branch.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

## Reporting a Vulnerability

While this project primarily involves historical archival data, I take the integrity of the research environment seriously.

If you discover a security vulnerability or a significant data integrity issue (e.g., a critical error in the cleaning pipeline that compromises results), please **do not open a public issue**. 

Instead, please report it via email to: **anton.ebsen@gmail.com**.

I will acknowledge your report within 48 hours and work to provide a patch or a "Corrigendum" note in the research monograph as soon as possible.

## Data Integrity

The raw archival files in `exam_project2/data/raw/` are tracked via Git/DVC to ensure an immutable audit trail. Any suspected corruption of the source files should be reported immediately.
