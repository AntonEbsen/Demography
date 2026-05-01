# Site Layouts

This directory contains the global templates and design systems for the research monograph.

## Components

- `Layout.astro`: The primary wrapper for all pages. It manages:
  - **Global CSS**: The "Imperial Dark" design tokens (Gold/Crimson/Slate).
  - **SEO**: JSON-LD metadata for academic discovery.
  - **PWA**: Service worker registration and manifest links for offline reading.
  - **Global Scripts**: Search indexing, language toggles, and Mermaid.js initialization.

## Design Philosophy

The layout is inspired by high-end academic journals and 19th-century typography, modernized for high-density data exploration on any device.
