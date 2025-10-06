# Product Requirements Document: Netball Analysis System (Executive Summary)

This document is a brief, stakeholder-friendly summary of the Netball Analysis System.
For the full, development-ready specification, please refer to the definitive PRD:

- Full PRD (Development-Ready): docs/prd_enhanced.md

## Overview
- AI-powered netball analysis compliant with MSSS 2025 rules
- YOLOv8 detection, DeepSORT/BYTETrack tracking, FastAPI backend, Streamlit frontend
- Focus on player/ball/hoop detection, tracking, possession, shooting, standings

## Objectives (Highlights)
- ≥80% mAP@50 player detection, ≥85% recall ball detection
- ≥2x real-time processing; ≤3 ID switches per minute
- MSSS compliance: zones, 3-second rule, standings tie-breakers

## Scope (Highlights)
- Detection & tracking, homography, zones, possession, shooting, standings
- Web UI (Streamlit) + API (FastAPI), CSV/JSON exports

## Acceptance (Highlights)
- Detect/track 14 players; enforce 3-second rule; correct standings

For detailed user stories, architecture, sprint plan, tasks, Definition of Done, and testing strategy, see `docs/prd_enhanced.md`.
