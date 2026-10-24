# BeautyPick

> Local FPEM facial scoring, TOPIQ-IAA photography scoring, and parameter-assisted Qwen body scoring. See the [installation guide](docs/runtime-install.md).


English | [中文](./README.zh.md)

## Overview

BeautyPick is a local portrait assessment and photo-selection application with an Agent Skill entry point. It provides model-based facial scores and local body assessments and TOPIQ-IAA photography scores. Scores reflect model preferences and heuristics, not objective beauty or validated anatomical judgments. Normal scoring runs locally without a Codex account.

Release version: **0.3.0**. See [release notes](./CHANGELOG.md).

## Preview

![BeautyPick Portrait Assessment and Library Dashboard](./assets/preview.png)

## Features

| Capability | What it helps you do |
|---|---|
| Facial Aesthetics Assessment | Uses a fixed standard face crop and local FPEM to predict an overall score; does not invent independent facial-part scores. |
| Visible Body Assessment | Local Qwen uses MediaPipe/FASHN candidate parameters for chest, waist, hips and legs; fixed weights with zero contribution for missing observations. |
| Photographic Aesthetics | Local TOPIQ-IAA predicts an overall score without fabricated independent facet scores. |
| Provenance Evidence | Inspects C2PA, EXIF and file markings; missing evidence does not establish origin. |
| Photo Curation & Ranking | Compares and ranks multiple portrait candidates across individual metric dimensions without duplicate model calls. |

## Installation

### Requirements

Prepare the backend and local runtime below before requesting installation.

| Item | Requirement |
|---|---|
| Full backend | Apple Silicon Mac, Python 3.13 and Ollama with support for the specified MLX model |
| Windows client | OpenSSH tunnel to your own Mac backend; the MLX body engine does not run natively on Windows |
| Model assets | FPEM, TOPIQ-IAA, FASHN, MediaPipe and the exact Qwen version; about 4.3 GB plus dependencies |
| Agent | Local CLI/API access; no Codex account required; an explicit skill installation directory |

### Quick Installation

Ask your Agent to install the application and register only the intended Skill directory:

```text
Please install BeautyPick from https://github.com/chemny/beauty-pick, check the runtime requirements, and register the skill only in the Agent directory I specify.
```

An existing skill directory is never overwritten. See the [installation and Windows connection guide](docs/runtime-install.md). The installer does not synchronize other Agents or silently change model versions.

## Quick Start

Send a conversational request to your Agent to evaluate a portrait photo:

```text
Help me evaluate this portrait using beauty-pick to analyze facial features, body proportions, and photographic quality.
```

To run the standalone Web interface, execute `./start.sh` and visit `http://127.0.0.1:8765`.

## Usage Examples

### Comparing and Selecting the Best Photo

```text
Compare the portrait photos in my album using beauty-pick and rank them based on photographic sharpness and composition.
```

### Checking Provenance and AI Generation Signs

```text
Inspect the C2PA credentials and EXIF metadata of this image to verify whether it was AI-synthesized or shot on a camera.
```

### Getting Practical Posing and Shooting Advice

```text
Analyze the pose and lighting of this photo with beauty-pick and provide shooting tips to improve the head room and shoulder balance.
```

## How It Works

BeautyPick uses local scoring with a separate Agent interface:

1. **Local Scoring**: FPEM predicts overall facial attractiveness; TOPIQ-IAA predicts photographic aesthetics. Scores reflect model and project preferences, not objective beauty.
2. **Body Parameters and Scores**: MediaPipe and FASHN extract landmarks, masks and candidate contours for local Qwen. Chest/waist/hips/legs use fixed weights of 25/30/25/20%; missing observations remain null and contribute zero to the total.
3. **Application & Task Queue Layer**: FastAPI server orchestrating background evaluation queues, caching, and SQLite metadata storage (`data/portrait.db`).
4. **Agent Skill Contract Layer**: A standard `SKILL.md` entry point defining CLI invocations and Pydantic schema validation.

## Repository Structure

```text
beautypick/
├── SKILL.md                  # Root Agent Skill entry point and operational specification
├── references/               # Architectural reference and REST API contracts
├── app/                      # FastAPI backend service, scoring algorithms, and CLI
│   ├── scoring.py            # Dimension weights, scoring schemas, and rubrics
│   ├── provider.py           # Scoring orchestration
│   ├── beauty_model.py       # Legacy ResNet50 rollback adapter
│   ├── photo_model.py        # Legacy CV photography adapter
│   └── pose_model.py         # Skeletal landmark detection and pose decoupling
├── web/                      # React 19 + TypeScript + Vite frontend
│   └── dist/                 # Prebuilt static web assets for zero-dependency launch
├── models/                   # Local vision models, landmark tasks, and C2PA trust anchors
├── scripts/                  # Model verification and helper tools
├── assets/                   # Public preview illustrations and screenshots
├── setup.sh                  # Explicit local installation; single-target skill registration
├── start.sh                  # Application launch script
└── requirements.txt          # Python dependency specifications
```

## License

This project is licensed under the [MIT License](LICENSE).

Third-party sources and weights retain their own terms, including noncommercial restrictions for TOPIQ. See [third-party notices](THIRD_PARTY.md); the root MIT license does not override them.

## About Me

Maintained by [@chemny](https://github.com/chemny). Designed for creators, photographers, and developers seeking local portrait assessment and photo-selection tools through AI agents. Feedback and contributions are welcome via GitHub Issues.
