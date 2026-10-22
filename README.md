# BeautyPick

English | [中文](./README.zh.md)

## Overview

BeautyPick is a local portrait assessment and photo-selection application with an Agent Skill entry point. It provides model-based facial scores and measurement-based body and photography assessments. Scores reflect model preferences and heuristics, not objective beauty or validated anatomical judgments. Normal scoring runs locally without a Codex account.

Current release: **0.2.0**. See [release notes](./CHANGELOG.md).

## Preview

![BeautyPick Portrait Assessment and Library Dashboard](./assets/preview.png)

## Features

| Capability | What it helps you do |
|---|---|
| Facial Aesthetics Assessment | Uses a fixed standard face crop and ResNet50 to predict an overall score; does not invent independent facial-part scores. |
| Body Line & Pose Evaluation | Evaluates shoulder, waist, hip, and leg proportions while decoupling posing tilt from physique ratings and dynamically normalizing visible parts. |
| Photographic Quality Scoring | Delivers continuous 0.1-decimal scores based on Tenengrad edge gradients, facial exposure, and depth-of-field separation to clearly differentiate photos. |
| Provenance & Authenticity | Inspects C2PA digital credentials and EXIF metadata to distinguish camera shots, AI generations, and edits. |
| Photo Curation & Ranking | Compares and ranks multiple portrait candidates across individual metric dimensions without duplicate model calls. |

## Installation

### Requirements

Before using BeautyPick, ensure your environment meets the following baseline requirements:

| Item | Requirement |
|---|---|
| Agent Client | Google Antigravity, OpenAI Codex, or Claude Code |
| Python Runtime | Python 3.10 or higher |
| Operating System | macOS (Apple Silicon MPS recommended), Linux, or Windows |

### Quick Installation

Ask your Agent client to install BeautyPick:

```text
Please install the BeautyPick skill from https://github.com/chemny/beauty-pick
```

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

1. **Local Scoring**: Face evaluation uses landmark-based square crops with a 12% margin, no rotation and 224×224 input. Existing ResNet50 weights produce one overall score using the legacy display scale. Body and photography use local measurements. Model scores are subjective references; 99-point saturation is a known limitation.
2. **Body Pose & Physique Decoupling Engine**: Identifies BlazePose 33 skeletal landmarks. Posing actions (e.g. raised shoulders or body tilt) are isolated as creative styling and excluded from body scoring.
3. **Application & Task Queue Layer**: FastAPI server orchestrating background evaluation queues, caching, and SQLite metadata storage (`data/portrait.db`).
4. **Agent Skill Contract Layer**: A standard `SKILL.md` entry point defining CLI invocations and Pydantic schema validation.

## Repository Structure

```text
beautypick/
├── SKILL.md                  # Root Agent Skill entry point and operational specification
├── references/               # Architectural reference and REST API contracts
├── app/                      # FastAPI backend service, scoring algorithms, and CLI
│   ├── scoring.py            # Dimension weights, scoring schemas, and rubrics
│   ├── provider.py           # Objective engine orchestration and scoring formulas
│   ├── beauty_model.py       # ResNet50 facial baseline and geometry evaluation
│   ├── photo_model.py        # Tenengrad sharpness, exposure, and composition metrics
│   └── pose_model.py         # Skeletal landmark detection and pose decoupling
├── web/                      # React 19 + TypeScript + Vite frontend
│   └── dist/                 # Prebuilt static web assets for zero-dependency launch
├── models/                   # Local vision models, landmark tasks, and C2PA trust anchors
├── scripts/                  # Model verification and helper tools
├── assets/                   # Public preview illustrations and screenshots
├── setup.sh                  # One-click environment installation and skill registration
├── start.sh                  # Application launch script
└── requirements.txt          # Python dependency specifications
```

## License

This project is licensed under the [MIT License](LICENSE).

Third-party models located in `models/` follow their respective Google MediaPipe and Content Authenticity Initiative licenses.

## About Me

Maintained by [@chemny](https://github.com/chemny). Designed for creators, photographers, and developers seeking local portrait assessment and photo-selection tools through AI agents. Feedback and contributions are welcome via GitHub Issues.
