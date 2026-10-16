# BeautyPick

English | [中文](./README.zh.md)

## Overview

BeautyPick is an objective, multi-dimensional portrait aesthetic assessment and curation system. It integrates computer vision, deep-learning bone structure baselines, and optical physical metrics to deliver continuous 0.1-decimal ratings for facial aesthetics, body proportions, and photographic quality. Built for photographers, digital artists, creators, and developers, it transforms subjective portrait selection into a grounded, reproducible evaluation workflow. It is available both as a standalone local web application and as an out-of-the-box Agent Skill across Google Antigravity, OpenAI Codex, and Claude Code.

## Preview

![BeautyPick Portrait Assessment and Library Dashboard](./assets/preview.png)

## Features

| Capability | What it helps you do |
|---|---|
| Facial Aesthetics Assessment | Computes ResNet50 bone-structure baseline scores, golden facial proportions, catchlights, and micro-skin texture while penalizing plastic over-smoothing. |
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

BeautyPick operates through a modular hybrid architecture combining local real-time measurement with agentic reasoning:

1. **Local Objective Physics Engine (`native_apple_mps_resnet50_mediapipe_cv`)**: Runs on hardware-accelerated local frameworks (Apple Silicon MPS / OpenCV / MediaPipe). Measures facial landmarks, bone proportions, Tenengrad sharpness gradients, dynamic range, and subject-to-background blur variance ratios (`dof_separation`).
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

Maintained by [@chemny](https://github.com/chemny). Designed for creators, photographers, and developers seeking scientific, objective, and multi-dimensional portrait quality assessment through AI agents. Feedback and contributions are welcome via GitHub Issues.
