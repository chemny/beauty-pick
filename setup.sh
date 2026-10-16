#!/usr/bin/env bash
# ==============================================================================
# 颜选 · BeautyPick 一键安装与开箱即用初始化脚本
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "=========================================================="
echo "      🌸 颜选 · BeautyPick 一键安装与环境初始化"
echo "=========================================================="
echo "项目路径: $ROOT_DIR"

# 1. 检查 Python 运行环境
echo ""
echo "[1/5] 检查 Python 环境..."
PYTHON_BIN=""
for cmd in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        version=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_BIN="$cmd"
            echo "  ✓ 找到兼容的 Python: $cmd (v$version)"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "  ✗ 错误: 未找到 Python 3.10+。请先安装 Python 3.10 或更高版本。" >&2
    exit 1
fi

# 2. 创建或验证虚拟环境
echo ""
echo "[2/5] 准备 Python 虚拟环境 (.venv)..."
if [ ! -d ".venv" ]; then
    echo "  正在创建独立虚拟环境..."
    "$PYTHON_BIN" -m venv .venv
fi
VENV_PY="$ROOT_DIR/.venv/bin/python"

echo "  正在升级 pip 并安装核心依赖 (FastAPI, PyTorch, MediaPipe, OpenCV, C2PA)..."
"$VENV_PY" -m pip install --quiet --upgrade pip
"$VENV_PY" -m pip install --quiet -r requirements.txt
echo "  ✓ Python 依赖安装完成"

# 3. 检查模型与离线资料
echo ""
echo "[3/5] 检查视觉模型与离线凭据..."
"$VENV_PY" scripts/download_models.py

# 4. 检查 Web 前端静态资源
echo ""
echo "[4/5] 检查 Web 前端产物..."
if [ -f "web/dist/index.html" ]; then
    echo "  ✓ 预编译前端界面已就绪 (web/dist/index.html)"
else
    if command -v npm >/dev/null 2>&1; then
        echo "  未检测到预编译前端，正在通过 npm 构建..."
        npm run build --prefix web
        echo "  ✓ 前端构建完成"
    else
        echo "  ✗ 警告: 缺少 web/dist/index.html 且未安装 npm。请安装 Node.js 后执行 npm run build --prefix web"
    fi
fi

# 5. 自动注册 Agent Skill 插件
echo ""
echo "[5/5] 自动注册 Agent Skill 插件..."
AGENT_SKILL_TARGETS=(
    "$HOME/.gemini/config/skills/beauty-pick"
    "$HOME/.agents/skills/beauty-pick"
    "$HOME/.codex/skills/beauty-pick"
    "$HOME/.claude/skills/beauty-pick"
)

SKILL_TEMPLATE="$ROOT_DIR/SKILL.md"
REFS_DIR="$ROOT_DIR/references"

if [ -f "$SKILL_TEMPLATE" ]; then
    REGISTERED_COUNT=0
    for target in "${AGENT_SKILL_TARGETS[@]}"; do
        parent_dir="$(dirname "$target")"
        if [ -d "$parent_dir" ]; then
            mkdir -p "$target/references"
            # 动态替换绝对路径写入目标 SKILL.md
            sed "s|{{BEAUTYPICK_DIR}}|$ROOT_DIR|g" "$SKILL_TEMPLATE" > "$target/SKILL.md"
            if [ -d "$REFS_DIR" ]; then
                cp -R "$REFS_DIR/"* "$target/references/"
            fi
            echo "  ✓ 已同步注册到 Agent 目录: $target"
            REGISTERED_COUNT=$((REGISTERED_COUNT + 1))
        fi
    done

    # 默认保底注册到 ~/.gemini/config/skills/beauty-pick 与 ~/.agents/skills/beauty-pick
    if [ "$REGISTERED_COUNT" -eq 0 ]; then
        default_target="$HOME/.gemini/config/skills/beauty-pick"
        mkdir -p "$default_target/references"
        sed "s|{{BEAUTYPICK_DIR}}|$ROOT_DIR|g" "$SKILL_TEMPLATE" > "$default_target/SKILL.md"
        if [ -d "$REFS_DIR" ]; then
            cp -R "$REFS_DIR/"* "$default_target/references/"
        fi
        echo "  ✓ 已注册默认 Skill 到: $default_target"
    fi
fi

echo ""
echo "=========================================================="
echo "  🎉 恭喜！颜选 · BeautyPick 全部环境与插件初始化完毕！"
echo "=========================================================="
echo "▶ 启动服务: ./start.sh"
echo "▶ 访问界面: http://127.0.0.1:8765"
echo "▶ 命令行测试: .venv/bin/python -m app.cli status"
echo "▶ 在 AI Agent (Antigravity/Codex/Claude) 中使用: 直接提问 '帮我用颜选评估这张人像照片'"
echo "=========================================================="
