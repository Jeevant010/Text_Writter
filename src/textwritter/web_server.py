#!/usr/bin/env python3
"""Web Server for Text_Writter.

Exposes a FastAPI backend + interactive UI to generate handwriting,
supporting both:
1. Realistic Ruled Notebook ('Copy with Lines') Engine:
   - Instant (<0.1s)
   - Real notebook paper with blue ruling lines, red margin, header
   - Natural human writing wobble, ink bleed, and natural flow
2. Neural Engine (HWT on CPU, One-DM on GPU) for experimental few-shot photo learning.

Usage:
    python src/textwritter/web_server.py
    # Then open http://localhost:8000
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

# Add src/ to sys.path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

from textwritter.notebook_engine import render_notebook_copy, STYLES, INKS
from textwritter.pipeline import DEFAULT_TEXT, create, run_judge
from textwritter.runtime import OUT_DIR, ROOT, detect_environment, resolve_engine

STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_DIR = OUT_DIR / "_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Text_Writter Studio",
    description="Interactive web UI & API for few-shot handwriting generation and evaluation",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the single-page application UI."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_file)


@app.get("/api/info")
async def get_info():
    """Return runtime environment information."""
    env = detect_environment()
    engine = resolve_engine("auto")
    return {
        "status": "ready",
        "engine": engine,
        "device": env.device,
        "gpu": env.gpu_name,
        "environment": env.kind,
        "notebook_styles": list(STYLES.keys()),
        "ink_options": list(INKS.keys()),
    }


@app.get("/api/image/{filename}")
async def get_image(filename: str):
    """Serve generated image artifacts from out/."""
    safe_name = Path(filename).name
    file_path = OUT_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)


@app.post("/api/generate")
async def generate_handwriting(
    text: str = Form(DEFAULT_TEXT),
    engine: str = Form("notebook"),  # 'notebook' (lined copy) or 'hwt' / 'onedm' / 'auto'
    notebook_style: str = Form("kalam"),
    ink_color: str = Form("blue"),
    paper_type: str = Form("ruled"),  # 'ruled', 'blank', 'grid'
    judge: str = Form("false"),
    style: UploadFile | None = File(None),
):
    """Generate handwriting from text.
    
    Defaults to the instant, realistic ruled notebook ('copy with lines') engine.
    """
    try:
        timestamp = int(time.time())
        should_judge = str(judge).lower() in ("true", "1", "yes")

        # -------------------------------------------------------------
        # Mode 1: Realistic Ruled Notebook ('Copy with Lines') Engine
        # -------------------------------------------------------------
        if engine == "notebook":
            out_filename = f"notebook_{uuid.uuid4().hex[:8]}.png"
            out_path = OUT_DIR / out_filename

            result_info = render_notebook_copy(
                text=text,
                out_path=out_path,
                style_key=notebook_style,
                ink_name=ink_color,
                paper_type=paper_type,
                wobble=True,
            )

            verdict = {"judged": False, "reason": "judging disabled for instant render"}
            if should_judge:
                # Judge the generated output
                verdict = run_judge(out_path, text[:120])

            return {
                "success": True,
                "engine": "notebook",
                "mode": "Realistic Lined Copy",
                "text": text,
                "image_url": f"/api/image/{out_filename}?t={timestamp}",
                "compare_url": None,
                "lines_written": result_info.get("lines_written", 1),
                "paper": paper_type,
                "ink": ink_color,
                "style": result_info.get("style", notebook_style),
                "verdict": verdict,
            }

        # -------------------------------------------------------------
        # Mode 2: Neural Engine (HWT / One-DM)
        # -------------------------------------------------------------
        style_path = None
        if style and style.filename:
            ext = Path(style.filename).suffix or ".png"
            temp_filename = f"{uuid.uuid4().hex[:10]}_{Path(style.filename).stem}{ext}"
            saved_target = UPLOAD_DIR / temp_filename
            with open(saved_target, "wb") as buffer:
                shutil.copyfileobj(style.file, buffer)
            style_path = str(saved_target)

        result = create(
            text=text,
            style=style_path,
            engine=engine,
            judge=should_judge,
        )

        out_filename = Path(result["out"]).name
        image_url = f"/api/image/{out_filename}?t={timestamp}"

        compare_url = None
        if result.get("compare"):
            compare_filename = Path(result["compare"]).name
            compare_url = f"/api/image/{compare_filename}?t={timestamp}"

        return {
            "success": True,
            "engine": result.get("engine"),
            "mode": "Neural Model",
            "text": text,
            "image_url": image_url,
            "compare_url": compare_url,
            "verdict": result.get("verdict", {}),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Start the Text_Writter web server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on (default 8000)")
    args = parser.parse_args()

    print(f"\n==================================================")
    print(f"  ✍️  Text_Writter Studio Web Server Starting")
    print(f"  Access UI at: http://{args.host}:{args.port}")
    print(f"==================================================\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
