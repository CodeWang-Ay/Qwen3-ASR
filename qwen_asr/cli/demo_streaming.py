# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Minimal web demo for Qwen3ASRModel Streaming Inference (vLLM backend).

Install:
  pip install qwen-asr[vllm]

Run:
  python streaming/demo_qwen3_asr_vllm_streaming.py
Open:
  http://127.0.0.1:7860
"""
import argparse
import os
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from flask import Flask, Response, jsonify, request
from qwen_asr import Qwen3ASRModel


@dataclass
class Session:
    state: object
    created_at: float
    last_seen: float


app = Flask(__name__)


# CORS middleware: add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

global asr
global UNFIXED_CHUNK_NUM
global UNFIXED_TOKEN_NUM
global CHUNK_SIZE_SEC

SESSIONS: Dict[str, Session] = {}
SESSION_TTL_SEC = 10 * 60


def _gc_sessions():
    now = time.time()
    dead = [sid for sid, s in SESSIONS.items() if now - s.last_seen > SESSION_TTL_SEC]
    for sid in dead:
        try:
            asr.finish_streaming_transcribe(SESSIONS[sid].state)
        except Exception:
            pass
        SESSIONS.pop(sid, None)


def _get_session(session_id: str) -> Optional[Session]:
    _gc_sessions()
    s = SESSIONS.get(session_id)
    if s:
        s.last_seen = time.time()
    return s



@app.get("/")
def index():
    html_path = os.path.join(os.path.dirname(__file__), "index_html.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return Response(html_content, mimetype="text/html; charset=utf-8")


# 配置跨域访问
@app.route("/api/<path:path>", methods=["OPTIONS"])
@app.route("/api", methods=["OPTIONS"])
def handle_options(path=None):
    """Handle CORS preflight requests."""
    response = Response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# 
from loguru import logger
@app.post("/api/start")
def api_start():
    session_id = uuid.uuid4().hex
    logger.info(f"session_id: {session_id}")
    state = asr.init_streaming_state(
        unfixed_chunk_num=UNFIXED_CHUNK_NUM,
        unfixed_token_num=UNFIXED_TOKEN_NUM,
        chunk_size_sec=CHUNK_SIZE_SEC,
    )
    now = time.time()
    SESSIONS[session_id] = Session(state=state, created_at=now, last_seen=now)
    return jsonify({"session_id": session_id})


@app.post("/api/chunk")
def api_chunk():
    session_id = request.args.get("session_id", "")
    s = _get_session(session_id)
    if not s:
        return jsonify({"error": "invalid session_id"}), 400

    if request.mimetype != "application/octet-stream":
        return jsonify({"error": "expect application/octet-stream"}), 400

    raw = request.get_data(cache=False)
    if len(raw) % 4 != 0:
        return jsonify({"error": "float32 bytes length not multiple of 4"}), 400

    wav = np.frombuffer(raw, dtype=np.float32).reshape(-1)

    asr.streaming_transcribe(wav, s.state)

    return jsonify(
        {
            "language": getattr(s.state, "language", "") or "",
            "text": getattr(s.state, "text", "") or "",
        }
    )


@app.post("/api/finish")
def api_finish():
    session_id = request.args.get("session_id", "")
    s = _get_session(session_id)
    if not s:
        return jsonify({"error": "invalid session_id"}), 400

    asr.finish_streaming_transcribe(s.state)
    out = {
        "language": getattr(s.state, "language", "") or "",
        "text": getattr(s.state, "text", "") or "",
    }
    SESSIONS.pop(session_id, None)
    return jsonify(out)


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-ASR Streaming Web Demo (vLLM backend)")
    # p.add_argument("--asr-model-path", default="Qwen/Qwen3-ASR-1.7B", help="Model name or local path")
    p.add_argument("--asr-model-path", default="/data/LLM/Qwen3-ASR-0.6B", help="Model name or local path")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=8888, help="Bind port")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="vLLM GPU memory utilization")
    p.add_argument("--gpu-id", type=str, default="7", help="GPU device ID(s) to use, e.g. '0' or '0,1'")

    p.add_argument("--unfixed-chunk-num", type=int, default=4)
    p.add_argument("--unfixed-token-num", type=int, default=5)
    p.add_argument("--chunk-size-sec", type=float, default=0.6)
    return p.parse_args()


def main():
    args = parse_args()

    # Set GPU device before loading model
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(f"Using GPU(s): {args.gpu_id}")

    global asr
    global UNFIXED_CHUNK_NUM
    global UNFIXED_TOKEN_NUM
    global CHUNK_SIZE_SEC

    UNFIXED_CHUNK_NUM = args.unfixed_chunk_num
    UNFIXED_TOKEN_NUM = args.unfixed_token_num
    CHUNK_SIZE_SEC = args.chunk_size_sec

    asr = Qwen3ASRModel.LLM(
        model=args.asr_model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=32,
    )
    print("Model loaded.")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()