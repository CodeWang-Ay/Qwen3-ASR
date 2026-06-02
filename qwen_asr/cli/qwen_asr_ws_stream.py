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
Qwen3-ASR Streaming Web Demo with WebSocket (FastAPI backend).

Install:
  pip install qwen-asr[vllm]
  pip install fastapi uvicorn

Run:
  python demo_streaming.py
Open:
  http://127.0.0.1:8888
"""
import argparse
import asyncio
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from qwen_asr import Qwen3ASRModel
from loguru import logger


@dataclass
class Session:
    state: object
    created_at: float
    last_seen: float


# 全局配置
asr: Optional[Qwen3ASRModel] = None
UNFIXED_CHUNK_NUM: int = 4
UNFIXED_TOKEN_NUM: int = 5
CHUNK_SIZE_SEC: float = 0.6

SESSIONS: Dict[str, Session] = {}
SESSION_TTL_SEC = 10 * 60

# 线程锁，保护 SESSIONS 的并发访问
sessions_lock = threading.Lock()


# FastAPI 应用
app = FastAPI(title="Qwen3-ASR Streaming WebSocket")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _gc_sessions():
    """清理过期会话"""
    now = time.time()
    with sessions_lock:
        dead = [sid for sid, s in SESSIONS.items() if now - s.last_seen > SESSION_TTL_SEC]
        for sid in dead:
            try:
                if asr:
                    asr.finish_streaming_transcribe(SESSIONS[sid].state)
            except Exception:
                pass
            SESSIONS.pop(sid, None)


def _get_session(session_id: str) -> Optional[Session]:
    """获取会话"""
    _gc_sessions()
    with sessions_lock:
        s = SESSIONS.get(session_id)
        if s:
            s.last_seen = time.time()
        return s


def _create_session(session_id: str) -> Session:
    """创建新会话"""
    state = asr.init_streaming_state(
        unfixed_chunk_num=UNFIXED_CHUNK_NUM,
        unfixed_token_num=UNFIXED_TOKEN_NUM,
        chunk_size_sec=CHUNK_SIZE_SEC,
    )
    now = time.time()
    session = Session(state=state, created_at=now, last_seen=now)
    with sessions_lock:
        SESSIONS[session_id] = session
    return session


def _remove_session(session_id: str):
    """删除会话"""
    with sessions_lock:
        s = SESSIONS.get(session_id)
        if s:
            try:
                asr.finish_streaming_transcribe(s.state)
            except Exception:
                pass
            SESSIONS.pop(session_id, None)


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回 HTML 页面"""
    html_path = os.path.join(os.path.dirname(__file__), "qwen_asr_ws_index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket):
    """WebSocket 流式 ASR 接口"""
    await websocket.accept()
    session_id = None

    try:
        # 连接建立，创建 session
        session_id = f"ws_{time.time_ns()}"
        session = _create_session(session_id)
        logger.info(f"[WebSocket] Client connected: {session_id}")

        # 发送连接成功消息
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id
        })

        while True:
            # 接收消息
            data = await websocket.receive()

            if data["type"] == "websocket.receive":
                if "bytes" in data:
                    # 处理音频数据
                    raw = data["bytes"]
                    if len(raw) % 4 != 0:
                        await websocket.send_json({
                            "type": "error",
                            "message": "float32 bytes length not multiple of 4"
                        })
                        continue

                    wav = np.frombuffer(raw, dtype=np.float32).reshape(-1)

                    # ASR 流式识别（在单独的线程中执行，避免阻塞）
                    asr.streaming_transcribe(wav, session.state)

                    # 返回识别结果
                    await websocket.send_json({
                        "type": "result",
                        "language": getattr(session.state, "language", "") or "",
                        "text": getattr(session.state, "text", "") or "",
                    })

                elif "text" in data:
                    # 处理文本消息（控制命令）
                    msg = data["text"]
                    if msg == "finish":
                        # 结束识别
                        asr.finish_streaming_transcribe(session.state)
                        result = {
                            "type": "final_result",
                            "language": getattr(session.state, "language", "") or "",
                            "text": getattr(session.state, "text", "") or "",
                        }
                        await websocket.send_json(result)
                        break

            elif data["type"] == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        logger.info(f"[WebSocket] Client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"[WebSocket] Error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        # 清理 session
        if session_id:
            _remove_session(session_id)
            logger.info(f"[WebSocket] Session cleaned: {session_id}")


def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="Qwen3-ASR Streaming WebSocket (FastAPI)")
    p.add_argument("--asr-model-path", default="/data/LLM/Qwen3-ASR-0.6B", help="Model path")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=8888, help="Bind port")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="GPU memory utilization")
    p.add_argument("--gpu-id", type=str, default="7", help="GPU device ID")
    p.add_argument("--unfixed-chunk-num", type=int, default=4)
    p.add_argument("--unfixed-token-num", type=int, default=5)
    p.add_argument("--chunk-size-sec", type=float, default=0.6)
    return p.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(f"Using GPU(s): {args.gpu_id}")

    global asr, UNFIXED_CHUNK_NUM, UNFIXED_TOKEN_NUM, CHUNK_SIZE_SEC

    UNFIXED_CHUNK_NUM = args.unfixed_chunk_num
    UNFIXED_TOKEN_NUM = args.unfixed_token_num
    CHUNK_SIZE_SEC = args.chunk_size_sec

    # 加载模型
    asr = Qwen3ASRModel.LLM(
        model=args.asr_model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=32,
    )
    print("Model loaded.")

    # 启动服务
    import uvicorn
    print(f"WebSocket server starting on http://{args.host}:{args.port}")
    print(f"WebSocket endpoint: ws://{args.host}:{args.port}/ws/asr")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()