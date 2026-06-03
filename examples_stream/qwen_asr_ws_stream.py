"""
基于 Qwen3-ASR 的实时流式语音识别 WebSocket 版本
通过 WebSocket 实现高并发支持
支持 PTT 模式

Install:
  pip install fastapi uvicorn qwen-asr[vllm]

Run:
  python qwen_asr_ws_stream.py
Open:
  http://127.0.0.1:21591
"""
import argparse
import os
import time
import asyncio
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from qwen_asr import Qwen3ASRModel
from dotenv import load_dotenv
from loguru import logger
import json

load_dotenv()
root_path = os.environ.get("model_root_dir", "/data/LLM/")
print(f"root_path: {root_path}")

# ─── 模型路径 ────────────────────────────────────────────────────
ASR_MODEL_PATH = root_path + "Qwen3-ASR-0.6B"

# ─── 音频参数 ────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS = 1

# ─── 流式参数 ────────────────────────────────────────────────────
UNFIXED_CHUNK_NUM = 4
UNFIXED_TOKEN_NUM = 5
CHUNK_SIZE_SEC = 0.6

# ─── 全局模型 ────────────────────────────────────────────────────
asr_model: Optional[Qwen3ASRModel] = None

# ─── 会话管理 ────────────────────────────────────────────────────
@dataclass
class Session:
    # ASR 相关
    asr_state: object              # Qwen ASR 流式状态
    sentence_text: str             # 当前句子累积文本
    sentence_start_time: float

    # 会话管理
    created_at: float
    last_seen: float

    # 模式支持
    mode: str = "ptt"              # "ptt" 模式
    session_text: str = ""         # 整个面试累积文本
    records: list = field(default_factory=list)  # 每轮录音记录 [{"text": ..., "duration": ...}]
    record_text: str = ""          # 当前录音累积文本
    record_start_time: float = 0.0 # 当前录音开始时间
    is_recording: bool = False     # PTT 模式是否正在录音

SESSIONS: Dict[str, Session] = {}
SESSION_TTL_SEC = 10 * 60

# 线程锁，保护 SESSIONS 的并发访问
sessions_lock = threading.Lock()

# ─── FastAPI 应用 ────────────────────────────────────────────────────
app = FastAPI(title="Qwen3-ASR 实时语音识别 WebSocket")

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
                if SESSIONS[sid].asr_state:
                    asr_model.finish_streaming_transcribe(SESSIONS[sid].asr_state)
            except Exception:
                pass
            SESSIONS.pop(sid, None)

def _create_session(session_id: str) -> Session:
    """创建新会话"""
    now = time.time()
    session = Session(
        asr_state=None,
        sentence_text="",
        sentence_start_time=0.0,
        created_at=now,
        last_seen=now,
    )
    with sessions_lock:
        SESSIONS[session_id] = session
    return session

def _remove_session(session_id: str):
    """删除会话"""
    with sessions_lock:
        s = SESSIONS.get(session_id)
        if s and s.asr_state:
            try:
                asr_model.finish_streaming_transcribe(s.asr_state)
            except Exception:
                pass
        SESSIONS.pop(session_id, None)

def _get_session(session_id: str) -> Optional[Session]:
    """获取会话"""
    _gc_sessions()
    with sessions_lock:
        s = SESSIONS.get(session_id)
        if s:
            s.last_seen = time.time()
        return s

# ─── ASR 处理函数 ─────────────────────────────────────────────────
def _init_asr_state(session: Session):
    """初始化 ASR 流式状态"""
    session.asr_state = asr_model.init_streaming_state(
        unfixed_chunk_num=UNFIXED_CHUNK_NUM,
        unfixed_token_num=UNFIXED_TOKEN_NUM,
        chunk_size_sec=CHUNK_SIZE_SEC,
    )
    session.sentence_text = ""
    session.sentence_start_time = time.time()

def _process_audio_chunk(wav: np.ndarray, session: Session) -> dict:
    """处理音频块，返回识别结果"""
    if not session.asr_state:
        return {"text": "", "is_sentence_end": False}

    try:
        asr_model.streaming_transcribe(wav, session.asr_state)
        text = getattr(session.asr_state, "text", "") or ""
        language = getattr(session.asr_state, "language", "") or ""
        return {
            "text": text,
            "language": language,
            "is_sentence_end": False
        }
    except Exception as e:
        logger.error(f"ASR 异常: {e}")
        return {"text": "", "is_sentence_end": False}

def _finish_asr(session: Session) -> dict:
    """结束 ASR，返回最终结果"""
    if not session.asr_state:
        return {"text": "", "duration": 0}

    try:
        asr_model.finish_streaming_transcribe(session.asr_state)
        text = getattr(session.asr_state, "text", "") or ""
        language = getattr(session.asr_state, "language", "") or ""
        duration = time.time() - session.sentence_start_time

        # 重置状态
        session.asr_state = None
        session.sentence_text = ""
        session.sentence_start_time = 0.0

        return {
            "text": text,
            "language": language,
            "duration": duration,
            "is_sentence_end": True
        }
    except Exception as e:
        logger.error(f"ASR 结束异常: {e}")
        return {"text": "", "duration": 0}

# ─── HTTP 路由 ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    """返回 HTML 页面"""
    html_path = os.path.join(os.path.dirname(__file__), "qwen_asr_ws_index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Qwen3-ASR WebSocket Service</h1><p>HTML file not found. WebSocket endpoint: /ws/asr</p>")

# ─── WebSocket 路由 ────────────────────────────────────────────────────
@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket):
    """WebSocket 流式 ASR 接口（支持 PTT 模式）"""
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

                    # PTT 模式：累积音频，处理识别
                    if session.mode == "ptt" and session.is_recording:
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, _process_audio_chunk, wav, session
                        )
                        await websocket.send_json({
                            "type": "partial",
                            "text": result["text"],
                            "language": result.get("language", "")
                        })

                elif "text" in data:
                    # 处理 JSON 控制命令
                    try:
                        msg = json.loads(data["text"])
                    except json.JSONDecodeError:
                        msg = {"type": data["text"]}

                    msg_type = msg.get("type")

                    # ─── 面试级别控制 ────────────────────────────────
                    if msg_type == "session_start":
                        session.session_text = ""
                        session.records = []
                        logger.info(f"[WebSocket] Session started: {session_id}")
                        await websocket.send_json({"type": "session_started"})

                    elif msg_type == "session_end":
                        # 返回完整记录
                        await websocket.send_json({
                            "type": "session_result",
                            "full_text": session.session_text,
                            "records": session.records
                        })
                        logger.info(f"[WebSocket] Session ended: {session_id}")

                    # ─── 模式选择 ────────────────────────────────
                    elif msg_type == "mode":
                        mode = msg.get("mode", "ptt")
                        if mode == "ptt":
                            session.mode = mode
                            logger.info(f"[WebSocket] Mode set to: {mode}")
                            await websocket.send_json({"type": "mode_set", "mode": mode})
                        else:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Qwen ASR only supports PTT mode, invalid mode: {mode}"
                            })

                    # ─── PTT 模式控制 ────────────────────────────────
                    elif msg_type == "record_start":
                        session.is_recording = True
                        session.record_text = ""
                        session.record_start_time = time.time()
                        # 初始化 ASR 流式状态
                        await asyncio.get_event_loop().run_in_executor(
                            None, _init_asr_state, session
                        )
                        logger.info(f"[WebSocket] PTT record started")
                        await websocket.send_json({"type": "record_started"})

                    elif msg_type == "record_end":
                        session.is_recording = False
                        # 强制结束 ASR，返回完整结果
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, _finish_asr, session
                        )
                        text = result["text"]
                        duration = result["duration"]
                        session.record_text = text
                        session.records.append({"text": text, "duration": duration})
                        session.session_text += text + "\n"
                        logger.info(f"[WebSocket] PTT record ended: {text}")
                        await websocket.send_json({
                            "type": "record_result",
                            "text": text,
                            "language": result.get("language", ""),
                            "duration": round(duration, 2)
                        })

            elif data["type"] == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        logger.info(f"[WebSocket] Client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"[WebSocket] Error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        # 清理 session
        if session_id:
            _remove_session(session_id)
            logger.info(f"[WebSocket] Session cleaned: {session_id}")

def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="Qwen3-ASR 实时语音识别 WebSocket (FastAPI)")
    p.add_argument("--asr-model-path", default=ASR_MODEL_PATH, help="Model name or local path")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=8888, help="Bind port")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="vLLM GPU memory utilization")
    p.add_argument("--gpu-id", type=str, default="7", help="GPU device ID(s) to use, e.g. '0' or '0,1'")
    p.add_argument("--unfixed-chunk-num", type=int, default=4)
    p.add_argument("--unfixed-token-num", type=int, default=5)
    p.add_argument("--chunk-size-sec", type=float, default=0.6)
    return p.parse_args()

def main():
    """主函数"""
    args = parse_args()

    # Set GPU device before loading model
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(f"Using GPU(s): {args.gpu_id}")

    # 更新全局参数
    global UNFIXED_CHUNK_NUM, UNFIXED_TOKEN_NUM, CHUNK_SIZE_SEC, ASR_MODEL_PATH
    UNFIXED_CHUNK_NUM = args.unfixed_chunk_num
    UNFIXED_TOKEN_NUM = args.unfixed_token_num
    CHUNK_SIZE_SEC = args.chunk_size_sec
    ASR_MODEL_PATH = args.asr_model_path

    # 加载 ASR 模型
    print("Loading Qwen3-ASR model...")
    global asr_model
    asr_model = Qwen3ASRModel.LLM(
        model=args.asr_model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=32,
    )
    print("Model loaded.\n")

    # 启动服务
    import uvicorn
    print("=" * 60)
    print("  Qwen3-ASR 实时语音识别 WebSocket 服务")
    print(f"  模型: {args.asr_model_path}")
    print(f"  访问: http://{args.host}:{args.port}")
    print(f"  WebSocket: ws://{args.host}:{args.port}/ws/asr")
    print("  按 Ctrl+C 停止")
    print("=" * 60 + "\n")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )

if __name__ == "__main__":
    main()