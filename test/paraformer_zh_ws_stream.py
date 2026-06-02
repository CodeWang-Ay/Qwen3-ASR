"""
基于 paraformer-zh-streaming + fsmn-vad 的实时流式中文语音识别 WebSocket 版本
通过 WebSocket 实现高并发支持

Install:
  pip install fastapi uvicorn funasr

Run:
  python paraformer_zh_ws_stream.py
Open:
  http://127.0.0.1:21590
"""
import argparse
import os
import time
import threading
import asyncio
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from funasr import AutoModel
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
root_path = os.environ.get("model_root_dir", "/data/LLM/")
print(f"root_path: {root_path}")

# ─── 模型路径 ────────────────────────────────────────────────────
ASR_MODEL_PATH = root_path + "paraformer-zh-streaming"
VAD_MODEL_PATH = root_path + "fsmn-vad"

# ─── 音频参数 ────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS = 1

VAD_CHUNK_MS = 200
VAD_CHUNK_SAMPLES = int(SAMPLE_RATE * VAD_CHUNK_MS / 1000)  # 3200

ASR_CHUNK_MS = 600
ASR_CHUNK_SAMPLES = int(SAMPLE_RATE * ASR_CHUNK_MS / 1000)  # 9600

CHUNK_SIZE_CFG = [0, 10, 5]
ENCODER_LOOK_BACK = 4
DECODER_LOOK_BACK = 1

# ─── 全局模型 ────────────────────────────────────────────────────
asr_model: Optional[AutoModel] = None
vad_model: Optional[AutoModel] = None

# ─── 会话管理 ────────────────────────────────────────────────────
@dataclass
class Session:
    vad_cache: dict
    is_speaking: bool
    silence_start: float
    asr_cache: dict
    asr_pending: list
    sentence_text: str
    sentence_start_time: float
    created_at: float
    last_seen: float


SESSIONS: Dict[str, Session] = {}
SESSION_TTL_SEC = 10 * 60

# 线程锁，保护 SESSIONS 的并发访问
sessions_lock = threading.Lock()


# ─── FastAPI 应用 ────────────────────────────────────────────────────
app = FastAPI(title="Paraformer 实时语音识别 WebSocket")

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
            SESSIONS.pop(sid, None)


def _create_session(session_id: str) -> Session:
    """创建新会话"""
    now = time.time()
    session = Session(
        vad_cache={},
        is_speaking=False,
        silence_start=0.0,
        asr_cache={},
        asr_pending=[],
        sentence_text="",
        sentence_start_time=0.0,
        created_at=now,
        last_seen=now
    )
    with sessions_lock:
        SESSIONS[session_id] = session
    return session


def _remove_session(session_id: str):
    """删除会话"""
    with sessions_lock:
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
def _feed_asr(chunk: np.ndarray, asr_cache: dict, is_final: bool) -> str:
    """送入 ASR，返回识别的文字片段"""
    try:
        result = asr_model.generate(
            input=chunk,
            cache=asr_cache,
            is_final=is_final,
            chunk_size=CHUNK_SIZE_CFG,
            encoder_chunk_look_back=ENCODER_LOOK_BACK,
            decoder_chunk_look_back=DECODER_LOOK_BACK,
            disable_pbar=True,
        )
        if result:
            return result[0].get("text", "").strip()
    except Exception as e:
        logger.error(f"ASR 异常: {e}")
    return ""


def _flush_pending(session: Session, is_final: bool) -> str:
    """处理待识别的音频，返回整句累积文字"""
    if is_final:
        if session.asr_pending:
            chunk = np.array(session.asr_pending, dtype=np.float32)
            session.asr_pending = []
            new_piece = _feed_asr(chunk, session.asr_cache, is_final=True)
        else:
            new_piece = _feed_asr(
                np.zeros(160, dtype=np.float32), session.asr_cache, is_final=True
            )
        if new_piece:
            session.sentence_text += new_piece
    else:
        while len(session.asr_pending) >= ASR_CHUNK_SAMPLES:
            chunk = np.array(
                session.asr_pending[:ASR_CHUNK_SAMPLES], dtype=np.float32
            )
            session.asr_pending = session.asr_pending[ASR_CHUNK_SAMPLES:]
            new_piece = _feed_asr(chunk, session.asr_cache, is_final=False)
            if new_piece:
                session.sentence_text += new_piece

    return session.sentence_text


def _end_sentence(session: Session) -> dict:
    """结束当前句子"""
    text = _flush_pending(session, is_final=True)
    duration = time.time() - session.sentence_start_time

    # 重置句子状态
    session.asr_cache = {}
    session.asr_pending = []
    session.sentence_text = ""
    session.is_speaking = False
    session.silence_start = 0.0
    session.sentence_start_time = 0.0

    return {
        "text": text,
        "duration": duration,
        "is_sentence_end": True
    }


def _process_audio_chunk(wav: np.ndarray, session: Session) -> dict:
    """处理音频块，返回识别结果"""
    # VAD 检测
    vad_speech_start = False
    vad_speech_end = False
    try:
        vad_res = vad_model.generate(
            input=wav,
            cache=session.vad_cache,
            is_final=False,
            chunk_size=VAD_CHUNK_MS,
            disable_pbar=True,
        )
        if vad_res and vad_res[0].get("value"):
            for seg in vad_res[0]["value"]:
                s_start, s_end = seg[0], seg[1]
                if s_start >= 0:
                    vad_speech_start = True
                if s_end >= 0:
                    vad_speech_end = True
    except Exception as ex:
        logger.error(f"VAD 异常: {ex}")

    # 语音开始
    if vad_speech_start and not session.is_speaking:
        session.is_speaking = True
        session.silence_start = 0.0
        session.sentence_start_time = time.time()
        session.sentence_text = ""
        logger.info(f"🎤 语音开始")

    if session.is_speaking:
        session.asr_pending.extend(wav.tolist())

        if vad_speech_end:
            # VAD 给出自然断句
            logger.info(f"⏹️ 语音结束")
            result = _end_sentence(session)
            return result
        else:
            # 流式识别
            text = _flush_pending(session, is_final=False)
            return {
                "text": text,
                "is_sentence_end": False
            }

    return {"text": "", "is_sentence_end": False}


# ─── HTTP 路由 ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    """返回 HTML 页面"""
    html_path = os.path.join(os.path.dirname(__file__), "paraformer_zh_ws_index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


# ─── WebSocket 路由 ────────────────────────────────────────────────────
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

                    # 处理音频（在单独线程中执行，避免阻塞事件循环）
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, _process_audio_chunk, wav, session
                    )

                    # 返回识别结果
                    await websocket.send_json({
                        "type": "result",
                        "text": result["text"],
                        "is_sentence_end": result["is_sentence_end"]
                    })

                elif "text" in data:
                    # 处理文本消息（控制命令）
                    msg = data["text"]
                    if msg == "finish":
                        # 结束识别
                        if session.is_speaking:
                            result = _end_sentence(session)
                        else:
                            result = {"text": "", "is_sentence_end": False}

                        await websocket.send_json({
                            "type": "final_result",
                            "text": result["text"],
                            "is_sentence_end": result["is_sentence_end"]
                        })
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
    p = argparse.ArgumentParser(description="Paraformer 实时语音识别 WebSocket (FastAPI)")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=21590, help="Bind port")
    return p.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 加载 ASR 模型
    print("⏳ 加载 ASR 模型...")
    global asr_model
    asr_model = AutoModel(
        model=ASR_MODEL_PATH,
        model_revision="v2.0.4",
        disable_update=True,
    )
    print("✅ ASR 模型加载完成")

    # 加载 VAD 模型
    print("⏳ 加载 VAD 模型...")
    global vad_model
    vad_model = AutoModel(
        model=VAD_MODEL_PATH,
        model_revision="v2.0.4",
        disable_update=True,
    )
    print("✅ VAD 模型加载完成\n")

    # 启动服务
    import uvicorn
    print("=" * 60)
    print("  Paraformer 实时语音识别 WebSocket 服务")
    print("  模型: paraformer-zh-streaming + fsmn-vad")
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