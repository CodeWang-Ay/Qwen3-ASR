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
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from funasr import AutoModel
from dotenv import load_dotenv
from loguru import logger
import json

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
    # VAD 相关
    vad_cache: dict
    is_speaking: bool
    silence_start: float

    # ASR 相关
    asr_cache: dict
    asr_pending: list
    sentence_text: str
    sentence_start_time: float

    # 会话管理
    created_at: float
    last_seen: float

    # 新增：双模式支持
    mode: str = "ptt"              # "ptt" 或 "vad"
    session_text: str = ""         # 整个面试累积文本
    records: list = field(default_factory=list)  # 每轮录音记录 [{"text": ..., "duration": ...}]
    record_text: str = ""          # 当前录音累积文本
    record_start_time: float = 0.0 # 当前录音开始时间
    vad_silence_timeout: float = 3.0  # VAD 静音超时时间（秒）
    is_vad_active: bool = False    # VAD 模式是否激活
    is_recording: bool = False     # PTT 模式是否正在录音

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
        last_seen=now,
        # 新增字段默认值已在 dataclass 中设置
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

def _process_audio_chunk_vad(wav: np.ndarray, session: Session) -> dict:
    """VAD 模式：处理音频块，检测静音超时"""
    # VAD 检测语音边界
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

    # ─── 语音开始（用户又开始说话）────────────────────────────────
    if vad_speech_start:
        # 如果正在静音计时，用户又开始说话了，重置静音计时
        if session.silence_start > 0:
            session.silence_start = 0.0
            logger.info(f"🎤 VAD: 静音中断，用户继续说话")
            # 继续处理当前音频块（累积音频）

        # 如果之前没有在说话，标记为开始说话
        if not session.is_speaking:
            session.is_speaking = True
            session.sentence_start_time = time.time()
            session.sentence_text = ""
            logger.info(f"🎤 VAD: 语音开始")
            # ✅ 累积当前音频块（包含前几个音）
            session.asr_pending.extend(wav.tolist())
            return {
                "text": "",
                "is_sentence_end": False,
                "vad_speech_start_detected": True
            }

    # ─── 正在说话 ────────────────────────────────────────────────
    if session.is_speaking:
        # 累积音频
        session.asr_pending.extend(wav.tolist())

        # ─── 正在静音计时中 ──────────────────────────────────────
        if session.silence_start > 0:
            # 继续累积静音时间，不重置
            silence_duration = time.time() - session.silence_start
            if silence_duration >= session.vad_silence_timeout:
                # 静音超时，返回完整结果
                result = _end_sentence(session)
                logger.info(f"⏹️ VAD: 静音超时 ({silence_duration:.1f}s)，识别结果: {result['text']}")
                return {
                    "text": result["text"],
                    "duration": result["duration"],
                    "is_sentence_end": True,
                    "is_vad_timeout": True
                }
            else:
                # 静音中但未超时，返回当前累积文本
                text = session.sentence_text
                return {
                    "text": text,
                    "is_sentence_end": False,
                    "silence_duration": silence_duration
                }

        # ─── 检测到语音结束信号 ───────────────────────────────────
        if vad_speech_end:
            # 计算说话时长，保护短暂噪音
            speech_duration = time.time() - session.sentence_start_time
            min_speech_duration = 0.5  # 最少说话 0.5 秒才能结束

            if speech_duration < min_speech_duration:
                # 说话时间太短，可能是噪音，忽略 vad_speech_end
                logger.info(f"⚠️ VAD: 说话时长过短 ({speech_duration:.2f}s)，忽略结束信号")
                text = _flush_pending(session, is_final=False)
                return {"text": text, "is_sentence_end": False}

            # 开始静音计时
            session.silence_start = time.time()
            logger.info(f"🔇 VAD: 进入静音，说话时长: {speech_duration:.1f}s")
            text = session.sentence_text
            return {
                "text": text,
                "is_sentence_end": False,
                "silence_duration": 0.0
            }

        # ─── 没有结束信号，继续说话 ───────────────────────────────
        # 流式识别
        text = _flush_pending(session, is_final=False)
        return {"text": text, "is_sentence_end": False}

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
    """WebSocket 流式 ASR 接口（支持 PTT 和 VAD 双模式）"""
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

                    # 根据模式处理音频
                    if session.mode == "ptt" and session.is_recording:
                        # PTT 模式：累积音频，不做 VAD 断句
                        session.asr_pending.extend(wav.tolist())
                        # 定期返回中间结果（可选）
                        text = _flush_pending(session, is_final=False)
                        await websocket.send_json({
                            "type": "partial",
                            "text": text
                        })

                    elif session.mode == "vad" and session.is_vad_active:
                        # VAD 模式：VAD 检测 + 静音超时
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, _process_audio_chunk_vad, wav, session
                        )

                        if result.get("vad_speech_start_detected"):
                            # VAD 检测到开始说话
                            await websocket.send_json({
                                "type": "vad_record_started"
                            })
                        elif result.get("is_vad_timeout"):
                            # VAD 静音超时，返回完整结果（类似 PTT 的 record_result）
                            session.records.append({
                                "text": result["text"],
                                "duration": result.get("duration", 0)
                            })
                            session.session_text += result["text"] + "\n"
                            logger.info(f"[WebSocket] VAD record ended: {result['text']}")
                            await websocket.send_json({
                                "type": "vad_record_result",
                                "text": result["text"],
                                "duration": result.get("duration", 0)
                            })
                        elif result.get("is_sentence_end"):
                            # VAD 断句（中间断句，不结束本轮）
                            await websocket.send_json({
                                "type": "sentence",
                                "text": result["text"]
                            })
                        elif result.get("text"):
                            # 流式中间结果
                            await websocket.send_json({
                                "type": "partial",
                                "text": result["text"],
                                "silence_duration": result.get("silence_duration", 0)
                            })

                elif "text" in data:
                    # 处理 JSON 控制命令
                    try:
                        msg = json.loads(data["text"])
                    except json.JSONDecodeError:
                        # 兼容旧的字符串命令
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
                        if mode in ("ptt", "vad"):
                            session.mode = mode
                            logger.info(f"[WebSocket] Mode set to: {mode}")
                            await websocket.send_json({"type": "mode_set", "mode": mode})
                        else:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Invalid mode: {mode}"
                            })

                    # ─── PTT 模式控制 ────────────────────────────────
                    elif msg_type == "record_start":
                        session.is_recording = True
                        session.asr_cache = {}
                        session.asr_pending = []
                        session.record_text = ""
                        session.sentence_text = ""        # 清空累积文本
                        session.sentence_start_time = 0.0
                        session.record_start_time = time.time()
                        logger.info(f"[WebSocket] PTT record started")
                        await websocket.send_json({"type": "record_started"})

                    elif msg_type == "record_end":
                        session.is_recording = False
                        # 强制结束 ASR，返回完整结果
                        text = _flush_pending(session, is_final=True)
                        duration = time.time() - session.record_start_time
                        session.record_text = text
                        session.records.append({"text": text, "duration": duration})
                        session.session_text += text + "\n"
                        logger.info(f"[WebSocket] PTT record ended: {text}")
                        await websocket.send_json({
                            "type": "record_result",
                            "text": text,
                            "duration": round(duration, 2)
                        })

                    # ─── VAD 模式控制 ────────────────────────────────
                    elif msg_type == "vad_start":
                        timeout = msg.get("timeout", 3.0)
                        session.vad_silence_timeout = timeout
                        session.is_vad_active = True
                        session.asr_cache = {}
                        session.asr_pending = []
                        session.record_text = ""
                        session.sentence_text = ""        # 清空累积文本
                        session.sentence_start_time = 0.0
                        session.record_start_time = time.time()
                        session.is_speaking = False
                        session.silence_start = 0.0
                        logger.info(f"[WebSocket] VAD started, timeout={timeout}s")
                        await websocket.send_json({
                            "type": "vad_started",
                            "timeout": timeout
                        })

                    elif msg_type == "vad_end":
                        session.is_vad_active = False
                        # 如果还有未处理的音频，强制结束
                        if session.asr_pending:
                            text = _flush_pending(session, is_final=True)
                            duration = time.time() - session.record_start_time
                            session.records.append({"text": text, "duration": duration})
                            session.session_text += text + "\n"
                            await websocket.send_json({
                                "type": "vad_result",
                                "text": text,
                                "duration": round(duration, 2)
                            })
                        logger.info(f"[WebSocket] VAD ended")
                        await websocket.send_json({"type": "vad_ended"})

                    # ─── 兼容旧协议 ────────────────────────────────
                    elif msg_type == "finish":
                        # 兼容旧的 finish 命令
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