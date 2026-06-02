"""
基于 paraformer-zh-streaming + fsmn-vad 的实时流式中文语音识别 Web 版本
通过 Web Audio API 获取麦克风音频，通过 HTTP 流式发送到后端

Install:
  pip install flask

Run:
  python web_asr_realtime.py
Open:
  http://127.0.0.1:8000
"""
import os
import queue
import time
import uuid
import numpy as np
import threading

from dataclasses  import dataclass
from funasr       import AutoModel
from dotenv       import load_dotenv
from typing       import Dict, Optional
from flask        import Flask, Response, jsonify, request

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

# ─── Flask 应用 ────────────────────────────────────────────────────
app = Flask(__name__)


# CORS middleware: add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response


# ════════════════════════════════════════════════════════════════
print("⏳ 加载 ASR 模型...")
asr_model = AutoModel(
    model=ASR_MODEL_PATH,
    model_revision="v2.0.4",
    disable_update=True,
)
print("✅ ASR 模型加载完成")

print("⏳ 加载 VAD 模型...")
vad_model = AutoModel(
    model=VAD_MODEL_PATH,
    model_revision="v2.0.4",
    disable_update=True,
)
print("✅ VAD 模型加载完成\n")


# ─── 会话管理 ──────────────────────────────────────────────────────
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


def _gc_sessions():
    """清理过期会话"""
    now = time.time()
    dead = [sid for sid, s in SESSIONS.items() if now - s.last_seen > SESSION_TTL_SEC]
    for sid in dead:
        SESSIONS.pop(sid, None)


def _get_session(session_id: str) -> Optional[Session]:
    """获取会话并更新最后访问时间"""
    _gc_sessions()
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
        print(f"\n⚠️  ASR 异常: {e}")
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


# ─── HTML 页面 ─────────────────────────────────────────────────────
INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Paraformer 实时语音识别</title>
  <style>
    :root{
      --bg:#ffffff;
      --card:#ffffff;
      --muted:#5b6472;
      --text:#0f172a;
      --border:#e5e7eb;
      --ok:#059669;
      --warn:#d97706;
      --danger:#e11d48;
    }

    html, body { height: 100%; }

    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans";
      background: var(--bg);
      color:var(--text);
    }

    .wrap{
      height: 100vh;
      max-width: none;
      margin: 0;
      padding: 16px;
      box-sizing: border-box;
      display: flex;
    }

    .card{
      width: 100%;
      height: 100%;
      background: var(--card);
      border:1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      box-sizing: border-box;
      box-shadow: 0 10px 30px rgba(0,0,0,.06);
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-height: 0;
    }

    h1{ font-size: 18px; margin: 0; letter-spacing:.2px; font-weight: 600;}

    .row{ display:flex; gap:12px; align-items:center; flex-wrap: wrap; }

    button{
      border:1px solid var(--border); border-radius: 12px;
      padding: 10px 16px; cursor:pointer; color:var(--text);
      background: #f8fafc;
      transition: transform .05s ease, background .15s ease, border-color .15s ease;
      font-weight: 600;
      font-size: 14px;
    }
    button:hover{ background: #f1f5f9; border-color:#cbd5e1; }
    button:active{ transform: translateY(1px); }
    button.primary{ border-color: rgba(5,150,105,.35); background: rgba(5,150,105,.10); }
    button.danger{ border-color: rgba(225,29,72,.35); background: rgba(225,29,72,.10); }
    button:disabled{ opacity:.5; cursor:not-allowed; }

    .pill{
      font-size: 12px; padding: 6px 10px; border-radius: 999px;
      border:1px solid var(--border); color: var(--muted);
      background: #f8fafc;
      user-select:none;
    }
    .pill.ok{ color: #065f46; border-color: rgba(5,150,105,.35); background: rgba(5,150,105,.10); }
    .pill.warn{ color: #92400e; border-color: rgba(217,119,6,.35); background: rgba(217,119,6,.10); }
    .pill.err{ color: #9f1239; border-color: rgba(225,29,72,.35); background: rgba(225,29,72,.10); }

    .panel{
      border:1px solid var(--border);
      border-radius: 12px;
      background: #ffffff;
      padding: 12px;
    }

    .panel.textpanel{
      flex: 1;
      display: flex;
      flex-direction: column;
      min-height: 0;
    }

    .label{ color:var(--muted); font-size: 12px; margin-bottom: 6px; }
    .mono{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New"; }

    #text{
      flex: 1;
      min-height: 0;
      white-space: pre-wrap;
      line-height: 1.6;
      font-size: 15px;
      padding: 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #f8fafc;
      overflow: auto;
    }

    a{ color: #2563eb; text-decoration: none; font-size: 13px; font-weight: 600; }

    .info{ font-size: 13px; color: var(--muted); }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Paraformer 实时语音识别</h1>

      <div class="row">
        <button id="btnStart" class="primary">开始识别</button>
        <button id="btnStop" class="danger" disabled>停止识别</button>
        <span id="status" class="pill warn">未开始</span>
        <a href="javascript:void(0)" id="btnClear" style="margin-left:auto;">清空文本</a>
      </div>

      <div class="panel">
        <div class="row">
          <span class="info">模型: paraformer-zh-streaming + fsmn-vad</span>
        </div>
      </div>

      <div class="panel textpanel">
        <div class="label">识别文本</div>
        <div id="text"></div>
      </div>
    </div>
  </div>

<script>
(() => {
  const $ = (id) => document.getElementById(id);

  const btnStart = $("btnStart");
  const btnStop  = $("btnStop");
  const btnClear = $("btnClear");
  const statusEl = $("status");
  const textEl   = $("text");

  // VAD 块大小 200ms，与后端一致
  const VAD_CHUNK_MS = 200;
  const TARGET_SR = 16000;

  let audioCtx = null;
  let processor = null;
  let source = null;
  let mediaStream = null;

  let sessionId = null;
  let running = false;

  let buf = new Float32Array(0);
  let pushing = false;

  // 当前句子的文本
  let currentSentence = "";
  let allText = [];

  function setStatus(text, cls){
    statusEl.textContent = text;
    statusEl.className = "pill " + (cls || "");
  }

  function lockUI(on){
    btnStart.disabled = on;
    btnStop.disabled = !on;
  }

  function concatFloat32(a, b){
    const out = new Float32Array(a.length + b.length);
    out.set(a, 0);
    out.set(b, a.length);
    return out;
  }

  function resampleLinear(input, srcSr, dstSr){
    if (srcSr === dstSr) return input;
    const ratio = dstSr / srcSr;
    const outLen = Math.max(0, Math.round(input.length * ratio));
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++){
      const x = i / ratio;
      const x0 = Math.floor(x);
      const x1 = Math.min(x0 + 1, input.length - 1);
      const t = x - x0;
      out[i] = input[x0] * (1 - t) + input[x1] * t;
    }
    return out;
  }

  function updateTextDisplay(){
    // 显示所有已完成句子 + 当前句子
    textEl.textContent = allText.join("") + currentSentence;
    // 滚动到底部
    textEl.scrollTop = textEl.scrollHeight;
  }

  async function apiStart(){
    const r = await fetch("/api/start", {method:"POST"});
    if(!r.ok) throw new Error(await r.text());
    const j = await r.json();
    sessionId = j.session_id;
  }

  async function apiPushChunk(float32_16k){
    const r = await fetch("/api/chunk?session_id=" + encodeURIComponent(sessionId), {
      method: "POST",
      headers: {"Content-Type":"application/octet-stream"},
      body: float32_16k.buffer
    });
    if(!r.ok) throw new Error(await r.text());
    return await r.json();
  }

  async function apiFinish(){
    const r = await fetch("/api/finish?session_id=" + encodeURIComponent(sessionId), {method:"POST"});
    if(!r.ok) throw new Error(await r.text());
    return await r.json();
  }

  btnClear.onclick = () => {
    allText = [];
    currentSentence = "";
    updateTextDisplay();
  };

  async function stopAudioPipeline(){
    try{
      if (processor){ processor.disconnect(); processor.onaudioprocess = null; }
      if (source) source.disconnect();
      if (audioCtx) await audioCtx.close();
      if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
    }catch(e){}
    processor = null; source = null; audioCtx = null; mediaStream = null;
  }

  btnStart.onclick = async () => {
    if (running) return;

    allText = [];
    currentSentence = "";
    updateTextDisplay();

    buf = new Float32Array(0);

    try{
      setStatus("启动中...", "warn");
      lockUI(true);

      await apiStart();

      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        },
        video: false
      });

      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      source = audioCtx.createMediaStreamSource(mediaStream);

      processor = audioCtx.createScriptProcessor(4096, 1, 1);
      const chunkSamples = Math.round(TARGET_SR * (VAD_CHUNK_MS / 1000));

      processor.onaudioprocess = (e) => {
        if (!running) return;
        const input = e.inputBuffer.getChannelData(0);
        const resampled = resampleLinear(input, audioCtx.sampleRate, TARGET_SR);
        buf = concatFloat32(buf, resampled);
        if (!pushing) pump();
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      running = true;
      setStatus("识别中...", "ok");

    }catch(err){
      console.error(err);
      setStatus("启动失败: " + err.message, "err");
      lockUI(false);
      running = false;
      sessionId = null;
      await stopAudioPipeline();
    }
  };

  async function pump(){
    if (pushing) return;
    pushing = true;

    const chunkSamples = Math.round(TARGET_SR * (VAD_CHUNK_MS / 1000));

    try{
      while (running && buf.length >= chunkSamples){
        const chunk = buf.slice(0, chunkSamples);
        buf = buf.slice(chunkSamples);

        const j = await apiPushChunk(chunk);

        if (j.is_sentence_end && j.text) {
          // 句子结束，保存到历史
          allText.push(j.text);
          currentSentence = "";
        } else if (j.text) {
          // 句子进行中，更新当前句子
          currentSentence = j.text;
        }

        updateTextDisplay();

        if (running) setStatus("识别中...", "ok");
      }
    }catch(err){
      console.error(err);
      if (running) setStatus("后端错误: " + err.message, "err");
    }finally{
      pushing = false;
    }
  }

  btnStop.onclick = async () => {
    if (!running) return;

    running = false;
    setStatus("停止中...", "warn");
    lockUI(false);

    await stopAudioPipeline();

    try{
      if (sessionId){
        const j = await apiFinish();
        if (j.is_sentence_end && j.text) {
          allText.push(j.text);
          currentSentence = "";
        } else if (j.text) {
          currentSentence = j.text;
        }
        updateTextDisplay();
      }
      setStatus("已停止", "");
    }catch(err){
      console.error(err);
      setStatus("停止失败: " + err.message, "err");
    }finally{
      sessionId = null;
      buf = new Float32Array(0);
      pushing = false;
    }
  };
})();
</script>
</body>
</html>
"""


# ─── Flask 路由 ───────────────────────────────────────────────────
@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html; charset=utf-8")

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


@app.post("/api/start")
def api_start():
    """启动一个新的识别会话"""
    session_id = uuid.uuid4().hex
    now = time.time()
    SESSIONS[session_id] = Session(
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
    print(f"✅ 新会话 {session_id}")
    return jsonify({"session_id": session_id})


@app.post("/api/chunk")
def api_chunk():
    """处理音频块"""
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

    # VAD 检测
    vad_speech_start = False
    vad_speech_end = False
    try:
        vad_res = vad_model.generate(
            input=wav,
            cache=s.vad_cache,
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
        print(f"\n⚠️  VAD 异常: {ex}")

    # 语音开始
    if vad_speech_start and not s.is_speaking:
        s.is_speaking = True
        s.silence_start = 0.0
        s.sentence_start_time = time.time()
        s.sentence_text = ""
        print(f"🎤 语音开始")

    if s.is_speaking:
        s.asr_pending.extend(wav.tolist())

        if vad_speech_end:
            # VAD 给出自然断句
            print(f"⏹️ 语音结束")
            result = _end_sentence(s)
            return jsonify(result)
        else:
            # 流式识别
            text = _flush_pending(s, is_final=False)
            return jsonify({
                "text": text,
                "is_sentence_end": False
            })

    return jsonify({"text": "", "is_sentence_end": False})


@app.post("/api/finish")
def api_finish():
    """结束识别会话"""
    session_id = request.args.get("session_id", "")
    s = _get_session(session_id)
    if not s:
        return jsonify({"error": "invalid session_id"}), 400

    # 如果正在说话，结束当前句子
    if s.is_speaking:
        result = _end_sentence(s)
        SESSIONS.pop(session_id, None)
        return jsonify(result)

    SESSIONS.pop(session_id, None)
    return jsonify({"text": "", "is_sentence_end": False})


if __name__ == "__main__":
    print("=" * 60)
    print("  Paraformer 实时语音识别 Web 服务")
    print("  模型: paraformer-zh-streaming + fsmn-vad")
    print("  访问: http://127.0.0.1:21590")
    print("  按 Ctrl+C 停止")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=21590, debug=False, threaded=True)

