import os
import time
import wave
import torch
import base64
import tempfile
import io
from typing import Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from qwen_asr import Qwen3ASRModel

# 初始化 FastAPI API
app = FastAPI(title="Qwen3-ASR API", description="基于 Qwen3-ASR 的语音识别 API 服务")

# 配置模型路径
ASR_MODEL_PATH = "/data/LLM/Qwen3-ASR-0.6B"
# FORCED_ALIGNER_PATH = "/data/LLM/Qwen3-ForcedAligner-0.6B"

# 全局加载模型（只加载一次，提升性能）
print(f"正在加载 Qwen3-ASR 模型...")
# asr = Qwen3ASRModel.LLM(
#     model=ASR_MODEL_PATH,
#     gpu_memory_utilization=0.8,
#     # forced_aligner=FORCED_ALIGNER_PATH,
#     # forced_aligner_kwargs=dict(
#     #     dtype=torch.bfloat16,
#     #     device_map="cuda:7",
#     # ),
#     max_inference_batch_size=32,
#     max_new_tokens=1024,
# )
def load_qwen_asr_model(ASR_MODEL_PATH):
    qwen_asr_model = Qwen3ASRModel.LLM(
        model=ASR_MODEL_PATH,
        gpu_memory_utilization=0.8,
        max_inference_batch_size=32,
        max_new_tokens=1024,
    )
    return qwen_asr_model


def _to_data_url_base64(audio_bytes: bytes, mime: str = "audio/wav") -> str:
    """将音频字节转为 data URL base64 格式"""
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _read_wav_from_bytes(audio_bytes: bytes):
    """从字节读取 WAV 文件返回 numpy 数组和采样率"""
    import soundfile as sf
    with io.BytesIO(audio_bytes) as f:
        wav, sr = sf.read(f, dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32), int(sr)


def recognize_audio(audio_path: str, language: Optional[str] = None, context: Optional[str] = None) -> dict:
    """
    通用音频识别函数
    :param audio_path: 音频文件路径（WAV格式）
    :param language: 指定语言（如 "Chinese", "English"），None 表示自动检测
    :param context: 上下文提示词
    :return: 识别结果字典
    """
    # 读取音频文件
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # 转换为 data URL base64 格式
    audio_b64 = _to_data_url_base64(audio_bytes, mime="audio/wav")

    # 构建参数
    kwargs = {
        "audio": audio_b64,
        "language": language,
    }
    if context is not None:
        kwargs["context"] = context

    # 调用 ASR 模型
    results = asr.transcribe(**kwargs)

    # 解析结果
    if results and len(results) > 0:
        result = results[0]
        return {
            "text": result.text,
            "language": result.language,
            "time_stamps": [
                {
                    "text": ts.text,
                    "start_time": ts.start_time,
                    "end_time": ts.end_time
                }
                for ts in result.time_stamps
            ] if result.time_stamps else None
        }
    return {"text": "", "language": None, "time_stamps": None}


@app.post("/asr/transcribe", summary="语音转文字（文件上传）")
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="需要识别的音频文件（支持wav格式）"),
    language: Optional[str] = Body(None, description="指定语言（如 Chinese, English），不填则自动检测"),
    context: Optional[str] = Body(None, description="上下文提示词")
):
    """
    将上传的音频文件转换为文字
    - **audio_file**: 上传的wav格式音频文件
    - **language**: 可选，指定识别语言
    - **context**: 可选，上下文提示词
    """
    try:
        # 记录开始时间
        start_time = time.time()

        # 创建临时文件保存上传的音频（避免文件路径问题）
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(await audio_file.read())
            temp_audio_path = temp_file.name

        # 调用通用识别函数
        result = recognize_audio(temp_audio_path, language=language, context=context)

        # 计算耗时
        cost_time = round(time.time() - start_time, 2)

        # 删除临时文件
        os.unlink(temp_audio_path)

        # 返回结果
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "transcription": result["text"],
                "language": result["language"],
                "cost_time_seconds": cost_time,
                "time_stamps": result["time_stamps"]
            }
        )

    except Exception as e:
        # 异常处理
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "message": "语音识别过程中出现错误"
            }
        )


@app.post("/asr/transcribe_pcm", summary="PCM数据转文字")
async def transcribe_pcm(
    pcm_base64: bytes = Body(..., description="PCM原始音频数据（base64编码）"),
    sample_rate: int = Body(16000, description="PCM采样率，默认16000"),
    channels: int = Body(1, description="声道数，默认单声道"),
    sample_width: int = Body(2, description="采样宽度（字节），默认2字节（16位）"),
    language: Optional[str] = Body(None, description="指定语言（如 Chinese, English），不填则自动检测"),
    context: Optional[str] = Body(None, description="上下文提示词")
):
    """
    将PCM原始音频数据转换为文字
    - **pcm_base64**: PCM二进制数据（base64编码，必填）
    - **sample_rate**: 采样率，如16000、8000（默认16000）
    - **channels**: 声道数，1=单声道，2=立体声（默认1）
    - **sample_width**: 采样宽度，1=8位，2=16位（默认2）
    - **language**: 可选，指定识别语言
    - **context**: 可选，上下文提示词
    """
    try:
        start_time = time.time()
        pcm_data = base64.b64decode(pcm_base64)

        # 创建临时WAV文件（将PCM转为WAV）
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_wav_path = temp_file.name
            # 写入WAV文件头和PCM数据
            with wave.open(temp_wav_path, 'wb') as wf:
                wf.setnchannels(channels)  # 设置声道数
                wf.setsampwidth(sample_width)  # 设置采样宽度
                wf.setframerate(sample_rate)  # 设置采样率
                wf.writeframes(pcm_data)  # 写入PCM数据

        # 调用通用识别函数
        result = recognize_audio(temp_wav_path, language=language, context=context)

        # 计算耗时
        cost_time = round(time.time() - start_time, 2)

        # 删除临时文件
        os.unlink(temp_wav_path)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "transcription": result["text"],
                "language": result["language"],
                "cost_time_seconds": cost_time,
                "pcm_params": {
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "sample_width": sample_width
                },
                "time_stamps": result["time_stamps"]
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "message": "PCM音频识别过程中出现错误"
            }
        )


@app.post("/asr/transcribe_base64", summary="Base64音频数据转文字")
async def transcribe_base64(
    audio_base64: str = Body(..., description="音频文件的base64编码"),
    language: Optional[str] = Body(None, description="指定语言（如 Chinese, English），不填则自动检测"),
    context: Optional[str] = Body(None, description="上下文提示词"),
    return_time_stamps: bool = Body(False, description="是否返回时间戳")
):
    """
    将base64编码的音频数据转换为文字
    - **audio_base64**: 音频数据的base64编码（必填）
    - **language**: 可选，指定识别语言
    - **context**: 可选，上下文提示词
    - **return_time_stamps**: 是否返回时间戳信息
    """
    try:
        start_time = time.time()

        # # 解码base64
        # audio_bytes = base64.b64decode(audio_base64)

        # # 构建参数
        # kwargs = {
        #     "audio": _to_data_url_base64(audio_bytes),
        #     "language": language,
        #     "return_time_stamps": return_time_stamps,
        # }
        
        kwargs = {
            "audio": f"data:audio/wav;base64,{audio_base64}",
            "language": language,
            "return_time_stamps": return_time_stamps,
        }       
        if context is not None:
            kwargs["context"] = context

        # 调用 ASR 模型
        results = asr.transcribe(**kwargs)
        # 计算耗时
        cost_time = round(time.time() - start_time, 2)
        # 解析结果
        if results and len(results) > 0:
            result = results[0]
            time_stamps = None
            if return_time_stamps and result.time_stamps:
                time_stamps = [
                    {
                        "text": ts.text,
                        "start_time": ts.start_time,
                        "end_time": ts.end_time
                    }
                    for ts in result.time_stamps
                ]

            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "transcription": result.text,
                    "language": result.language,
                    "cost_time_seconds": cost_time,
                    "time_stamps": time_stamps
                }
            )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "transcription": "",
                "language": None,
                "cost_time_seconds": cost_time,
                "time_stamps": None
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "message": "Base64音频识别过程中出现错误"
            }
        )


@app.get("/health", summary="健康检查")
async def health_check():
    """检查服务是否正常运行"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": ASR_MODEL_PATH
    }


# 启动服务的入口
if __name__ == "__main__":
    asr = load_qwen_asr_model(ASR_MODEL_PATH)
    import uvicorn
    # 启动服务，默认端口 8801，允许外部访问
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8800,
        workers=1  # 模型加载在全局，建议单worker避免重复加载
    )
