import os
import time
import wave
import torch
import base64
import tempfile

import numpy            as np
from fastapi            import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses  import JSONResponse
from transformers       import AutoModel, AutoProcessor

# 初始化 FastAPI 应用
app = FastAPI(title="GLM-ASR-Nano API", description="基于 GLM-ASR-Nano 的语音识别 API 服务")

# 配置设备和模型路径
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
REPO_ID = "/data/LLM/GLM-ASR-Nano-2512"

# 全局加载模型和处理器（只加载一次，提升性能）
print(f"正在加载模型到 {DEVICE}...")
processor = AutoProcessor.from_pretrained(REPO_ID)
model = AutoModel.from_pretrained(REPO_ID, dtype=torch.bfloat16, device_map=DEVICE)
print("模型加载完成！")

# 通用的语音识别函数（抽离公共逻辑，避免代码重复）
def recognize_audio(audio_path: str) -> str:
    """
    通用音频识别函数
    :param audio_path: 音频文件路径（WAV格式）
    :return: 识别后的文本
    """
    # 构建输入消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "url": audio_path},
                {"type": "text", "text": "Please transcribe this audio into text"},
            ],
        }
    ]
    
    # 处理输入
    inputs = processor.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_dict=True, 
        return_tensors="pt"
    )
    inputs = inputs.to(DEVICE, dtype=torch.bfloat16)
    
    # 生成识别结果
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    
    # 解码结果
    transcription = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )[0]
    return transcription

@app.post("/asr/transcribe", summary="语音转文字（文件上传）")
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="需要识别的音频文件（支持wav格式）")
):
    """
    将上传的音频文件转换为文字
    - **audio_file**: 上传的wav格式音频文件
    """
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 创建临时文件保存上传的音频（避免文件路径问题）
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(await audio_file.read())
            temp_audio_path = temp_file.name
        
        # 调用通用识别函数
        transcription = recognize_audio(temp_audio_path)
        
        # 计算耗时
        cost_time = round(time.time() - start_time, 2)
        
        # 删除临时文件
        os.unlink(temp_audio_path)
        
        # 返回结果
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "transcription": transcription,
                "cost_time_seconds": cost_time,
                "device": DEVICE
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
    pcm_base64: bytes = Body(..., description="PCM原始音频数据（二进制）"),
    sample_rate: int = Body(16000, description="PCM采样率，默认16000"),
    channels: int = Body(1, description="声道数，默认单声道"),
    sample_width: int = Body(2, description="采样宽度（字节），默认2字节（16位）")
):
    """
    将PCM原始音频数据转换为文字
    - **pcm_data**: PCM二进制数据（必填）
    - **sample_rate**: 采样率，如16000、8000（默认16000）
    - **channels**: 声道数，1=单声道，2=立体声（默认1）
    - **sample_width**: 采样宽度，1=8位，2=16位（默认2）
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
        transcription = recognize_audio(temp_wav_path)
        
        # 计算耗时
        cost_time = round(time.time() - start_time, 2)
        
        # 删除临时文件
        os.unlink(temp_wav_path)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "transcription": transcription,
                "cost_time_seconds": cost_time,
                "device": DEVICE,
                "pcm_params": {
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "sample_width": sample_width
                }
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

@app.get("/health", summary="健康检查")
async def health_check():
    """检查服务是否正常运行"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": DEVICE
    }

# 启动服务的入口
if __name__ == "__main__":
    import uvicorn
    # 启动服务，默认端口 8000，允许外部访问
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8800,
        workers=1  # 模型加载在全局，建议单worker避免重复加载
    )