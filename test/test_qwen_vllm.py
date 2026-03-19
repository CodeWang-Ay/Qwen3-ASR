import os
# 设置使用的 GPU 显卡序号（修改这里的数字选择不同的显卡）
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from qwen_asr import Qwen3ASRModel
import wave
import base64
from typing import Union
import numpy as np
import soundfile as sf
from io import BytesIO

# ASR_MODEL_PATH = "Qwen/Qwen3-ASR-1.7B"
# FORCED_ALIGNER_PATH = "Qwen/Qwen3-ForcedAligner-0.6B"

ASR_MODEL_PATH = "/data/LLM/Qwen3-ASR-0.6B"
FORCED_ALIGNER_PATH = "/data/LLM/Qwen3-ForcedAligner-0.6B"
# URL_ZH = "wjg.wav"
URL_ZH = "audio_data/01.wav"

def load_qwen_asr_model(ASR_MODEL_PATH):
    qwen_asr_model = Qwen3ASRModel.LLM(
        model=ASR_MODEL_PATH,
        gpu_memory_utilization=0.8,
        max_inference_batch_size=32,
        max_new_tokens=1024,
    )
    return qwen_asr_model


def load_pcm_data(wav_path):
    with wave.open(wav_path, "rb") as wf:
        # 获取元信息
        n_channels = wf.getnchannels()      # 声道数
        sample_width = wf.getsampwidth()    # 采样宽度（字节）：1→8bit, 2→16bit, 3→24bit（注意！）
        sample_rate = wf.getframerate()     # 采样率 Hz
        n_frames = wf.getnframes()          # 总采样点数（每声道）

        print(f"声道数: {n_channels}")
        print(f"采样宽度: {sample_width} 字节 ({sample_width * 8}-bit)")
        print(f"采样率: {sample_rate} Hz")
        print(f"总帧数: {n_frames}")
        print(f"时长: {n_frames / sample_rate:.2f} 秒")

        # 读取原始字节数据
        raw_data = wf.readframes(n_frames)
        return raw_data, n_channels, sample_width, sample_rate

def load_wav_bytes(wav_path):
    """读取完整的 WAV 文件字节（包含 WAV 文件头）"""
    with open(wav_path, "rb") as f:
        return f.read()

def asr_transcribe_by_path(asr_model: Qwen3ASRModel):
    results = asr_model.transcribe(
        audio=URL_ZH,
        language=None,
        return_time_stamps=False,
    )
    print(results)

def asr_transcribe_by_pcm_data(asr_model: Qwen3ASRModel, audio_base64):
    # 方案1: 转换为 data URL base64 格式
    results = asr_model.transcribe(
        audio=audio_base64,
        context=["面试者为阳强华 热词有matlab,AD9361,1024,2048,visuallog,"],
        language=None,
        return_time_stamps=False,
    )
    print(results)

def convert_wav_to_target_format(wav_path, target_format="numpy"):
    """
    将本地WAV文件转换为指定的目标格式（简化版，仅支持本地路径）
    
    参数:
        wav_path (str): WAV文件的本地路径
        target_format (str): 目标格式，可选值:
            - "str": 返回base64 data url字符串（符合str格式要求）
            - "numpy": 返回 (np.ndarray, 采样率) 元组
            - "list": 返回包含上述两种格式的列表
    
    返回:
        对应格式的转换结果
    """
    # 验证目标格式
    valid_formats = ["str", "numpy", "list"]
    if target_format not in valid_formats:
        raise ValueError(f"目标格式仅支持: {valid_formats}")

    # 读取本地WAV文件为numpy数组和采样率（核心步骤）
    try:
        audio_data, sample_rate = sf.read(wav_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到WAV文件: {wav_path}")
    except Exception as e:
        raise RuntimeError(f"读取WAV文件失败: {str(e)}")

    # 转换为目标格式
    if target_format == "numpy":
        return (audio_data, sample_rate)
    
    elif target_format == "str":
        # 转换为base64 data url字符串（符合str格式要求）
        buffer = BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV')
        buffer.seek(0)
        base64_encoded = base64.b64encode(buffer.read()).decode('utf-8')
        return f"data:audio/wav;base64,{base64_encoded}"
    
    elif target_format == "list":
        # 返回包含str和numpy格式的列表
        str_format = convert_wav_to_target_format(wav_path, "str")
        numpy_format = convert_wav_to_target_format(wav_path, "numpy")
        return [str_format, numpy_format]

if __name__ == "__main__":
    # URL_ZH = "wjg.wav"
    qwen_asr_model = load_qwen_asr_model(ASR_MODEL_PATH)
    # 使用完整的 WAV 文件字节（包含文件头），而不是纯 PCM 数据
    str_data = convert_wav_to_target_format(URL_ZH, "str")
    # print(f"\nbase64字符串（前80字符）: {str_data[:80]}...")
    # print(str_data)
    asr_transcribe_by_pcm_data(qwen_asr_model, str_data)
    


