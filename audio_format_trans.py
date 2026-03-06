import base64
import numpy as np
import soundfile as sf
from io import BytesIO

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

# --------------------------
# 极简使用示例
# --------------------------
if __name__ == "__main__":
    # 替换为你的本地WAV文件路径
    WAV_FILE_PATH = "wjg.wav"
    
    # 示例1: 转换为 (numpy数组, 采样率) 格式（最常用）
    np_data, sr = convert_wav_to_target_format(WAV_FILE_PATH, "numpy")
    print(f"音频数据形状: {np_data.shape}, 采样率: {sr}")
    
    # 示例2: 转换为base64字符串格式
    str_data = convert_wav_to_target_format(WAV_FILE_PATH, "str")
    print(f"\nbase64字符串（前80字符）: {str_data[:80]}...")
    
    # 示例3: 转换为列表格式
    list_data = convert_wav_to_target_format(WAV_FILE_PATH, "list")
    print(f"\n列表格式长度: {len(list_data)}, 元素类型: {type(list_data[0])}, {type(list_data[1])}")