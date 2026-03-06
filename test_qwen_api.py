import requests
import base64
import wave
import os

# API 配置
BASE_URL = "http://localhost:8800"
AUDIO_FILE_WAV = "wjg.wav"


def test_transcribe_file():
    """测试 /asr/transcribe 接口（文件上传）"""
    print("\n" + "=" * 50)
    print("测试 /asr/transcribe 接口（文件上传）")
    print("=" * 50)

    if not os.path.exists(AUDIO_FILE_WAV):
        print(f"错误：找不到文件 {AUDIO_FILE_WAV}")
        return

    url = f"{BASE_URL}/asr/transcribe"

    # 准备上传文件
    with open(AUDIO_FILE_WAV, "rb") as f:
        files = {"audio_file": (AUDIO_FILE_WAV, f, "audio/wav")}
        data = {
            "language": "Chinese",
            "context": ""
        }

        response = requests.post(url, files=files, data=data)

    print(f"状态码: {response.status_code}")
    print(f"响应内容:")
    print(response.json())


def test_transcribe_pcm():
    """测试 /asr/transcribe_pcm 接口（PCM数据）"""
    print("\n" + "=" * 50)
    print("测试 /asr/transcribe_pcm 接口（PCM数据）")
    print("=" * 50)

    if not os.path.exists(AUDIO_FILE_WAV):
        print(f"错误：找不到文件 {AUDIO_FILE_WAV}")
        return

    url = f"{BASE_URL}/asr/transcribe_pcm"

    # 读取 WAV 文件获取 PCM 数据和参数
    with wave.open(AUDIO_FILE_WAV, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        pcm_data = wf.readframes(n_frames)

    print(f"音频参数:")
    print(f"  声道数: {n_channels}")
    print(f"  采样宽度: {sample_width} 字节")
    print(f"  采样率: {sample_rate} Hz")

    # 将 PCM 数据编码为 base64
    pcm_base64 = base64.b64encode(pcm_data).decode("utf-8")

    # 发送请求
    data = {
        "pcm_base64": pcm_base64,
        "sample_rate": sample_rate,
        "channels": n_channels,
        "sample_width": sample_width,
        "language": "Chinese",
        "context": ""
    }

    response = requests.post(url, json=data)

    print(f"状态码: {response.status_code}")
    print(f"响应内容:")
    print(response.json())


def test_transcribe_base64():
    """测试 /asr/transcribe_base64 接口（Base64音频数据）"""
    print("\n" + "=" * 50)
    print("测试 /asr/transcribe_base64 接口（Base64音频数据）")
    print("=" * 50)

    if not os.path.exists(AUDIO_FILE_WAV):
        print(f"错误：找不到文件 {AUDIO_FILE_WAV}")
        return

    url = f"{BASE_URL}/asr/transcribe_base64"

    # 读取完整的 WAV 文件字节
    with open(AUDIO_FILE_WAV, "rb") as f:
        wav_bytes = f.read()

    # 编码为 base64
    wav_base64 = base64.b64encode(wav_bytes).decode("utf-8")

    data = {
        "audio_base64": wav_base64,
        "language": "Chinese",
        "context": "",
        "return_time_stamps": False
    }

    response = requests.post(url, json=data)

    print(f"状态码: {response.status_code}")
    print(f"响应内容:")
    print(response.json())


def test_health():
    """测试 /health 接口"""
    print("\n" + "=" * 50)
    print("测试 /health 接口")
    print("=" * 50)

    url = f"{BASE_URL}/health"
    response = requests.get(url)

    print(f"状态码: {response.status_code}")
    print(f"响应内容:")
    print(response.json())


if __name__ == "__main__":
    # 首先测试健康检查
    try:
        test_health()
    except Exception as e:
        print(f"健康检查失败: {e}")
        print("请确保服务已启动: python qwen_asr_app.py")
        exit(1)

    # # 测试文件上传接口
    # try:
    #     test_transcribe_file()
    # except Exception as e:
    #     print(f"测试失败: {e}")

    # # 测试 PCM 数据接口
    # try:
    #     test_transcribe_pcm()
    # except Exception as e:
    #     print(f"测试失败: {e}")

    # 测试 Base64 接口
    try:
        test_transcribe_base64()
    except Exception as e:
        print(f"测试失败: {e}")
