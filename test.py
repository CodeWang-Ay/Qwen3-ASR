import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
 
client = OpenAI(base_url="http://10.2.5.121:8872/v1", api_key="EMPTY")
 
def transcribe_audio(file_path):
    try:
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="/data/LLM/Qwen3-ASR-0.6B",
                file=audio_file,
                response_format="json"
            )
        return {"file": file_path, "text": transcription.text, "status": "success"}
    except Exception as e:
        return {"file": file_path, "error": str(e), "status": "failed"}
 
# 批量处理音频文件
audio_files = [
    "audio_data/01.wav", 
    "audio_data/02.wav", 
    "audio_data/03.wav", 
]
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(transcribe_audio, audio_files))
 
for result in results:
    print(f"文件: {result['file']}, 状态: {result['status']}")
    if result['status'] == 'success':
        print(f"  识别结果: {result['text'][:100]}...")