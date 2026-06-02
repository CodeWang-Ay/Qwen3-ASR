"""
SenseVoiceSmall ASR 并发性能测试

测试不同并发数下的推理耗时，帮助判断 GPU 瓶颈。

用法:
    python test_concurrent.py
    python test_concurrent.py --max-workers 10
    python test_concurrent.py --audio example/en.mp3
"""

import time
import argparse
import statistics
import subprocess
import concurrent.futures
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


def get_audio_duration(audio_path):
    """获取音频时长(秒)，优先用 soundfile，fallback 到 ffprobe"""
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        pass
    try:
        import librosa
        duration = librosa.get_duration(filename=audio_path)
        return duration
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return None

# MODEL_DIR = "D:/gitlab_workspace/env_set/xiaozhi-server-dev/main/xiaozhi-server/models/SenseVoiceSmall"
MODEL_DIR = "/data/LLM/Fun-ASR-Nano-2512"


def load_model():
    print("loading model...")
    # model = AutoModel(
    #     model=MODEL_DIR,
    #     vad_kwargs={"max_single_segment_time": 30000},
    #     device="cuda:1",
    #     hub="hf",
    # )
    model = AutoModel(
        model=MODEL_DIR,
        trust_remote_code=True,
        remote_code="./model.py",
        device="cuda:1",
        # hub：download models from ms (for ModelScope) or hf (for Hugging Face).
        hub="ms"
    )
    print("model loaded")
    return model


def single_infer(model, audio_path, task_id, audio_duration=None):
    """单次推理任务"""
    wait_start = time.monotonic()
    res = model.generate(
        input=[audio_path],
        cache={},
        batch_size=1,
        hotwords=["verilLog"],
        # 中文、英文、日文 for Fun-ASR-Nano-2512
        # 中文、英文、粤语、日文、韩文、越南语、印尼语、泰语、马来语、菲律宾语、阿拉伯语、
        # 印地语、保加利亚语、克罗地亚语、捷克语、丹麦语、荷兰语、爱沙尼亚语、芬兰语、希腊语、
        # 匈牙利语、爱尔兰语、拉脱维亚语、立陶宛语、马耳他语、波兰语、葡萄牙语、罗马尼亚语、
        # 斯洛伐克语、斯洛文尼亚语、瑞典语 for Fun-ASR-MLT-Nano-2512
        language="中文",
        itn=True,  # or False
    )

    elapsed = time.monotonic() - wait_start
    # text = rich_transcription_postprocess(res[0]["text"])
    text = rich_transcription_postprocess(res[0]["text"])
    rtf = elapsed / audio_duration if audio_duration else None
    return {
        "task_id": task_id,
        "elapsed": elapsed,
        "rtf": rtf,
        "text": text[:50],
    }


def run_concurrent_test(model, audio_path, num_workers, audio_duration=None):
    """指定并发数运行测试"""
    print(f"\n{'='*60}")
    print(f"concurrent workers: {num_workers} | audio duration: {audio_duration:.2f}s" if audio_duration else f"concurrent workers: {num_workers}")
    print(f"{'='*60}")

    results = []
    total_start = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(single_infer, model, audio_path, i, audio_duration): i
            for i in range(num_workers)
        }

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            rtf_str = f"RTF: {result['rtf']:.4f}" if result['rtf'] else "RTF: N/A"
            print(
                f"  task {result['task_id']:>2d} done | "
                f"elapsed: {result['elapsed']:.3f}s | "
                f"{rtf_str} | "
                f"text: {result['text']}"
            )

    total_elapsed = time.monotonic() - total_start
    times = [r["elapsed"] for r in results]
    rtfs = [r["rtf"] for r in results if r["rtf"] is not None]

    print(f"\n  --- summary ---")
    print(f"  audio len:    {audio_duration:.2f}s" if audio_duration else "  audio len:    N/A")
    print(f"  total time:   {total_elapsed:.3f}s")
    print(f"  min:          {min(times):.3f}s")
    print(f"  max:          {max(times):.3f}s")
    print(f"  avg:          {statistics.mean(times):.3f}s")
    if len(times) > 1:
        print(f"  stdev:        {statistics.stdev(times):.3f}s")
    if rtfs:
        print(f"  avg RTF:      {statistics.mean(rtfs):.4f}")
        print(f"  max RTF:      {max(rtfs):.4f}")
    print(f"  throughput:   {num_workers / total_elapsed:.2f} req/s")

    return {
        "workers": num_workers,
        "total": total_elapsed,
        "avg": statistics.mean(times),
        "max": max(times),
        "avg_rtf": statistics.mean(rtfs) if rtfs else None,
        "max_rtf": max(rtfs) if rtfs else None,
        "throughput": num_workers / total_elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="ASR concurrent benchmark")
    parser.add_argument(
        "--audio",
        # default=f"{MODEL_DIR}/example/zh.mp3",            # 简单
        default=f"audio_data/02.wav",             # 困难
        help="audio file path",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="max concurrent workers to test (default: 8)",
    )
    args = parser.parse_args()

    model = load_model()

    audio_duration = get_audio_duration(args.audio)
    if audio_duration:
        print(f"\naudio file:     {args.audio}")
        print(f"audio duration: {audio_duration:.2f}s")
    else:
        print(f"\naudio file:     {args.audio}")
        print(f"audio duration: N/A (install soundfile/librosa/ffprobe to detect)")

    # warmup
    print("\nwarmup...")
    single_infer(model, args.audio, 0, audio_duration)
    print("warmup done")

    # 逐步增加并发数测试: 1, 2, 4, 8, ...
    test_levels = []
    n = 1
    while n <= args.max_workers:
        test_levels.append(n)
        n *= 2
    if args.max_workers not in test_levels:
        test_levels.append(args.max_workers)

    all_results = []
    for workers in test_levels:
        result = run_concurrent_test(model, args.audio, workers, audio_duration)
        all_results.append(result)

    # 汇总对比
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    if audio_duration:
        print(f"  audio file:     {args.audio}")
        print(f"  audio duration: {audio_duration:.2f}s")
        print()
    has_rtf = all_results[0]["avg_rtf"] is not None
    if has_rtf:
        print(f"{'workers':>8} | {'total(s)':>9} | {'avg(s)':>8} | {'max(s)':>8} | {'avg RTF':>8} | {'max RTF':>8} | {'throughput':>12}")
        print(f"{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}")
    else:
        print(f"{'workers':>8} | {'total(s)':>9} | {'avg(s)':>8} | {'max(s)':>8} | {'throughput':>12}")
        print(f"{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}")

    baseline_avg = all_results[0]["avg"]
    for r in all_results:
        slowdown = r["avg"] / baseline_avg
        if has_rtf:
            print(
                f"{r['workers']:>8} | {r['total']:>9.3f} | {r['avg']:>8.3f} | "
                f"{r['max']:>8.3f} | {r['avg_rtf']:>8.4f} | {r['max_rtf']:>8.4f} | "
                f"{r['throughput']:>8.2f} req/s  "
                f"(avg {slowdown:.1f}x vs single)"
            )
        else:
            print(
                f"{r['workers']:>8} | {r['total']:>9.3f} | {r['avg']:>8.3f} | "
                f"{r['max']:>8.3f} | {r['throughput']:>8.2f} req/s  "
                f"(avg {slowdown:.1f}x vs single)"
            )

    print(f"\n  baseline (1 worker) avg: {baseline_avg:.3f}s")
    if has_rtf:
        print(f"  baseline (1 worker) RTF: {all_results[0]['avg_rtf']:.4f}")
        print(f"  RTF < 1.0 = faster than realtime | RTF > 1.0 = slower than realtime")
    print(f"  if avg grows linearly with workers -> GPU is the bottleneck")

# 测试funasr_local并发问题
if __name__ == "__main__":
    main()
