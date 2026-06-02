# benchmark_url.py
import asyncio
import aiohttp
import time
from tqdm.asyncio import tqdm_asyncio

url = "http://10.2.5.121:8872/v1/chat/completions"
# url = "http://10.2.5.121:8870/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# ⚠️ 改成你客户端机器的内网 IP
MY_IP = "10.2.5.121"
AUDIO_FILENAME = "02.wav"
AUDIO_URL = f"http://{MY_IP}:17800/{AUDIO_FILENAME}"

# 提前构造好请求体，所有并发复用，避免重复序列化
import json
PAYLOAD = json.dumps({
    "messages": [{
        "role": "user",
        "content": [{
            "type": "audio_url",
            "audio_url": {"url": AUDIO_URL}
        }]
    }]
}).encode("utf-8")

async def send_request(session, semaphore, request_id):
    async with semaphore:
        start_time = time.time()
        try:
            async with session.post(
                url,
                headers=headers,
                data=PAYLOAD,  # 直接传 bytes，跳过重复序列化
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                response.raise_for_status()
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                elapsed = time.time() - start_time
                return {"request_id": request_id, "status": "success", "content": content, "elapsed": elapsed}
        except Exception as e:
            elapsed = time.time() - start_time
            return {"request_id": request_id, "status": "error", "error": str(e), "elapsed": elapsed}

async def main(total_requests=30, concurrency=20):
    semaphore = asyncio.Semaphore(concurrency)
    print(f"音频URL: {AUDIO_URL}")
    print(f"总请求数: {total_requests}, 并发量: {concurrency}")
    print("-" * 50)

    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, semaphore, i) for i in range(total_requests)]
        results = await tqdm_asyncio.gather(*tasks, desc="处理请求")
    total_elapsed = time.time() - start_time

    success_results = [r for r in results if r["status"] == "success"]
    error_results = [r for r in results if r["status"] == "error"]

    print("\n" + "=" * 50)
    print(f"测试结果统计 并发数: {concurrency}")
    print("=" * 50)
    print(f"总请求数:    {total_requests}")
    print(f"成功数:      {len(success_results)}")
    print(f"失败数:      {len(error_results)}")
    print(f"总耗时:      {total_elapsed:.2f}s")

    if success_results:
        elapsed_times = [r["elapsed"] for r in success_results]
        print(f"elapsed_times: {[round(t, 2) for t in elapsed_times[:20]]}s")
        print(f"平均响应时间: {sum(elapsed_times)/len(elapsed_times):.2f}s")
        print(f"最快响应时间: {min(elapsed_times):.2f}s")
        print(f"最慢响应时间: {max(elapsed_times):.2f}s")

    if error_results:
        print("\n失败详情:")
        for r in error_results:
            print(f"  请求 #{r['request_id']}: {r['error']}")

    print("\n成功响应示例:")
    for r in success_results[:2]:
        print(f"  请求 #{r['request_id']} ({r['elapsed']:.2f}s): {r['content'][:100]}")

if __name__ == "__main__":
    asyncio.run(main(total_requests=30, concurrency=30))