# file_server.py
import http.server
import socketserver
import os

# PORT = 8080
PORT = 8800
AUDIO_DIR = "audio_data"  # 改成你的音频目录

os.chdir(AUDIO_DIR)

handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), handler) as httpd:
    print(f"文件服务已启动: http://10.2.5.121:{PORT}/")
    print(f"音频访问地址示例: http://10.2.5.121:{PORT}/02.wav")
    httpd.serve_forever()