## fork

https://github.com/hankcs/HanLP

## 安装

```
py -3.11 -m venv venv
venv\Scripts\activate

pip install -e .[dev]

ctc-forced-aligner --audio_path "D:\AI\Montreal-Forced-Aligner\results01\325528cosyvoice\325528cosyvoice.wav" --text_path "D:\AI\Montreal-Forced-Aligner\results01\325528cosyvoice\325528cosyvoice.txt" --language "zlm" --romanize

pip install -r ./api_requirements.txt -i https://mirrors.aliyun.com/pypi/simple

nvidia-smi -L  # 查看GUID

nvidia-smi -q | grep Persistence # 查看参数
nvidia-smi -q | findstr /C:"Persistence" # 查看参数（windows）

nvidia-smi -pm 1 # 开启 GPU 的“持久化模式”（Persistence Mode）
nvidia-smi -pl 350 # 设置 GPU 的 Power Limit（功耗上限）, 温度高可降低功耗

```
