import os
from pathlib import Path

import numpy as np
from fastapi import UploadFile
from pydub import AudioSegment

from wdd.file_utils import add_suffix_to_filename, logging


class AudioProcessor:
    def __init__(self):
        """
        初始化音频处理器，设置临时文件目录。
        """

    # noinspection PyTypeChecker
    async def save_upload_to_wav(
        self,
        upload_file: UploadFile,
        audio_dir: str = "",
        prefix: str = "",
    ):
        """
        保存上传文件并转换为 WAV 格式（如果需要）

        参数：
            upload_file (UploadFile): FastAPI 上传的音频文件对象
            audio_dir: str 音频存放目录
            prefix (str): 文件名前缀（默认值为空字符串）

        返回：
            Path: 处理后保存的 WAV 文件路径

        异常：
            Exception: 在文件保存或处理过程中可能引发异常
        """
        # 构建保存文件的完整路径
        upload_path = os.path.join(audio_dir, f"{prefix}{upload_file.filename}")
        # 如果目标文件已存在，删除旧文件以避免冲突
        if os.path.exists(upload_path):
            os.remove(upload_path)
        # 将路径对象化，方便后续操作
        upload_path = Path(upload_path)
        # 如果文件格式不是 WAV，准备转换为 WAV 格式
        if upload_path.suffix.lower() != ".wav":
            # 创建转换后的 WAV 文件路径
            wav_path = str(
                upload_path.with_stem(f"{upload_path.stem}_new").with_suffix(".wav")
            )
        else:
            wav_path = str(upload_path)
        # 返回字符串路径
        upload_path = str(upload_path)

        logging.info(f"Received upload request: {upload_file.filename} → {upload_path}")

        try:
            # 保存上传的原始音频文件
            with open(upload_path, "wb") as f:
                f.write(await upload_file.read())
            # 加载音频文件为 AudioSegment 对象（支持多种格式）
            audio = AudioSegment.from_file(upload_path)
            # 导出处理后的音频文件为 WAV 格式
            audio.export(wav_path, format="wav")

            return wav_path
        except Exception as e:
            # 捕获并抛出任何在处理过程中发生的异常
            raise Exception(
                f"Failed to save audio file {upload_file.filename}: {str(e)}"
            )
        finally:
            await upload_file.close()  # 显式关闭上传文件
