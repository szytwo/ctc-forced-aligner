from pydantic import BaseModel, Field

from wdd.model.APIBaseModel import ResponseBaseModel


class AlignAudioWord(BaseModel):
    start: float
    end: float
    word: str


class AlignAudioSegment(BaseModel):
    start: float
    end: float
    text: str
    words: list[AlignAudioWord]


class AlignAudioResult(BaseModel):
    segments: list[AlignAudioSegment]
    full_text: str
    duration: float
    language: str


class AlignAudioResponse(ResponseBaseModel):
    results: AlignAudioResult = Field(
        default=None,
        description="语音对齐结果",
    )

    class Config:
        json_schema_extra = {
            "description": "响应结果",
            "example": {
                "errcode": 0,
                "errmsg": "ok",
                "results": {
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "text": "示例文本",
                            "words": [{"start": 0.0, "end": 0.5, "word": "示例"}],
                        }
                    ],
                    "full_text": "完整示例文本",
                    "duration": 1.0,
                },
            },
        }
