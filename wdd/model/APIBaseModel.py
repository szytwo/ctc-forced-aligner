from pydantic import BaseModel, Field


# 定义响应体模型
class ResponseBaseModel(BaseModel):
    errcode: int = Field(
        default=0,
        description="错误码，0 表示成功",
    )
    errmsg: str = Field(
        default="ok",
        description="错误信息，成功时为 'ok'",
    )

    class Config:
        json_schema_extra = {
            "description": "响应结果",
            "example": {"errcode": 0, "errmsg": "ok"},
        }
