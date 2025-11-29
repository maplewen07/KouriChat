# modules/socket/models.py
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any
import json

SocketType = Literal["rpc", "chat", "system"]


@dataclass
class SocketEnvelope:
    """
    顶层协议封装：
    {
      "type": "rpc" | "chat" | "system",
      "event": "...",
      "role": "YueJi",
      "payload": { ... },
      "request_id": "xxx"
    }
    """
    type: SocketType
    event: str
    role: Optional[str]
    payload: Dict[str, Any]
    request_id: Optional[str] = None

    @classmethod
    def from_raw(cls, raw: str) -> "SocketEnvelope":
        """从原始 JSON 字符串解析为协议对象"""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"非法 JSON: {e}") from e

        if "type" not in data or "event" not in data:
            raise ValueError("缺少必要字段: type 或 event")

        msg_type = data["type"]
        if msg_type not in ("rpc", "chat", "system"):
            raise ValueError(f"不支持的 type: {msg_type}")

        return cls(
            type=msg_type,
            event=data.get("event", ""),
            role=data.get("role"),
            payload=data.get("payload") or {},
            request_id=data.get("request_id"),
        )


@dataclass
class ChatPayload:
    """
    chat 类型消息的 payload：
    {
      "msg_id": "...",
      "msg_type": "friend",
      "chat_name": "张三",
      "sender": "张三",
      "content": "你好"
    }
    """
    msg_id: Optional[str]
    msg_type: str
    chat_name: str
    sender: str
    content: str

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ChatPayload":
        return cls(
            msg_id=payload.get("msg_id"),
            msg_type=payload.get("msg_type", "friend"),
            chat_name=payload.get("chat_name") or payload.get("sender") or "",
            sender=payload.get("sender") or "",
            content=payload.get("content") or "",
        )


@dataclass
class SocketMessage:
    """
    提供给 message_dispatcher / PrivateChatBot 使用的统一消息对象
    尽量模拟原来 wxauto 的 msg：
      - id
      - type
      - content
      - sender
      - who/chat_name
    """
    id: Optional[str]
    type: str
    content: str
    sender: str
    chat_name: str  # 相当于原来的 who

    @classmethod
    def from_chat(cls, chat: ChatPayload) -> "SocketMessage":
        return cls(
            id=chat.msg_id,
            type=chat.msg_type,
            content=chat.content,
            sender=chat.sender,
            chat_name=chat.chat_name,
        )
