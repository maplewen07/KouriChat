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
    chat payload（新协议）：
    {
      "msg_id": "1700000000",
      "msg_type": "friend",
      "chat_id": "!room:server/",
      "sender": "maplewen",
      "content": "你好"
    }

    兼容旧字段：
    - chat_name -> chat_id
    """
    msg_id: Optional[str]
    msg_type: str
    chat_id: str
    sender: str
    content: str

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ChatPayload":
        sender = payload.get("sender") or ""

        # ✅ 新字段 chat_id 优先；兼容旧 chat_name
        chat_id = payload.get("chat_id") or payload.get("chat_name") or ""

        msg_type = payload.get("msg_type") or "friend"
        msg_id = payload.get("msg_id")
        msg_id = str(msg_id) if msg_id is not None else None

        return cls(
            msg_id=msg_id,
            msg_type=msg_type,
            chat_id=chat_id,
            sender=sender,
            content=payload.get("content") or "",
        )


@dataclass
class SocketMessage:
    """
    统一消息对象（给 dispatcher 用）
    """
    id: Optional[str]
    type: str
    content: str
    sender: str
    chat_id: str

    @classmethod
    def from_chat(cls, chat: ChatPayload) -> "SocketMessage":
        return cls(
            id=chat.msg_id,
            type=chat.msg_type,
            content=chat.content,
            sender=chat.sender,
            chat_id=chat.chat_id,
        )
