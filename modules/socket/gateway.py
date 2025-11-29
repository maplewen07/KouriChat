# modules/socket/gateway.py
from queue import Queue
from typing import Optional, Callable, Any
from collections import deque
import json
import time

from .models import SocketEnvelope, ChatPayload, SocketMessage


class SocketGateway:
    """
    WebSocket 网关：
    - 解析顶层 Envelope
    - 按 type/event 分发
    - chat: 保存最近 100 条消息，并推入 message_dispatcher
    - rpc/system: 执行指令
    """

    def __init__(self, inbound_queue, sender, switch_avatar_func=None, logger=None, message_cache_limit=100):
        self.inbound_queue = inbound_queue
        self.sender = sender  # ⭐ 新增
        self.switch_avatar_func = switch_avatar_func
        self.logger = logger
        self.message_cache = deque(maxlen=message_cache_limit)

    async def handle_raw(self, raw: str, websocket):
        """WebSocket 收到一条字符串消息时调用"""
        try:
            envelope = SocketEnvelope.from_raw(raw)
        except ValueError as e:
            if self.logger:
                self.logger.warning(f"解析 WebSocket 消息失败: {e}")
            try:
                await websocket.send(json.dumps({
                    "type": "system",
                    "event": "error",
                    "payload": {"message": str(e)},
                }))
            except Exception:
                pass
            return

        if envelope.type == "chat":
            await self._handle_chat(envelope, websocket)
        elif envelope.type == "rpc":
            await self._handle_rpc(envelope, websocket)
        elif envelope.type == "system":
            await self._handle_system(envelope, websocket)

    async def _handle_chat(self, envelope: SocketEnvelope, websocket):
        """处理聊天消息：缓存 + 分发"""

        chat_payload = ChatPayload.from_payload(envelope.payload)
        msg = SocketMessage.from_chat(chat_payload)

        self.message_cache.append({
            "msg_id": msg.id,
            "msg_type": msg.type,
            "content": msg.content,
            "sender": msg.sender,
            "chat_name": msg.chat_name,
            "role": envelope.role,
            "timestamp": time.time(),
        })

        if self.logger:
            self.logger.debug(
                f"[Gateway] chat: {msg.sender}({msg.chat_name}) → {msg.content}"
            )

        # 在 _handle_chat() 中：
        self.sender.bind_chat(msg.chat_name, websocket)

        # 推送给 dispatcher（保持原结构）
        self.inbound_queue.put(msg)

        # 可选：按 role 自动切人设
        if envelope.role and self.switch_avatar_func:
            try:
                self.switch_avatar_func(envelope.role)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"切换角色失败({envelope.role}): {e}")

    async def _handle_rpc(self, envelope: SocketEnvelope, websocket):
        """RPC 指令"""
        if self.logger:
            self.logger.info(f"[Gateway] rpc: {envelope.event}, payload={envelope.payload}")

        event = envelope.event

        if event == "switch_role":
            role = envelope.payload.get("role") or envelope.role
            if role and self.switch_avatar_func:
                try:
                    self.switch_avatar_func(role)
                    return await self._reply_ok(websocket, envelope, {"message": f"已切换角色为 {role}"})
                except Exception as e:
                    return await self._reply_error(websocket, envelope, str(e))
            return await self._reply_error(websocket, envelope, "缺少 role")

        elif event == "ping":
            return await self._reply_ok(websocket, envelope, {"pong": True})

        elif event == "get_recent_messages":
            # ⭐ 新增 RPC：返回最近消息
            history = list(self.message_cache)
            return await self._reply_ok(websocket, envelope, {"messages": history})

        else:
            return await self._reply_error(websocket, envelope, f"未知 RPC: {event}")

    async def _handle_system(self, envelope: SocketEnvelope, websocket):
        if envelope.event == "ping":
            return await websocket.send(json.dumps({
                "type": "system",
                "event": "pong",
                "payload": {},
                "request_id": envelope.request_id,
            }))

    async def _reply_ok(self, websocket, envelope, data):
        await websocket.send(json.dumps({
            "type": "rpc",
            "event": envelope.event + ".ok",
            "payload": data,
            "request_id": envelope.request_id,
        }))

    async def _reply_error(self, websocket, envelope, message: str):
        await websocket.send(json.dumps({
            "type": "rpc",
            "event": envelope.event + ".error",
            "payload": {"message": message},
            "request_id": envelope.request_id,
        }))
