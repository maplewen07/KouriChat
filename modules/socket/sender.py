# modules/socket/sender.py
import asyncio
import json


class SocketSender:
    """提供统一的服务器发送接口：根据 chat_id 找到 websocket，然后发送消息"""

    def __init__(self):
        # chat_id -> websocket 映射
        self.chat_connections = {}

        # 保存事件循环，用于跨线程向 websocket 发送消息
        self.loop = None

    def set_event_loop(self, loop):
        """在启动 ws 服务器时记录 asyncio loop"""
        self.loop = loop

    def bind_chat(self, chat_id: str, websocket):
        """当收到消息时，把 chat_id 与 websocket 建立映射"""
        self.chat_connections[chat_id] = websocket

    def unbind(self, websocket):
        """客户端断开时清理绑定"""
        for key, ws in list(self.chat_connections.items()):
            if ws is websocket:
                del self.chat_connections[key]

    def send_text(self, chat_id: str, content: str, sender = "Server"):
        """机器人主动向指定 chat_id 发送文本消息"""

        ws = self.chat_connections.get(chat_id)
        if not ws:
            print(f"[SocketSender] 没有找到 chat_id={chat_id} 的连接")
            return False

        payload = {
            "type": "chat",
            "event": "message",
            "payload": {
                "content": content,
                "sender": sender,
                "chat_name": chat_id,
            }
        }

        # 确保跨线程安全发送
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                ws.send(json.dumps(payload)),
                self.loop
            )
            return True

        print("[SocketSender] loop 未初始化")
        return False
