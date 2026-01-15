import asyncio
import json
import time


class SocketSender:
    """
    服务器发送接口（按 sender 绑定 websocket）
    - connections: sender_name -> websocket
    - last_chat_id: sender_name -> 最近一次 chat_id（用于默认回房间）
    """

    def __init__(self):
        self.connections = {}      # sender_name -> websocket
        self.last_chat_id = {}     # sender_name -> chat_id
        self.loop = None

    def set_event_loop(self, loop):
        self.loop = loop

    def bind_sender(self, sender_name: str, chat_id: str, websocket):
        """收到消息时绑定 sender -> websocket，并记录 sender 的 last_chat_id"""
        if sender_name:
            self.connections[sender_name] = websocket
        if sender_name and chat_id:
            self.last_chat_id[sender_name] = chat_id

    def unbind(self, websocket):
        """客户端断开时清理绑定"""
        for s, ws in list(self.connections.items()):
            if ws is websocket:
                del self.connections[s]
                self.last_chat_id.pop(s, None)

    def send_text(self, sender_name: str, content: str, chat_id: str = "", sender="Server"):
        """
        服务器主动发消息给指定 sender 对应的 websocket。
        - 如果 chat_id 不传：用该 sender 最近一次 chat_id
        """
        ws = self.connections.get(sender_name)
        if not ws:
            print(f"[SocketSender] 没有找到 sender={sender_name} 的连接")
            return False

        if not chat_id:
            chat_id = self.last_chat_id.get(sender_name, "")

        if not chat_id:
            print(f"[SocketSender] sender={sender_name} 没有可用 chat_id（还没收到过任何消息）")
            return False

        payload = {
            "type": "chat",
            "event": "message",
            "payload": {
                "msg_id": str(int(time.time())),  # 秒时间戳
                "msg_type": "friend",
                "chat_id": chat_id,
                "sender": sender,
                "content": content,
            }
        }

        if self.loop:
            asyncio.run_coroutine_threadsafe(
                ws.send(json.dumps(payload, ensure_ascii=False)),
                self.loop
            )
            return True

        print("[SocketSender] loop 未初始化")
        return False
