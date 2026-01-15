import asyncio
import json
import time
from typing import Any, Dict

import websockets


def now_ms() -> int:
    return int(time.time() * 1000)


def make_chat_message(chat_name: str, content: str, sender: str = "WS-Test-Server") -> Dict[str, Any]:
    return {
        "type": "chat",
        "event": "message",
        "payload": {
            "msg_id": str(now_ms()),
            "msg_type": "server",
            "chat_name": chat_name,
            "sender": sender,
            "content": content,
        }
    }


def make_rpc_reply(req_event: str, request_id: str, ok: bool, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "rpc",
        "event": f"{req_event}.ok" if ok else f"{req_event}.error",
        "payload": payload,
        "request_id": request_id,
    }


async def handle(ws):
    print("✅ 客户端已连接:", ws.remote_address)

    # 可选：连接上来就发一条系统消息（你可以在 Matrix 里看到）
    try:
        await ws.send(json.dumps(make_chat_message(
            chat_name="",
            content="[server] connected (this goes to default_room_id if chat_name empty)",
            sender="test"
        ), ensure_ascii=False))
    except Exception:
        pass

    try:
        async for raw in ws:
            print("\n--- RAW FROM CLIENT ---")
            print(raw)

            # 尝试解析
            try:
                env = json.loads(raw)
            except Exception as e:
                err = {"type": "system", "event": "error", "payload": {"message": f"invalid json: {e}"}}
                await ws.send(json.dumps(err, ensure_ascii=False))
                continue

            msg_type = env.get("type")
            event = env.get("event", "")
            payload = env.get("payload") or {}
            request_id = env.get("request_id")

            # 1) chat.message：立刻回一条 echo
            if msg_type == "chat" and event == "message":
                chat_name = payload.get("chat_name", "")
                sender = payload.get("sender", "")
                content = payload.get("content", "")

                # 回显到同一个 chat_name（你插件用 room_id 做 chat_name，就能回到原房间）
                reply_text = f"[echo] from={sender} content={content}"
                reply = make_chat_message(chat_name=chat_name, content=reply_text, sender="WS-Test-Server")
                await ws.send(json.dumps(reply, ensure_ascii=False))
                print("--- SENT ECHO TO CLIENT ---")
                print(json.dumps(reply, ensure_ascii=False))

            # 2) rpc.ping：返回 rpc.ping.ok
            elif msg_type == "rpc" and event == "ping":
                if not request_id:
                    # 没有 request_id 也回，但你那边可能不会等
                    request_id = str(now_ms())
                reply = make_rpc_reply("ping", request_id, True, {"pong": True})
                await ws.send(json.dumps(reply, ensure_ascii=False))
                print("--- SENT RPC PONG ---")
                print(json.dumps(reply, ensure_ascii=False))

            # 3) 其他消息：回 system.ack
            else:
                ack = {
                    "type": "system",
                    "event": "ack",
                    "payload": {"seen_type": msg_type, "seen_event": event},
                    "request_id": request_id,
                }
                await ws.send(json.dumps(ack, ensure_ascii=False))
                print("--- SENT ACK ---")
                print(json.dumps(ack, ensure_ascii=False))

    except websockets.ConnectionClosed:
        print("⚠️ 客户端断开:", ws.remote_address)
    except Exception as e:
        print("❌ handle error:", repr(e))


async def main(host: str = "0.0.0.0", port: int = 16667):
    print(f"🚀 WS Test Server listening on ws://{host}:{port}")
    async with websockets.serve(handle, host, port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
