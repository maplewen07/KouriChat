import asyncio
import json
import time
import websockets

current_client = None  # 先只维护一个客户端


async def handle_connection(websocket):
    global current_client
    current_client = websocket
    print("🟢 客户端已连接")

    # 先发一条欢迎消息
    await send_chat_to_client("ServerBot", "欢迎连接到 Python WebSocket 测试服务器！")

    try:
        await asyncio.gather(
            recv_loop(websocket),
            input_loop(websocket),
        )
    except websockets.ConnectionClosed:
        print("🔴 客户端断开连接")
    finally:
        if current_client is websocket:
            current_client = None


async def recv_loop(ws):
    """接收客户端消息并打印"""
    async for raw in ws:
        print(f"📩 收到客户端消息: {raw}")


async def input_loop(ws):
    """从终端输入内容，发送给客户端"""
    loop = asyncio.get_running_loop()
    while True:
        # 注意：input 是阻塞的，所以放到线程池中执行
        text = await loop.run_in_executor(None, lambda: input("server> "))
        text = text.strip()
        if not text:
            continue

        if text.lower() in ("exit", "quit"):
            print("⏹ 关闭连接")
            await ws.close()
            break

        await send_chat_to_client("ServerBot", text)


async def send_chat_to_client(chat_name: str, content: str):
    global current_client
    ws = current_client
    if ws is None:
        print("⚠ 没有客户端连接，无法发送消息")
        return

    msg = {
        "type": "chat",
        "event": "message",
        "role": chat_name,
        "payload": {
            "msg_id": f"server-{int(time.time() * 1000)}",
            "sender": chat_name,
            "chat_name": chat_name,
            "content": content,
            "timestamp": int(time.time()),
        },
        "request_id": f"server-msg-{int(time.time() * 1000)}",
    }

    raw = json.dumps(msg, ensure_ascii=False)
    await ws.send(raw)
    print(f"📤 已发送给客户端: {raw}")


async def main():
    print("🚀 启动 WebSocket 服务器 ws://0.0.0.0:12345")
    async with websockets.serve(handle_connection, "0.0.0.0", 12345):
        await asyncio.Future()  # 一直阻塞


if __name__ == "__main__":
    asyncio.run(main())
