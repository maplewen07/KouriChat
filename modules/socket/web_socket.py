import asyncio
import websockets
from typing import Optional

from .gateway import SocketGateway

_gateway: Optional[SocketGateway] = None


async def handle(websocket):
    print("客户端已通过 WebSocket 连接")

    try:
        async for raw in websocket:
            print(raw)
            if _gateway is None:
                print("SocketGateway 未初始化，消息丢弃:", raw)
                continue

            await _gateway.handle_raw(raw, websocket)

    except Exception as e:
        print(f"处理客户端数据时发生错误: {e}")
    finally:
        # 断开时清理绑定
        try:
            if _gateway is not None and getattr(_gateway, "sender", None) is not None:
                _gateway.sender.unbind(websocket)
        except Exception:
            pass


async def run_server(host="0.0.0.0", port=12345):
    async with websockets.serve(handle, host, port):
        print(f"WebSocket 服务器已启动：ws://{host}:{port}")
        await asyncio.Future()  # 永不结束


def start_websocket_server(inbound_queue, sender, switch_avatar_func=None, logger=None, host="0.0.0.0", port=12345):
    global _gateway

    loop = asyncio.new_event_loop()
    sender.set_event_loop(loop)
    asyncio.set_event_loop(loop)

    _gateway = SocketGateway(
        inbound_queue=inbound_queue,
        sender=sender,
        switch_avatar_func=switch_avatar_func,
        logger=logger,
    )

    loop.run_until_complete(run_server(host, port))
    loop.run_forever()
