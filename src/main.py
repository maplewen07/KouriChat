import logging
import random
from datetime import datetime, timedelta
import threading
import time
import os
import shutil
from src.utils.console import print_status

# 率先初始化网络适配器以覆盖所有网络库
try:
    from src.autoupdate.core.manager import initialize_system

    initialize_system()
    print_status("网络适配器初始化成功", "success", "CHECK")
except Exception as e:
    print_status(f"网络适配器初始化失败: {str(e)}", "error", "CROSS")

# 导入其余模块
from data.config import config, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL, MAX_TOKEN, TEMPERATURE, MAX_GROUPS
import re
from modules.socket.sender import SocketSender
from src.handlers.message import MessageHandler
from src.services.ai.llm_service import LLMService
from modules.socket.web_socket import start_websocket_server
from modules.socket.models import SocketMessage
from modules.socket.web_socket import start_websocket_server
from src.services.ai.image_recognition_service import ImageRecognitionService
from modules.memory.memory_service import MemoryService
from modules.memory.content_generator import ContentGenerator
from src.utils.logger import LoggerConfig
from colorama import init, Style
from src.AutoTasker.autoTasker import AutoTasker
from src.handlers.autosend import AutoSendHandler
import queue
from collections import defaultdict

# 创建一个事件对象来控制线程的终止
stop_event = threading.Event()

# 获取项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 检查并初始化配置文件
config_path = os.path.join(root_dir, 'src', 'config', 'config.json')
config_template_path = os.path.join(root_dir, 'src', 'config', 'config.json.template')

if not os.path.exists(config_path) and os.path.exists(config_template_path):
    logger = logging.getLogger('main')
    logger.info("配置文件不存在，正在从模板创建...")
    shutil.copy2(config_template_path, config_path)
    logger.info(f"已从模板创建配置文件: {config_path}")

# 初始化colorama
init()

# 全局变量
logger = None
# 从配置中读取监听列表，默认空列表
listen_list = []
# WebSocket服务器配置
host = '0.0.0.0'
port = 12345

def initialize_logging():
    """初始化日志系统"""
    global logger, listen_list, port

    # 清除所有现有日志处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger_config = LoggerConfig(root_dir)
    logger = logger_config.setup_logger('main')
    listen_list = config.user.listen_list
    port = config.user.websocket_port

    # 确保autoupdate模块的日志级别设置为DEBUG
    logging.getLogger("autoupdate").setLevel(logging.DEBUG)
    logging.getLogger("autoupdate.core").setLevel(logging.DEBUG)
    logging.getLogger("autoupdate.interceptor").setLevel(logging.DEBUG)
    logging.getLogger("autoupdate.network_optimizer").setLevel(logging.DEBUG)


# 消息队列接受消息时间间隔
wait = 1

# 添加消息队列用于分发
private_message_queue = queue.Queue()

# socket 统一入口队列，供 message_dispatcher 消费
socket_inbound_queue = queue.Queue()

socket_sender = SocketSender()

class PrivateChatBot:
    """专门处理私聊的机器人"""

    def __init__(self, message_handler, image_recognition_service, auto_sender, emoji_handler):
        self.message_handler = message_handler
        self.auto_sender = auto_sender

        # @Todo
        # self.wx = WeChat()
        # self.robot_name = self.wx.A_MyIcon.Name
        # logger.info(f"私聊机器人初始化完成 - 机器人名称: {self.robot_name}")

        # 私聊始终使用默认人设
        from data.config import config
        default_avatar_path = config.behavior.context.avatar_dir
        self.current_avatar = os.path.basename(default_avatar_path)
        logger.info(f"私聊机器人使用默认人设: {self.current_avatar}")

    def handle_private_message(self, msg, chat_name):
        """处理私聊消息"""
        try:
            username = msg.sender
            content = getattr(msg, 'content', None) or getattr(msg, 'text', None)

            # 重置倒计时
            self.auto_sender.start_countdown()

            logger.info(f"[私聊] 收到消息 - 来自: {username}")
            logger.debug(f"[私聊] 消息内容: {content}")

            is_image_recognition = False

            # 处理消息
            if content:
                self.message_handler.handle_user_message(
                    content=content,
                    chat_id=chat_name,
                    sender_name=username,
                    username=username,
                    is_group=False,
                    is_image_recognition=is_image_recognition
                )

        except Exception as e:
            logger.error(f"[私聊] 消息处理失败: {str(e)}")


def private_message_processor():
    """私聊消息处理线程"""
    logger.info("私聊消息处理线程启动")

    while not stop_event.is_set():
        try:
            # 从队列获取私聊消息
            msg_data = private_message_queue.get(timeout=1)
            logger.info(msg_data)
            if msg_data is None:  # 退出信号
                break
            
            msg, sender = msg_data

            private_chat_bot.handle_private_message(msg, sender)
            private_message_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"私聊消息处理线程出错: {str(e)}")


# 全局变量
prompt_content = ""
emoji_handler = None
image_handler = None
memory_service = None
content_generator = None
message_handler = None
image_recognition_service = None
auto_sender = None
private_chat_bot = None
group_chat_bot = None
ROBOT_WX_NAME = ""
processed_messages = set()
last_processed_content = {}


def initialize_services():
    """初始化服务实例"""
    global prompt_content, emoji_handler, image_handler, memory_service, content_generator
    global message_handler, image_recognition_service, auto_sender, private_chat_bot, group_chat_bot, ROBOT_WX_NAME

    # 尝试获取热更新模块状态信息以确认其状态
    try:
        from src.autoupdate.core.manager import get_manager
        try:
            status = get_manager().get_status()
            if status:
                print_status(f"热更新模块已就绪", "success", "CHECK")
            else:
                print_status("热更新模块状态异常", "warning", "CROSS")

        except Exception as e:
            print_status(f"检查热更新模块状态时出现异常: {e}", "error", "ERROR")

    except Exception as e:
        print_status(f"检查热更新模块状态时出现异常: {e}", "error", "ERROR")

    # 读取提示文件
    avatar_dir = os.path.join(root_dir, config.behavior.context.avatar_dir)
    prompt_path = os.path.join(avatar_dir, "avatar.md")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as file:
            prompt_content = file.read()

        # 处理无法读取文件的情况
    else:
        raise FileNotFoundError(f"avatar.md 文件不存在: {prompt_path}")

    memory_service = MemoryService(
        root_dir=root_dir,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        model=MODEL,
        max_token=MAX_TOKEN,
        temperature=TEMPERATURE,
        max_groups=MAX_GROUPS
    )

    content_generator = ContentGenerator(
        root_dir=root_dir,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        model=MODEL,
        max_token=MAX_TOKEN,
        temperature=TEMPERATURE
    )

    # 创建消息处理器
    message_handler = MessageHandler(
        ws_sender=socket_sender,
        root_dir=root_dir,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        model=config.llm.model,
        max_token=config.llm.max_tokens,
        temperature=config.llm.temperature,
        max_groups=config.behavior.context.max_groups,
        robot_name=ROBOT_WX_NAME,  # 使用动态获取的机器人名称
        prompt_content=prompt_content,
        image_handler=image_handler,
        emoji_handler=emoji_handler,
        memory_service=memory_service,  # 使用新的记忆服务
        content_generator=content_generator  # 直接传递内容生成器实例
    )

    # 创建主动消息处理器
    auto_sender = AutoSendHandler(message_handler, config, listen_list)

    # 创建并行聊天机器人实例 
    private_chat_bot = PrivateChatBot(message_handler, image_recognition_service, auto_sender, emoji_handler)

    # 启动主动消息倒计时
    auto_sender.start_countdown()


def message_dispatcher():
    """消息分发器 - 将消息分发到对应的处理队列"""
    global logger, wait, processed_messages, last_processed_content, listen_list

    logger.info("消息分发器启动")

    while not stop_event.is_set():
        try:
            try:
                msg: SocketMessage = socket_inbound_queue.get(timeout=1)
            except queue.Empty:
                continue

            who = msg.chat_name  # 会话名
            sender = msg.sender
            content = msg.content
            msg_id = msg.id
            msgtype = msg.type

            if not who or not sender:
                logger.debug("消息缺少 chat_name 或 sender，已忽略")
                continue

            if not content:
                continue

            # 去重
            if msg_id and msg_id in processed_messages:
                logger.debug(f"跳过已处理的消息ID: {msg_id}")
                continue
            if msg_id:
                processed_messages.add(msg_id)
            last_processed_content[who] = content

            # 可选：监听列表过滤（保持原逻辑）
            if listen_list and who not in listen_list:
                logger.debug(f"消息来源不在监听列表中，忽略: {who}")
                continue

            # 私聊：窗口名 == 发送人
            if who == sender and msgtype == "friend":
                logger.debug(f"[分发] 私聊消息 -> 私聊队列: {who}")
                private_message_queue.put((msg, sender))
                continue


        except Exception as e:
            logger.debug(f"消息分发出错: {str(e)}")
            wx = None
        time.sleep(wait)


def initialize_auto_tasks(message_handler):
    """初始化自动任务系统"""
    print_status("初始化自动任务系统...", "info", "CLOCK")

    try:
        # 导入config变量
        from data.config import config

        # 创建AutoTasker实例
        auto_tasker = AutoTasker(message_handler)
        print_status("创建AutoTasker实例成功", "success", "CHECK")

        # 清空现有任务
        auto_tasker.scheduler.remove_all_jobs()
        print_status("清空现有任务", "info", "CLEAN")

        # 从配置文件读取任务信息
        if hasattr(config, 'behavior') and hasattr(config.behavior, 'schedule_settings'):
            schedule_settings = config.behavior.schedule_settings
            if schedule_settings and schedule_settings.tasks:  # 直接检查 tasks 列表
                tasks = schedule_settings.tasks
                if tasks:
                    print_status(f"从配置文件读取到 {len(tasks)} 个任务", "info", "TASK")
                    tasks_added = 0

                    # 遍历所有任务并添加
                    for task in tasks:
                        try:
                            # 添加定时任务
                            auto_tasker.add_task(
                                task_id=task.task_id,
                                chat_id=listen_list[0],  # 使用 listen_list 中的第一个聊天ID
                                content=task.content,
                                schedule_type=task.schedule_type,
                                schedule_time=task.schedule_time
                            )
                            tasks_added += 1
                            print_status(f"成功添加任务 {task.task_id}: {task.content}", "success", "CHECK")
                        except Exception as e:
                            print_status(f"添加任务 {task.task_id} 失败: {str(e)}", "error", "ERROR")

                    print_status(f"成功添加 {tasks_added}/{len(tasks)} 个任务", "info", "TASK")
                else:
                    print_status("配置文件中没有找到任务", "warning", "WARNING")
        else:
            print_status("未找到任务配置信息", "warning", "WARNING")
            print_status(f"当前 behavior 属性: {dir(config.behavior)}", "info", "INFO")

        return auto_tasker

    except Exception as e:
        print_status(f"初始化自动任务系统失败: {str(e)}", "error", "ERROR")
        logger.error(f"初始化自动任务系统失败: {str(e)}")
        return None


def switch_avatar(new_avatar_name):
    # 使用全局变量
    global emoji_handler, private_chat_bot, group_chat_bot, root_dir

    # 导入config变量
    from data.config import config

    # 更新配置
    config.behavior.context.avatar_dir = f"avatars/{new_avatar_name}"

    # 更新私聊和群聊机器人中的 emoji_handler
    if private_chat_bot:
        private_chat_bot.emoji_handler = emoji_handler
        private_chat_bot.message_handler.emoji_handler = emoji_handler


def main():
    # 初始化变量
    dispatcher_thread = None
    private_thread = None
    ws_thread = None

    try:
        # 初始化日志系统
        initialize_logging()

        # 初始化服务实例
        initialize_services()

        # 验证记忆目录
        print_status("验证角色记忆存储路径...", "info", "FILE")
        avatar_dir = os.path.join(root_dir, config.behavior.context.avatar_dir)
        avatar_name = os.path.basename(avatar_dir)
        memory_dir = os.path.join(avatar_dir, "memory")
        if not os.path.exists(memory_dir):
            os.makedirs(memory_dir)
            print_status(f"创建角色记忆目录: {memory_dir}", "success", "CHECK")

        # 初始化记忆文件 - 为每个监听用户创建独立的记忆文件
        print_status("初始化记忆文件...", "info", "FILE")

        # 为每个监听的用户创建独立记忆
        for user_name in listen_list:
            print_status(f"为用户 '{user_name}' 创建独立记忆...", "info", "USER")
            # 使用用户名作为用户ID
            memory_service.initialize_memory_files(avatar_name, user_id=user_name)
            print_status(f"用户 '{user_name}' 记忆初始化完成", "success", "CHECK")

        avatar_dir = os.path.join(root_dir, config.behavior.context.avatar_dir)
        prompt_path = os.path.join(avatar_dir, "avatar.md")
        if not os.path.exists(prompt_path):
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write("# 核心人格\n[默认内容]")
            print_status(f"创建人设提示文件", "warning", "WARNING")
        # 启动并行消息处理系统
        print_status("启动并行消息处理系统...", "info", "ANTENNA")

        # 启动消息分发线程
        dispatcher_thread = threading.Thread(target=message_dispatcher, name="MessageDispatcher")
        dispatcher_thread.daemon = True

        # 启动私聊处理线程
        private_thread = threading.Thread(target=private_message_processor, name="PrivateProcessor")
        private_thread.daemon = True

        ws_thread = threading.Thread(
            target=start_websocket_server,
            args=(socket_inbound_queue,socket_sender,switch_avatar, logger, host, port),
            name="WebSocketServer",
            daemon=True
        )

        # 启动所有线程
        dispatcher_thread.start()
        private_thread.start()
        ws_thread.start()

        print_status("并行消息处理系统已启动", "success", "CHECK")
        print_status("  ├─ 消息分发器线程", "info", "ANTENNA")
        print_status("  ├─ 私聊处理器线程", "info", "USER")
        print_status("  ├─ WebSocket 服务器线程", "info", "ANTENNA")

        # 初始化主动消息系统
        print_status("初始化主动消息系统...", "info", "CLOCK")
        print_status("主动消息系统已启动", "success", "CHECK")

        print("-" * 50)
        print_status("系统初始化完成", "success", "STAR_2")
        print("=" * 50)

        # 初始化自动任务系统
        auto_tasker = initialize_auto_tasks(message_handler)
        if not auto_tasker:
            print_status("自动任务系统初始化失败", "error", "ERROR")
            return

        # 主循环 - 监控并行处理线程状态
        while True:
            time.sleep(1)

            # 检查关键线程状态
            threads_status = [
                ("消息分发器", dispatcher_thread),
                ("私聊处理器", private_thread),
                ("WebSocket 服务器", ws_thread)
            ]

            dead_threads = []
            for thread_name, thread in threads_status:
                if not thread.is_alive():
                    dead_threads.append(thread_name)

            if dead_threads:
                print_status(f"检测到线程异常: {', '.join(dead_threads)}", "warning", "WARNING")
                # 这里可以添加重启逻辑，暂时先记录
                time.sleep(5)

    except Exception as e:
        print_status(f"主程序异常: {str(e)}", "error", "ERROR")
        logger.error(f"主程序异常: {str(e)}", exc_info=True)
    finally:
        # 清理资源
        if 'auto_sender' in locals():
            auto_sender.stop()

        # 设置事件以停止线程
        stop_event.set()

        # 向队列发送退出信号
        try:
            private_message_queue.put(None)
        except:
            pass

        # 等待所有处理线程结束
        threads_to_wait = [
            ("消息分发器", dispatcher_thread),
            ("私聊处理器", private_thread),
            ("WebSocket 服务器", ws_thread)
        ]

        for thread_name, thread in threads_to_wait:
            if thread and thread.is_alive():
                print_status(f"正在关闭{thread_name}线程...", "info", "SYNC")
                thread.join(timeout=3)
                if thread.is_alive():
                    print_status(f"{thread_name}线程未能正常关闭", "warning", "WARNING")

        print_status("正在关闭系统...", "warning", "STOP")
        print_status("系统已退出", "info", "BYE")
        print("\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n")
        print_status("用户终止程序", "warning", "STOP")
        print_status("感谢使用，再见！", "info", "BYE")
        print("\n")
    except Exception as e:
        print_status(f"程序异常退出: {str(e)}", "error", "ERROR")