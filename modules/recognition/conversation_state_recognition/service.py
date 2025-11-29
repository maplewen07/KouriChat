"""
对话状态识别服务

负责识别对话的当前状态（进行中、即将结束、已结束）
并判断是否应该发送回复
"""

import json
import logging
import os
import sys
from time import sleep
from typing import Optional, Dict

from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.services.ai.llm_service import LLMService
from src.autoupdate.updater import Updater
from data.config import config

logger = logging.getLogger('main')


class ConversationStateRecognitionService:
    def __init__(self, llm_service: LLMService):
        """
        初始化对话状态识别服务
        
        Args:
            llm_service: LLM 服务实例，用于调用 LLM
        """
        self.llm_service = llm_service
        self.reply_decision_settings = {
            "api_key": config.reply_decision.api_key,
            "base_url": config.reply_decision.base_url,
            "model": config.reply_decision.model,
            "temperature": config.reply_decision.temperature
        }

        self.updater = Updater()
        self.client = OpenAI(
            api_key=self.reply_decision_settings["api_key"],
            base_url=self.reply_decision_settings["base_url"],
            default_headers={
                "Content-Type": "application/json",
                "User-Agent": self.updater.get_version_identifier(),
                "X-KouriChat-Version": self.updater.get_current_version()
            }
        )
        self.config = self.llm_service.config

        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_dir, "prompt.md"), "r", encoding="utf-8") as f:
            self.sys_prompt = f.read().strip()

    def recognize(self, recent_messages: list) -> Dict:
        """
        识别对话状态
        
        Args:
            recent_messages: 最近的对话消息列表，格式为 [{"role": "user/assistant", "content": "..."}]
        
        Returns:
            Optional[Dict]: 包含对话状态信息的字典
            {
                "state": "ONGOING|ENDING|ENDED",
                "confidence": 0.0-1.0,
                "reason": "判断理由",
                "should_reply": true|false
            }
        """
        delay = 2
        current_model = self.reply_decision_settings["model"]
        logger.info(f"调用模型{current_model}进行对话状态识别...")

        # 构建对话历史文本
        conversation_text = self._format_conversation(recent_messages)

        messages = [{"role": "system", "content": self.sys_prompt}]

        # 加载示例
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_dir, "example_message.json"), 'r', encoding='utf-8') as f:
            data = json.load(f)

        for example in data.values():
            messages.append({
                "role": example["input"]["role"],
                "content": example["input"]["content"]
            })
            messages.append({
                "role": example["output"]["role"],
                "content": str(example["output"]["content"])
            })

        # 添加当前对话
        messages.append({
            "role": "user",
            "content": f"最近对话历史：\n{conversation_text}"
        })

        request_config = {
            "model": self.reply_decision_settings["model"],
            "messages": messages,
            "temperature": self.reply_decision_settings["temperature"],
            "max_tokens": self.config["max_token"],
        }

        for retries in range(3):
            try:
                response = self.client.chat.completions.create(**request_config)
                response_content = response.choices[0].message.content

                # 针对 Gemini 模型的回复进行预处理
                if response_content.startswith("```json") and response_content.endswith("```"):
                    response_content = response_content[7:-3].strip()
                elif response_content.startswith("```") and response_content.endswith("```"):
                    response_content = response_content[3:-3].strip()

                # 尝试解析JSON
                try:
                    result = json.loads(response_content)

                    # 验证返回格式
                    if self._validate_result(result):
                        logger.info(
                            f"对话状态识别成功: {result['state']}, 置信度: {result['confidence']}, 应该回复: {result['should_reply']}")
                        return result
                    else:
                        logger.warning(f"对话状态识别结果格式不正确: {result}")

                except json.JSONDecodeError as e:
                    logger.warning(f"识别对话状态失败：JSON解析错误 {str(e)}，进行重试...({retries + 1}/3)")
                    logger.info(f"响应内容：{response_content}")

            except Exception as e:
                logger.warning(f"识别对话状态失败：{str(e)}，进行重试...({retries + 1}/3)")

            sleep(delay)
            delay *= 2

        logger.error("多次重试后仍未能识别对话状态，返回默认值（进行中）")
        return {
            "state": "ONGOING",
            "confidence": 0.5,
            "reason": "识别失败，默认为进行中",
            "should_reply": True
        }

    def _format_conversation(self, messages: list) -> str:
        """
        格式化对话历史为文本
        
        Args:
            messages: 消息列表
            
        Returns:
            str: 格式化后的对话文本
        """
        formatted = []
        for msg in messages[-5:]:  # 只取最近5条消息
            role = "用户" if msg["role"] == "user" else "AI"
            content = msg["content"]
            # 移除可能的群聊标记
            if content.startswith("[群聊消息]"):
                content = content.replace("[群聊消息]", "").strip()
                # 移除发送者名称
                if ":" in content:
                    content = content.split(":", 1)[1].strip()
            formatted.append(f"{role}：{content}")
        return "\n".join(formatted)

    def _validate_result(self, result: Dict) -> bool:
        """
        验证识别结果的格式
        
        Args:
            result: 识别结果字典
            
        Returns:
            bool: 是否有效
        """
        if not isinstance(result, dict):
            return False

        # 检查必需字段
        required_fields = ["state", "confidence", "reason", "should_reply"]
        if not all(field in result for field in required_fields):
            return False

        # 验证state值
        if result["state"] not in ["ONGOING", "ENDING", "ENDED"]:
            return False

        # 验证confidence范围
        if not isinstance(result["confidence"], (int, float)) or not (0 <= result["confidence"] <= 1):
            return False

        # 验证should_reply类型
        if not isinstance(result["should_reply"], bool):
            return False

        return True


'''
单独对模块进行调试时，可以使用该代码
'''
if __name__ == '__main__':
    llm_service = LLMService(
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        model=config.llm.model,
        max_token=1024,
        temperature=0.8,
        max_groups=5
    )
    test = ConversationStateRecognitionService(llm_service)

    # 测试用例
    test_messages = [
        {"role": "user", "content": "今天天气怎么样？"},
        {"role": "assistant", "content": "今天天气不错，阳光明媚呢"},
        {"role": "user", "content": "好的知道了"}
    ]

    result = test.recognize(test_messages)
    if result:
        print(f"状态: {result['state']}")
        print(f"置信度: {result['confidence']}")
        print(f"理由: {result['reason']}")
        print(f"应该回复: {result['should_reply']}")
