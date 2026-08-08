"""记忆机制实现 - 包含短期记忆和长期记忆"""
import json
import os
from typing import List, Dict, Optional
from datetime import datetime

class ShortTermMemory:
    """短期记忆 - 存储当前对话历史"""
    
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages: List[Dict] = []
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """添加消息到短期记忆"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        # 保持消息数量在限制内
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_recent_messages(self, n: int = 10) -> List[Dict]:
        """获取最近n条消息"""
        return self.messages[-n:]
    
    def get_all_messages(self) -> List[Dict]:
        """获取所有消息"""
        return self.messages
    
    def clear(self):
        """清空短期记忆"""
        self.messages = []
    
    def get_context_string(self) -> str:
        """获取格式化的上下文字符串"""
        context = []
        for msg in self.messages:
            context.append(f"[{msg['role']}]: {msg['content']}")
        return "\n".join(context)


class LongTermMemory:
    """长期记忆 - 持久化存储关键信息"""
    
    def __init__(self, storage_path: str = "code/memory/long_term_memory.json"):
        self.storage_path = storage_path
        self.memory: Dict = self._load_memory()
    
    def _load_memory(self) -> Dict:
        """从文件加载记忆"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"key_points": [], "arguments": [], "conclusions": []}
        return {"key_points": [], "arguments": [], "conclusions": []}
    
    def _save_memory(self):
        """保存记忆到文件"""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)
    
    def add_key_point(self, point: str, source: str = ""):
        """添加关键点"""
        self.memory["key_points"].append({
            "point": point,
            "source": source,
            "timestamp": datetime.now().isoformat()
        })
        self._save_memory()
    
    def add_argument(self, side: str, argument: str):
        """添加论点"""
        self.memory["arguments"].append({
            "side": side,
            "argument": argument,
            "timestamp": datetime.now().isoformat()
        })
        self._save_memory()
    
    def add_conclusion(self, conclusion: str):
        """添加结论"""
        self.memory["conclusions"].append({
            "conclusion": conclusion,
            "timestamp": datetime.now().isoformat()
        })
        self._save_memory()
    
    def get_key_points(self) -> List[Dict]:
        """获取所有关键点"""
        return self.memory["key_points"]
    
    def get_arguments(self, side: Optional[str] = None) -> List[Dict]:
        """获取论点，可按立场筛选"""
        if side:
            return [arg for arg in self.memory["arguments"] if arg["side"] == side]
        return self.memory["arguments"]
    
    def get_all_memory(self) -> Dict:
        """获取所有长期记忆"""
        return self.memory
    
    def clear(self):
        """清空长期记忆"""
        self.memory = {"key_points": [], "arguments": [], "conclusions": []}
        self._save_memory()
