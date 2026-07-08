"""
对话记忆管理模块
管理每个用户/会话的对话历史，支持多轮对话
支持文件持久化存储，重启后历史不丢失
支持多会话管理：创建、列出、删除、重命名

性能优化:
- 会话存储LRU淘汰：_session_store 限制最大数量，避免内存无限增长
- 自动清理：长时间未访问的会话从内存中移除（文件持久化不受影响）
"""
import os
import json
import uuid
import time
import logging
from collections import OrderedDict
from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from app.config import settings

logger = logging.getLogger(__name__)

# ===== 会话存储上限（防止内存泄漏）=====
# 超过此数量的会话将按 LRU 策略淘汰最久未访问的
# 淘汰仅释放内存中的对象，不影响文件持久化（下次访问时自动重新加载）
MAX_SESSION_STORE_SIZE = 200

# 会话最大空闲时间（秒），超过此时间未访问的会话从内存移除
# [性能修复] 从2小时缩短到30分钟：长时间打开但未操作的会话不应占用内存
SESSION_MAX_IDLE_SECONDS = 1800

# [性能修复] 单个会话最大消息数量，超过时淘汰最早的消息
# 防止长时间运行后消息无限增长导致内存膨胀和序列化变慢
MAX_MESSAGES_PER_SESSION = 200

# [性能修复] 单条消息最大字符数，超过时截断
# 工具输出（搜索结果、文档内容）可能非常长，存入历史会导致内存和序列化开销
MAX_MESSAGE_LENGTH = 8000


class FileBasedHistory(BaseChatMessageHistory):
    """
    基于文件的对话历史存储
    每个会话保存为一个 JSON 文件
    重启后历史不会丢失
    
    [性能修复] 写入防抖：add_message() 不再每次都写磁盘，而是延迟2秒批量写入
    长对话中每次 add_message 都全量序列化+写磁盘，随消息数增加越来越慢
    改为防抖写入：2秒内的多次 add_message 只触发一次磁盘写入
    """

    def __init__(self, session_id: str):
        self._session_id = session_id
        self._messages: list[BaseMessage] = []
        self._file_path = os.path.join(
            settings.DATA_DIR, "conversations", f"{session_id}.json"
        )
        self._dirty = False
        self._save_timer = None
        self._load_from_file()

    def _load_from_file(self):
        """从文件加载历史"""
        if os.path.exists(self._file_path):
            try:
                with open(self._file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for msg_data in data:
                    if msg_data["role"] == "user":
                        self._messages.append(HumanMessage(content=msg_data["content"]))
                    elif msg_data["role"] == "assistant":
                        self._messages.append(AIMessage(content=msg_data["content"]))
            except Exception:
                self._messages = []

    def _save_to_file(self):
        """保存历史到文件"""
        os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
        data = []
        for msg in self._messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # [性能修复] 写入文件时也截断过长内容，防止JSON文件膨胀
            if len(content) > MAX_MESSAGE_LENGTH:
                content = content[:MAX_MESSAGE_LENGTH] + f"\n\n[...内容过长已截断，原文{len(content)}字]"
            data.append({"role": role, "content": content})
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._dirty = False

    def _schedule_save(self):
        """防抖写入：2秒内的多次 add_message 只触发一次磁盘写入"""
        self._dirty = True
        if self._save_timer is not None:
            self._save_timer.cancel()
        import threading
        self._save_timer = threading.Timer(2.0, self._flush_save)
        self._save_timer.daemon = True
        self._save_timer.start()

    def _flush_save(self):
        """执行实际的磁盘写入"""
        if self._dirty:
            try:
                self._save_to_file()
            except Exception as e:
                logger.warning(f"会话 {self._session_id} 保存失败: {e}")

    def flush(self):
        """强制将待写入的数据刷到磁盘（会话清理时调用）"""
        if self._save_timer is not None:
            self._save_timer.cancel()
            self._save_timer = None
        if self._dirty:
            self._save_to_file()

    @property
    def messages(self) -> list[BaseMessage]:
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        # [性能修复] 截断过长的消息内容，防止工具输出占满内存
        content = message.content
        if isinstance(content, str) and len(content) > MAX_MESSAGE_LENGTH:
            truncated = content[:MAX_MESSAGE_LENGTH]
            truncated += f"\n\n[...内容过长已截断，原文{len(content)}字，保留前{MAX_MESSAGE_LENGTH}字]"
            if isinstance(message, HumanMessage):
                message = HumanMessage(content=truncated)
            elif isinstance(message, AIMessage):
                message = AIMessage(content=truncated)
            else:
                message = message.__class__(content=truncated)
        
        self._messages.append(message)
        
        # [性能修复] 超过最大消息数量时，淘汰最早的消息
        while len(self._messages) > MAX_MESSAGES_PER_SESSION:
            self._messages.pop(0)
        
        self._schedule_save()  # [性能修复] 防抖写入替代每次写磁盘

    def clear(self) -> None:
        self._messages = []
        self._dirty = False
        if self._save_timer is not None:
            self._save_timer.cancel()
            self._save_timer = None
        if os.path.exists(self._file_path):
            os.remove(self._file_path)


class InMemoryHistory(BaseChatMessageHistory):
    """
    基于内存的对话历史存储（后备方案）
    """

    def __init__(self):
        self._messages: list[BaseMessage] = []

    @property
    def messages(self) -> list[BaseMessage]:
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)

    def clear(self) -> None:
        self._messages = []


# 全局会话存储：session_id -> (ChatMessageHistory, last_access_time)
# 使用 OrderedDict 实现 LRU 淘汰，避免长时间运行后内存无限增长
_session_store: OrderedDict[str, tuple] = OrderedDict()


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """获取指定会话的对话历史（文件持久化 + LRU淘汰）
    
    LRU策略：每次访问将会话移到OrderedDict末尾，
    超过 MAX_SESSION_STORE_SIZE 时淘汰最久未访问的会话。
    淘汰仅释放内存，不影响磁盘文件（下次访问时自动重新加载）。
    """
    if session_id in _session_store:
        # LRU: 移到末尾（最近访问）
        _session_store.move_to_end(session_id)
        history, _ = _session_store[session_id]
        _session_store[session_id] = (history, time.time())
        return history
    
    # 新会话：加载文件
    try:
        history = FileBasedHistory(session_id)
    except Exception:
        history = InMemoryHistory()
    
    _session_store[session_id] = (history, time.time())
    
    # LRU淘汰：超过上限时移除最久未访问的（先flush防止数据丢失）
    while len(_session_store) > MAX_SESSION_STORE_SIZE:
        oldest_id, (oldest_history, _) = _session_store.popitem(last=False)
        # [BUG FIX] 淘汰前先flush，防止防抖写入中的数据丢失
        if hasattr(oldest_history, 'flush'):
            try:
                oldest_history.flush()
            except Exception:
                pass
        logger.debug(f"LRU淘汰会话: {oldest_id}（内存释放，文件保留）")
    
    return history


def clear_session_history(session_id: str) -> None:
    """清除指定会话的对话历史"""
    if session_id in _session_store:
        history, _ = _session_store[session_id]
        history.clear()
        del _session_store[session_id]


def flush_session(session_id: str) -> None:
    """强制将指定会话的待写入数据刷到磁盘（导出前调用，确保数据一致性）"""
    if session_id in _session_store:
        history, _ = _session_store[session_id]
        if hasattr(history, 'flush'):
            try:
                history.flush()
                logger.debug(f"会话 {session_id} 已flush到磁盘")
            except Exception as e:
                logger.warning(f"会话 {session_id} flush失败: {e}")


def get_history_messages(session_id: str) -> list[dict]:
    """获取会话历史的格式化版本（用于 API 返回）
    
    优先从内存读取（快），如果内存中没有则从文件加载。
    """
    history = get_session_history(session_id)
    messages = []
    for msg in history.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        messages.append({"role": role, "content": content})
    return messages


def get_history_messages_from_file(session_id: str) -> list[dict]:
    """强制从文件读取会话历史（导出时使用，确保跨worker数据一致性）
    
    与 get_history_messages 不同，此函数：
    1. 先flush当前worker的内存缓存到磁盘
    2. 直接从文件重新加载，不使用内存缓存
    3. 确保导出时读到的是最新的持久化数据
    """
    # 先flush当前worker的缓存
    flush_session(session_id)
    
    # 直接从文件重新加载
    file_path = os.path.join(settings.DATA_DIR, "conversations", f"{session_id}.json")
    if not os.path.exists(file_path):
        logger.warning(f"导出时未找到会话文件: {file_path}")
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        messages = []
        for msg_data in data:
            role = msg_data.get("role", "assistant")
            content = msg_data.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            messages.append({"role": role, "content": content})
        logger.info(f"从文件读取会话 {session_id}: {len(messages)} 条消息")
        return messages
    except Exception as e:
        logger.error(f"从文件读取会话 {session_id} 失败: {e}")
        # 降级：使用内存缓存
        return get_history_messages(session_id)


def cleanup_idle_sessions() -> int:
    """清理长时间未访问的会话（释放内存，不影响文件持久化）
    
    定期调用此函数，将超过 SESSION_MAX_IDLE_SECONDS 未访问的会话从内存中移除。
    下次访问时自动从文件重新加载。
    
    [性能修复] 清理前先flush脏数据，防止防抖写入中的数据丢失
    
    Returns:
        int: 被清理的会话数量
    """
    now = time.time()
    to_remove = []
    for sid, (history, last_access) in _session_store.items():
        if now - last_access > SESSION_MAX_IDLE_SECONDS:
            to_remove.append(sid)
    
    for sid in to_remove:
        history, _ = _session_store[sid]
        # [性能修复] 清理前先flush，确保防抖写入中未持久化的数据不丢失
        if hasattr(history, 'flush'):
            try:
                history.flush()
            except Exception:
                pass
        del _session_store[sid]
    
    if to_remove:
        logger.info(f"清理了 {len(to_remove)} 个空闲超过 {SESSION_MAX_IDLE_SECONDS}s 的会话（内存释放，文件保留）")
    
    return len(to_remove)


# ===== 多会话管理 =====

def _get_user_chats_file(username: str) -> str:
    """获取用户的会话索引文件路径"""
    return os.path.join(settings.DATA_DIR, "users", f"{username}_chats.json")


def _load_user_chats(username: str) -> list[dict]:
    """加载用户的会话列表"""
    chats_file = _get_user_chats_file(username)
    if os.path.exists(chats_file):
        try:
            with open(chats_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_user_chats(username: str, chats: list[dict]) -> None:
    """保存用户的会话列表"""
    chats_file = _get_user_chats_file(username)
    os.makedirs(os.path.dirname(chats_file), exist_ok=True)
    with open(chats_file, "w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=2)


# 每个智能体最多保留的历史对话数量
MAX_CHATS_PER_AGENT = 2


def create_chat(username: str, title: str = "新对话", mode: str = "agent", agent_id: str = None) -> dict:
    """
    为用户创建一个新的会话

    Args:
        mode: 会话模式 "agent" 或 "chat"，用于隔离不同模式的对话列表
        agent_id: 智能体ID，会话归属到指定智能体

    Returns:
        dict: 包含 chat_id、title 和 mode

    当同一智能体下的会话数超过 MAX_CHATS_PER_AGENT 时，自动删除最老的会话
    """
    chat_id = f"{username}_{uuid.uuid4().hex[:8]}"
    chats = _load_user_chats(username)

    # 自动淘汰：如果指定了 agent_id，检查该智能体下已有会话数
    if agent_id:
        agent_chats = [c for c in chats if c.get("agent_id") == agent_id]
        if len(agent_chats) >= MAX_CHATS_PER_AGENT:
            # 按 updated_at 升序排列，最老的在前面
            agent_chats_sorted = sorted(agent_chats, key=lambda x: x.get("updated_at", 0))
            to_remove_count = len(agent_chats) - MAX_CHATS_PER_AGENT + 1  # +1 因为还要新建一个
            for old_chat in agent_chats_sorted[:to_remove_count]:
                old_chat_id = old_chat["chat_id"]
                chats = [c for c in chats if c["chat_id"] != old_chat_id]
                # 同时清理对话历史文件和内存缓存
                clear_session_history(old_chat_id)
                logger.info(f"智能体 [{agent_id}] 会话数达上限({MAX_CHATS_PER_AGENT})，自动淘汰旧会话: {old_chat_id}")

    chat_info = {
        "chat_id": chat_id,
        "title": title,
        "mode": mode,
        "agent_id": agent_id or "",
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    chats.insert(0, chat_info)  # 新会话放在最前面
    _save_user_chats(username, chats)

    return chat_info


def list_chats(username: str, mode: str = None, skip_auto_title: bool = False) -> list[dict]:
    """列出用户的会话，按更新时间倒序

    Args:
        mode: 可选，按模式过滤 "agent" 或 "chat"，为 None 则返回全部
        skip_auto_title: 跳过自动标题更新（GET请求时为True，避免读副作用）
    """
    chats = _load_user_chats(username)

    # 自动更新标题：仅在需要时执行（非GET请求或显式请求时）
    if not skip_auto_title:
        updated = False
        for chat in chats:
            chat_id = chat["chat_id"]
            if not chat.get("title_custom"):
                history = get_session_history(chat_id)
                if history.messages:
                    for msg in history.messages:
                        if isinstance(msg, HumanMessage):
                            title = msg.content[:30].replace("\n", " ")
                            if len(msg.content) > 30:
                                title += "..."
                            chat["title"] = title
                            updated = True
                            break
        # 只有标题确实变更了才写回
        if updated:
            chats.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
            _save_user_chats(username, chats)
            if mode:
                chats = [c for c in chats if c.get("mode", "agent") == mode]
            return chats

    # 按更新时间倒序（不写回文件）
    chats.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
    # 按 mode 过滤（如果指定了 mode）
    if mode:
        # 兼容旧数据：没有 mode 字段的会话默认归为 "agent"
        chats = [c for c in chats if c.get("mode", "agent") == mode]
    return chats


def delete_chat(username: str, chat_id: str) -> bool:
    """删除用户的某个会话"""
    chats = _load_user_chats(username)
    chats = [c for c in chats if c["chat_id"] != chat_id]
    _save_user_chats(username, chats)
    # 同时清除对话历史文件
    clear_session_history(chat_id)
    return True


def rename_chat(username: str, chat_id: str, new_title: str) -> bool:
    """重命名用户的某个会话"""
    chats = _load_user_chats(username)
    for chat in chats:
        if chat["chat_id"] == chat_id:
            chat["title"] = new_title
            chat["title_custom"] = True
            chat["updated_at"] = time.time()
            break
    _save_user_chats(username, chats)
    return True


# [性能修复] 用户聊天列表写入防抖：避免每条消息都触发磁盘读写
# 缓存：username -> (chats_data, last_save_time, dirty_flag)
_user_chats_cache: dict = {}
_USER_CHATS_SAVE_INTERVAL = 5.0  # 至少间隔5秒才写一次磁盘
import threading as _threading
_user_chats_cache_lock = _threading.Lock()  # [BUG FIX] 防止并发写入导致数据丢失


def update_chat_time(username: str, chat_id: str) -> None:
    """更新会话的更新时间（发送消息时调用）
    
    [性能修复] 使用内存缓存+防抖写入，避免每条消息都读写磁盘
    """
    chats = _load_user_chats(username)
    for chat in chats:
        if chat["chat_id"] == chat_id:
            chat["updated_at"] = time.time()
            # 自动更新标题（取第一条用户消息）
            if not chat.get("title_custom"):
                history = get_session_history(chat_id)
                for msg in history.messages:
                    if isinstance(msg, HumanMessage):
                        title = msg.content[:30].replace("\n", " ")
                        if len(msg.content) > 30:
                            title += "..."
                        chat["title"] = title
                        break
            break
    
    # [性能修复] 防抖写入：检查距上次写入是否超过间隔
    now = time.time()
    with _user_chats_cache_lock:  # [BUG FIX] 加锁防止并发写入
        cache_entry = _user_chats_cache.get(username)
        if cache_entry is None:
            _save_user_chats(username, chats)
            _user_chats_cache[username] = (chats, now, False)
        else:
            _, last_save, _ = cache_entry
            if now - last_save >= _USER_CHATS_SAVE_INTERVAL:
                _save_user_chats(username, chats)
                _user_chats_cache[username] = (chats, now, False)
            else:
                _user_chats_cache[username] = (chats, last_save, True)


def flush_user_chats_cache():
    """将所有脏缓存刷到磁盘（定期清理时调用）"""
    with _user_chats_cache_lock:  # [BUG FIX] 加锁防止并发
        for username, (chats, last_save, dirty) in _user_chats_cache.items():
            if dirty:
                try:
                    _save_user_chats(username, chats)
                    _user_chats_cache[username] = (chats, time.time(), False)
                except Exception as e:
                    logger.warning(f"flush用户聊天缓存失败 [{username}]: {e}")
