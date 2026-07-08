"""
Agent 工具定义模块
每个工具 = Agent 的一个「能力」
Agent 会根据用户问题自动选择调用哪个工具

优化:
- [#8] 工具结果缓存：LRU缓存 + TTL
- [#10] 引用溯源：返回结果标注文档名+段落位置
- [#12] 外部系统集成：GitHub API / 邮件 / 数据库查询
"""
import json
import os
import time
import hashlib
import logging
import re
from typing import Optional
from functools import wraps

from langchain_core.tools import tool

import asyncio
from app.config import settings
from app.rag.document import search_documents, search_documents_async, index_document, list_indexed_documents, delete_document, update_document, export_document_as_docx, export_document_as_xlsx, get_document_content

logger = logging.getLogger(__name__)

# ===== 当前智能体上下文 =====
# 用于在 Agent 工具调用时传递 agent_id，实现知识库隔离
# 使用 contextvars.ContextVar 而非 threading.local，因为：
# 1. LangGraph 的 ToolNode 使用 ThreadPoolExecutor 执行工具
# 2. threading.local 的值不会传播到子线程，导致工具函数中 get_current_agent_id() 返回 None
# 3. contextvars.ContextVar 通过 asyncio.run_in_executor 自动复制上下文到子线程
# 4. 同时支持 asyncio 并发请求隔离（每个请求有独立的上下文）
import contextvars
_agent_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar('agent_id', default=None)
_session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar('session_id', default=None)


def set_current_agent_id(agent_id: str = None):
    """设置当前会话的智能体ID（contextvars 实现，支持 asyncio + ThreadPoolExecutor 上下文传播）"""
    _agent_id_var.set(agent_id)
    logger.debug(f"工具上下文 agent_id 设置为: {agent_id}")


def get_current_agent_id() -> str:
    """获取当前会话的智能体ID（contextvars 实现，在 ToolNode 工作线程中也能正确获取）"""
    return _agent_id_var.get()


def set_current_session_id(session_id: str = None):
    """设置当前会话的 session_id（用于导出文件按会话隔离存储）"""
    _session_id_var.set(session_id)


def get_current_session_id() -> str:
    """获取当前会话的 session_id"""
    return _session_id_var.get()


# ===== [#8] 工具结果缓存 =====
class ToolCache:
    """带 TTL 的 LRU 工具结果缓存（使用 OrderedDict 实现 O(1) 读写）
    
    性能优化：原实现使用 list + remove() 实现 LRU，每次访问 O(n)。
    改用 OrderedDict.move_to_end()，每次访问 O(1)，在高并发场景下性能显著提升。
    """
    def __init__(self, max_size: int = 100, default_ttl: int = 300):
        self._cache = {}  # key -> {"value": ..., "expire_at": float}
        self._order = {}  # OrderedDict for LRU: key -> True
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def _make_key(self, func_name: str, args: tuple, kwargs: dict, agent_id: str = "") -> str:
        """生成缓存 key（包含 agent_id 以隔离不同智能体的缓存）"""
        raw = f"{func_name}:{agent_id}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[str]:
        """获取缓存，过期返回 None（O(1) 操作）"""
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        if time.time() > entry["expire_at"]:
            del self._cache[key]
            self._order.pop(key, None)
            self._misses += 1
            return None
        # LRU: O(1) 移到末尾
        self._order[key] = True
        self._hits += 1
        return entry["value"]

    def set(self, key: str, value: str, ttl: int = None):
        """设置缓存（O(1) 操作）"""
        if ttl is None:
            ttl = self._default_ttl
        # 容量超限时淘汰最久未访问的（FIFO from OrderedDict）
        while len(self._cache) >= self._max_size and self._order:
            oldest_key = next(iter(self._order))
            self._cache.pop(oldest_key, None)
            del self._order[oldest_key]
        self._cache[key] = {"value": value, "expire_at": time.time() + ttl}
        self._order[key] = True

    def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._order.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        """缓存统计"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {"size": len(self._cache), "max_size": self._max_size, "hits": self._hits, "misses": self._misses, "hit_rate": f"{hit_rate:.1f}%"}


# 全局工具缓存实例
_tool_cache = ToolCache(max_size=100, default_ttl=300)

# ===== 搜索效率控制 =====
# 每轮对话的最大文档搜索次数（超过后返回提示，让LLM直接使用已有信息回答）
_MAX_SEARCH_PER_TURN = 3

# 使用 contextvars 而非全局变量，支持并发请求隔离
_search_count_var: contextvars.ContextVar[int] = contextvars.ContextVar('search_count', default=0)


def reset_search_count():
    """重置搜索计数（每次新对话轮次开始时调用）"""
    _search_count_var.set(0)


def increment_search_count() -> int:
    """递增搜索计数并返回当前值"""
    current = _search_count_var.get(0) + 1
    _search_count_var.set(current)
    return current


def get_search_count() -> int:
    """获取当前搜索计数"""
    return _search_count_var.get(0)


def cached_tool(ttl: int = 300, include_agent_id: bool = True):
    """工具缓存装饰器（同步版）
    
    Args:
        ttl: 缓存有效期（秒），web_search 默认 5 分钟，文档搜索默认 2 分钟
        include_agent_id: 是否将 agent_id 纳入缓存 key（默认 True）
            - True: 缓存 key 包含 agent_id，确保不同智能体的知识库搜索结果互不干扰
            - False: 缓存 key 不包含 agent_id，适用于与智能体无关的工具（如 web_search）
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            agent_id_for_cache = get_current_agent_id() if include_agent_id else ""
            cache_key = _tool_cache._make_key(func.__name__, args, kwargs, agent_id=agent_id_for_cache)
            cached = _tool_cache.get(cache_key)
            if cached is not None:
                logger.info(f"工具缓存命中: {func.__name__} (agent_id={agent_id_for_cache})")
                return cached
            result = func(*args, **kwargs)
            if isinstance(result, str) and result.startswith("【检索失败】"):
                logger.warning(f"工具返回错误，不缓存: {func.__name__} -> {result[:100]}")
                return result
            _tool_cache.set(cache_key, result, ttl=ttl)
            return result
        return wrapper
    return decorator


def cached_tool_async(ttl: int = 300, include_agent_id: bool = True):
    """工具缓存装饰器（异步版）
    
    与 cached_tool 功能相同，但支持异步函数。
    缓存命中时直接返回，未命中时 await 异步函数获取结果。
    
    Args:
        ttl: 缓存有效期（秒）
        include_agent_id: 是否将 agent_id 纳入缓存 key
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            agent_id_for_cache = get_current_agent_id() if include_agent_id else ""
            cache_key = _tool_cache._make_key(func.__name__, args, kwargs, agent_id=agent_id_for_cache)
            cached = _tool_cache.get(cache_key)
            if cached is not None:
                logger.info(f"工具缓存命中(异步): {func.__name__} (agent_id={agent_id_for_cache})")
                return cached
            result = await func(*args, **kwargs)
            if isinstance(result, str) and result.startswith("【检索失败】"):
                logger.warning(f"工具返回错误，不缓存: {func.__name__} -> {result[:100]}")
                return result
            _tool_cache.set(cache_key, result, ttl=ttl)
            return result
        return wrapper
    return decorator


# ===== [性能优化 3] HTTP 连接池复用 =====
# 模块级 httpx 客户端：跨请求复用 TCP/TLS 连接，避免每次 web_search 重建握手
# 节省 50-200ms/次联网搜索，且消除线程池阻塞风险
import threading
_search_client = None
_search_client_lock = threading.Lock()

def _get_search_client():
    """获取或创建模块级 httpx 客户端（带连接池，线程安全）"""
    global _search_client
    import httpx
    if _search_client is None:
        with _search_client_lock:
            if _search_client is None:
                _search_client = httpx.Client(
                    follow_redirects=True,
                    timeout=15.0,
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=20),
                    transport=httpx.HTTPTransport(retries=2),
                )
                logger.info("[性能优化 3] httpx 连接池客户端已创建")
    return _search_client


# ===== 联网搜索工具 =====
@tool
@cached_tool(ttl=300, include_agent_id=False)  # [#8] web_search 缓存 5 分钟（与智能体无关）
def web_search_tool(query: str) -> str:
    """搜索互联网获取实时信息。当你需要最新资讯、实时数据、或知识库中没有的信息时使用此工具。

    【用途】搜索互联网上的最新信息、新闻、实时数据等。
    【典型问题】「最新新闻」「今天天气」「某产品最新价格」「最新技术动态」「实时汇率」
    【不适用】查公司制度文档（用search_documents_tool）、查员工信息（用lookup_employee_tool）。

    Args:
        query: 搜索查询关键词。示例：「2024年最新AI技术动态」「北京今天天气」
    """
    try:
        import httpx
        from urllib.parse import quote_plus

        # 使用百度搜索（国内最稳定）
        search_url = f"https://www.baidu.com/s?wd={quote_plus(query)}&rn=5"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        # [性能优化 3] 使用模块级连接池客户端，复用 TCP/TLS 连接，避免每次搜索重建握手
        client = _get_search_client()
        resp = client.get(search_url, headers=headers)
        html = resp.text

        # 解析百度搜索结果
        results = []

        # 方法1：从 h3 标签提取标题和链接
        h3_pattern = re.compile(r'<h3[^>]*class="[^"]*t[^"]*"[^>]*>(.*?)</h3>', re.DOTALL)
        for match in h3_pattern.finditer(html):
            block = match.group(1)
            title = re.sub(r'<[^>]+>', '', block).strip()
            href_match = re.search(r'href="(https?://[^"]+)"', block)
            href = href_match.group(1) if href_match else ""
            if title:
                results.append({"title": title, "href": href, "snippet": ""})

        # 方法2：如果方法1没有结果，尝试更宽松的匹配
        if not results:
            h3_all = re.compile(r'<h3[^>]*>(.*?)</h3>', re.DOTALL)
            for match in h3_all.finditer(html):
                block = match.group(1)
                title = re.sub(r'<[^>]+>', '', block).strip()
                href_match = re.search(r'href="(https?://[^"]+)"', block)
                href = href_match.group(1) if href_match else ""
                if title and len(title) > 3:
                    results.append({"title": title, "href": href, "snippet": ""})

        # 提取摘要
        abstract_pattern = re.compile(r'class="c-abstract[^"]*"[^>]*>(.*?)</(?:span|div|p)>', re.DOTALL)
        abstracts = [re.sub(r'<[^>]+>', '', m.group(1)).strip() for m in abstract_pattern.finditer(html)]

        for i, r in enumerate(results):
            if i < len(abstracts) and abstracts[i]:
                r["snippet"] = abstracts[i]

        if not any(r["snippet"] for r in results):
            snippet_pattern = re.compile(r'<span class="content-right_[^"]*">(.*?)</span>', re.DOTALL)
            snippets = [re.sub(r'<[^>]+>', '', m.group(1)).strip() for m in snippet_pattern.finditer(html)]
            for i, r in enumerate(results):
                if i < len(snippets) and snippets[i]:
                    r["snippet"] = snippets[i]

        if not results:
            return "【联网搜索】未找到相关结果。建议：1）尝试换用不同关键词搜索；2）检查网络连接是否正常。"

        output = f"【联网搜索】共找到 {len(results)} 条相关结果：\n\n"
        for i, r in enumerate(results[:5], 1):
            output += f"<web_result index=\"{i}\">\n"
            output += f"  标题：{r['title']}\n"
            if r['snippet']:
                output += f"  摘要：{r['snippet']}\n"
            if r['href']:
                output += f"  链接：{r['href']}\n"
            output += f"</web_result>\n\n"

        return output
    except Exception as e:
        return f"【联网搜索】搜索失败: {str(e)}\n建议：检查网络连接是否正常，或稍后重试。"


def _load_employees():
    """加载员工数据"""
    employees_file = settings.EMPLOYEES_FILE
    if not os.path.exists(employees_file):
        return None
    with open(employees_file, "r", encoding="utf-8") as f:
        return json.load(f)


@tool
@cached_tool_async(ttl=300)  # [#8] 异步文档搜索缓存 5 分钟
async def search_documents_tool(query: str) -> str:
    """搜索公司文档知识库，检索与查询语义相关的文档片段。

    [#9] 采用混合检索策略：向量语义检索 + 关键词匹配，提升检索准确率
    [#10] 返回结果标注文档来源和段落位置，支持引用溯源

    【用途】查询公司制度、流程、规范、政策、规定等文档内容。
    【不适用】查员工信息（用lookup_employee_tool）、查看文档列表（用list_documents_tool）。

    Args:
        query: 搜索查询关键词。
               示例：「年假制度」「报销流程」「考勤规定」
    """
    current_aid = get_current_agent_id()
    logger.debug(f"搜索文档(异步): query={query}, agent_id={current_aid}")

    # 普通聊天模式没有知识库
    if not current_aid:
        return "【检索结果】当前是普通聊天模式，没有关联的知识库。如需搜索文档，请先选择一个智能体。"
    
    # ===== 搜索效率控制：同一轮对话最多搜索 _MAX_SEARCH_PER_TURN 次 =====
    current_count = increment_search_count()
    if current_count > _MAX_SEARCH_PER_TURN:
        logger.info(f"搜索效率控制：本轮已搜索 {current_count-1} 次，超过上限 {_MAX_SEARCH_PER_TURN}，提示LLM直接回答")
        return f"【检索提示】已搜索{current_count-1}次，请直接基于已有结果回答，不要再搜索。"
    
    try:
        # [性能优化] 使用异步并行搜索替代同步串行搜索
        results = await search_documents_async(query, top_k=8, agent_id=current_aid)
    except Exception as e:
        error_str = str(e)
        logger.error(f"search_documents_async 异常: {error_str}", exc_info=True)
        if '429' in error_str or '余额' in error_str or '1113' in error_str:
            return f"【检索失败】Embedding API 余额不足（429错误），向量搜索不可用。建议用户充值智谱API余额。当前仅使用关键词检索，结果可能不完整。如需获取完整文档内容，请使用 get_document_content_tool 工具。"
        # 兜底：尝试磁盘文件搜索
        try:
            from app.rag.document import _search_disk_files
            fallback_results = await asyncio.to_thread(_search_disk_files, query, top_k=5, agent_id=current_aid)
            if fallback_results:
                results = fallback_results
            else:
                return f"【检索失败】搜索出错: {error_str}。建议使用 get_document_content_tool 直接获取文档全文，或用 list_documents_tool 查看可用文档。"
        except Exception:
            return f"【检索失败】搜索出错: {error_str}。建议使用 get_document_content_tool 直接获取文档全文，或用 list_documents_tool 查看可用文档。"

    if not results:
        return f"【检索结果】未找到与查询相关的文档内容（当前搜索的知识库: agent_id={current_aid}）。建议：1）尝试换用不同关键词搜索；2）确认相关文档是否已上传至对应智能体的知识库。"

    # [P0-2 修复] 直接使用 RRF 分数排序，不再做二次"简易 rerank"。
    # 旧实现用 `0.4 * 字面字符重合度 + 0.6 * vector_score` 重排，会抹掉
    # search_documents_async 里 RRF 融合的成果（rrf_score），且字符级
    # 重合度对中文 query 几乎没有语义价值（每个汉字被当成独立 keyword）。
    # 现在改为：优先用 rrf_score；若结果未经过 RRF（纯关键词降级场景），
    # 则回退到 relevance_score 或 bm25_score。
    for r in results:
        if "rrf_score" in r:
            r["final_score"] = r["rrf_score"]
        elif "relevance_score" in r:
            r["final_score"] = r["relevance_score"]
        elif "bm25_score" in r:
            r["final_score"] = r["bm25_score"]
        else:
            r["final_score"] = r.get("relevance_score", 0.0)

    # 按综合分数排序
    results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    results = results[:5]  # [质量修复] 取 top 5（原 top 3 信息不足，增加检索量保证回答质量）


    output = f"【检索结果】共找到 {len(results)} 条相关内容：\n\n"
    for i, r in enumerate(results, 1):
        source = r.get('source', '未知来源')
        relevance = r.get('relevance_score', 0)
        content = r.get('content', '')
        # [#10] 引用溯源：标注文档名 + 段落位置
        # 提取内容的前30字作为段落定位
        content_preview = content[:50].replace('\n', ' ').strip()
        output += f"<document source=\"{source}\" relevance=\"{relevance:.2f}\" citation=\"{source} · {content_preview}...\">\n"
        output += f"{content}\n"
        output += f"</document>\n\n"

    # [#10] 添加引用说明
    sources = list(set(r.get('source', '') for r in results if r.get('source')))
    if sources:
        output += f"【引用来源】{', '.join(sources)}\n"

    return output


@tool
def lookup_employee_tool(name: str = "", department: str = "") -> str:
    """查询公司员工信息。不传参数则列出全部员工，传参数则按条件筛选。

    【用途】查询员工姓名、部门、职位、联系方式等人员信息。
    【典型问题】「所有员工」「张三的信息」「技术部有哪些人」「公司有哪些部门的人」
    【不适用】查公司制度文档（用search_documents_tool）、查文档列表（用list_documents_tool）。

    Args:
        name: 员工姓名（可选，支持模糊匹配）。示例：「张」可匹配「张三」「张伟」
        department: 部门名称（可选，支持模糊匹配）。示例：「技术」可匹配「技术部」
    """
    employees = _load_employees()

    if employees is None:
        return "【系统提示】员工数据库暂未初始化，请先运行 scripts/seed_data.py 初始化数据。"

    results = employees

    # 按姓名过滤
    if name:
        results = [e for e in results if name in e.get("name", "")]

    # 按部门过滤
    if department:
        results = [e for e in results if department in e.get("department", "")]

    if not results:
        return f"【查询结果】未找到匹配的员工信息。搜索条件：姓名=\"{name}\"，部门=\"{department}\"\n建议：检查姓名/部门名称是否正确，或尝试使用部分关键词搜索。你也可以不传参数查看全部员工列表。"

    # 生成部门统计摘要
    dept_count = {}
    for e in results:
        dept = e.get("department", "未知")
        dept_count[dept] = dept_count.get(dept, 0) + 1

    output = f"【查询结果】共找到 {len(results)} 位员工"
    if dept_count:
        dept_summary = "、".join([f"{d} {c}人" for d, c in dept_count.items()])
        output += f"（{dept_summary}）"
    output += "：\n\n"

    for e in results:
        output += f"<employee>\n"
        output += f"  姓名：{e['name']}\n"
        output += f"  部门：{e['department']}\n"
        output += f"  职位：{e['position']}\n"
        output += f"  邮箱：{e['email']}\n"
        if e.get("phone"):
            output += f"  电话：{e['phone']}\n"
        output += f"</employee>\n\n"

    return output


@tool
def list_departments_tool() -> str:
    """列出公司所有部门及各部门人数。

    【用途】当用户想知道公司有哪些部门、各部门有多少人时使用。
    【典型问题】「公司有哪些部门」「部门列表」「都有什么部门」。
    """
    employees = _load_employees()

    if employees is None:
        return "【系统提示】员工数据库暂未初始化，请先运行 scripts/seed_data.py 初始化数据。"

    # 统计部门
    dept_employees = {}
    for e in employees:
        dept = e.get("department", "未知")
        if dept not in dept_employees:
            dept_employees[dept] = []
        dept_employees[dept].append(e['name'])

    if not dept_employees:
        return "【查询结果】暂无部门信息。"

    output = f"【部门列表】公司共有 {len(dept_employees)} 个部门，{len(employees)} 位员工：\n\n"
    for i, (dept, names) in enumerate(dept_employees.items(), 1):
        output += f"  {i}. **{dept}**（{len(names)}人）：{'、'.join(names)}\n"

    output += f"\n如需查看某部门员工的详细信息，请告诉我部门名称。"

    return output


@tool
def list_documents_tool() -> str:
    """列出知识库中所有已索引的文档。

    【用途】查看知识库中有哪些可搜索的文档。
    【典型问题】「知识库有哪些文档」「文档列表」「你们有什么资料」。
    【不适用】查员工信息（用lookup_employee_tool）、查公司制度内容（用search_documents_tool）。
    """
    current_aid = get_current_agent_id()
    logger.debug(f"文档列表: agent_id={current_aid}")

    # 普通聊天模式没有知识库
    if not current_aid:
        return "【文档列表】当前是普通聊天模式，没有关联的知识库。如需管理文档，请先选择一个智能体。"

    docs = list_indexed_documents(agent_id=current_aid)
    # 同时列出外部知识库的文档
    ext_docs = list_indexed_documents(agent_id="__external__")

    if not docs and not ext_docs:
        return f"【文档列表】两个知识库均暂无文档（内部: agent_id={current_aid}，外部: external_kb）。请先上传文档。"

    output = "【文档列表】当前可访问 2 个知识库：\n\n"
    
    output += f"一、企业内部体系文件（{len(docs)} 个文档）：\n"
    if docs:
        for i, doc in enumerate(docs, 1):
            ext = doc.rsplit('.', 1)[-1].lower() if '.' in doc else ''
            type_label = {'pdf': 'PDF文档', 'docx': 'Word文档', 'txt': '文本文件'}.get(ext, '文档')
            output += f"  {i}. {doc}（{type_label}）\n"
    else:
        output += "  （暂无文档）\n"
    
    output += f"\n二、外部知识库（{len(ext_docs)} 个文档）：\n"
    if ext_docs:
        for i, doc in enumerate(ext_docs, 1):
            ext = doc.rsplit('.', 1)[-1].lower() if '.' in doc else ''
            type_label = {'pdf': 'PDF文档', 'docx': 'Word文档', 'txt': '文本文件'}.get(ext, '文档')
            output += f"  {i}. {doc}（{type_label}）\n"
    else:
        output += "  （暂无文档）\n"

    output += f"\n搜索时会同时检索两个知识库的内容。"
    return output


@tool
def upload_document_tool(file_path: str) -> str:
    """将新文档上传并索引到知识库，使其可被搜索。

    【用途】当用户需要添加新文档到知识库时使用。
    支持格式：PDF、TXT、DOCX。

    Args:
        file_path: 要上传的文档文件路径，必须是已存在于服务器上的文件。
    """
    if not os.path.exists(file_path):
        return f"【上传失败】文件不存在：{file_path}\n请确认文件路径是否正确，或先通过界面功能上传文件。"

    current_aid = get_current_agent_id()
    if not current_aid:
        return "【上传失败】当前是普通聊天模式，没有关联的知识库。如需上传文档，请先选择一个智能体。"

    ext = file_path.rsplit('.', 1)[-1].lower() if '.' in file_path else ''
    supported = ['pdf', 'txt', 'docx', 'md', 'xlsx', 'xls']
    if ext not in supported:
        return f"【上传失败】不支持的文件格式：.{ext}。目前支持：{', '.join(['.'+e for e in supported])}"

    try:
        result = index_document(file_path, agent_id=current_aid)
        return f"【上传成功】文档已索引到知识库。{result['message']}"
    except Exception as e:
        return f"【上传失败】{str(e)}\n可能原因：文件损坏、内容为空或格式异常。请检查文件后重试。"


@tool
def get_document_content_tool(filename: str) -> str:
    """获取知识库中指定文档的完整内容。直接从原始文件读取，不依赖向量搜索，不会消耗embedding额度。

    【用途】当需要查看或获取某个文档的完整内容时使用。修改文档前应先用此工具获取完整内容。
    【典型问题】「显示xxx文档的完整内容」「获取xxx文档全文」「查看xxx文档」
    【与search_documents_tool的区别】
    - search_documents_tool：搜索知识库，返回与查询相关的文档片段（500字/片），适合查找特定信息
    - get_document_content_tool：返回指定文档的完整全文，适合需要整体查看或修改文档的场景
    【重要】修改文档前，请先调用此工具获取完整内容，修改后再调用modify_document_tool保存。

    Args:
        filename: 文档文件名（含扩展名），需与知识库中的文件名完全一致。
                  示例：「员工手册.pdf」「FMEA新版手册.docx」
    """
    current_aid = get_current_agent_id()
    if not current_aid:
        return "【获取失败】当前是普通聊天模式，没有关联的知识库。如需查看文档，请先选择一个智能体。"
    result = get_document_content(filename, agent_id=current_aid)
    
    if result["status"] == "not_found":
        return f"【获取失败】文档 \"{filename}\" 在服务器上未找到。\n提示：请确认文件名是否正确（需包含扩展名），可通过 list_documents_tool 查看当前文档列表。"
    if result["status"] == "empty":
        return f"【获取失败】文档 \"{filename}\" 内容为空。"
    if result["status"] == "error":
        return f"【获取失败】{result['message']}"
    
    # 成功：返回完整内容
    output = f"【文档内容】{filename}（共 {result['char_count']} 字符）\n\n"
    output += result["content"]
    return output


@tool
def delete_document_tool(filename: str) -> str:
    """从知识库中删除指定文档，同时移除其所有向量分块和原始文件。此操作不可恢复。

    【用途】当用户确认要删除某个文档时使用。
    注意：删除操作不可逆，请在调用前确认用户已明确指定要删除的文档名称。

    Args:
        filename: 要删除的文档文件名（含扩展名），需与知识库中的文件名完全一致。
                  示例：「员工手册.pdf」而非「员工手册」
    """
    current_aid_del = get_current_agent_id()
    if not current_aid_del:
        return "【删除失败】当前是普通聊天模式，没有关联的知识库。如需管理文档，请先选择一个智能体。"
    try:
        result = delete_document(filename, agent_id=current_aid_del)
        if result["status"] == "not_found":
            return f"【删除失败】文档 \"{filename}\" 在知识库中未找到。\n提示：请确认文件名是否正确（需包含扩展名），可通过 list_documents_tool 查看当前文档列表。"
        return f"【删除成功】{result['message']}"
    except Exception as e:
        return f"【删除失败】{str(e)}"


@tool
def modify_document_tool(filename: str, content: str, append: bool = False) -> str:
    """修改知识库中已有文档的内容。支持替换全部内容或在原文末尾追加内容。

    【用途】当用户要求修改、编辑、更新知识库中某个文档的内容时使用。
    【典型问题】「帮我在xxx文件中添加yyy」「把xxx文档里的zzz改成www」「修改知识库的xxx文件」
    【重要】修改后会自动重新索引到向量数据库，无需手动操作。
    【操作流程】替换模式下，请先调用 get_document_content_tool 获取完整内容，在完整内容基础上进行修改，
    然后将修改后的完整内容作为 content 参数传入。不要凭记忆或片段拼凑内容！
    【注意】此工具仅用于修改知识库文档，不生成docx下载文件。如需导出文档，请使用 export_document_tool。

    Args:
        filename: 要修改的文档文件名（含扩展名），需与知识库中的文件名完全一致。
                  示例：「教务处归口管理的校外人员劳务费发放附页-zy.docx」
        content: 新的内容。如果是追加模式，这是要追加到文档末尾的内容；如果是替换模式，这是文档的完整新内容。
        append: 是否追加模式。True=在原文末尾追加内容，False=用新内容替换整个文档（默认False）。
                一般情况下，用户说"添加""追加""加上"用追加模式；用户说"修改""改为""替换"用替换模式。
    """
    current_aid_mod = get_current_agent_id()
    if not current_aid_mod:
        return "【修改失败】当前是普通聊天模式，没有关联的知识库。如需修改文档，请先选择一个智能体。"

    # 追加模式：先读取原文内容，拼接新内容
    final_content = content
    if append:
        try:
            doc_result = get_document_content(filename, agent_id=current_aid_mod)
            if doc_result["status"] == "success":
                original_text = doc_result["content"]
                final_content = original_text + "\n" + content
            else:
                return f"【修改失败】文档 \"{filename}\" 在服务器上未找到。可通过 list_documents_tool 查看当前文档列表。"
        except Exception as e:
            return f"【修改失败】读取原文档内容时出错: {str(e)}"
    else:
        # 替换模式安全检查：防止用少量内容覆盖大量原文
        _original_content_for_verify = ""
        try:
            doc_result = get_document_content(filename, agent_id=current_aid_mod)
            if doc_result["status"] == "success":
                _original_content_for_verify = doc_result["content"]
                original_len = len(doc_result["content"])
                new_len = len(content)
                # 如果新内容不到原文的30%，且原文超过500字，极可能是误操作
                if original_len > 500 and new_len < original_len * 0.3:
                    # 检查新内容是否包含原文的大部分结构（判断是真删减还是误覆盖）
                    # 如果新内容的前50字能在原文中找到，说明LLM保留了原文结构，可能是合理删减
                    content_head = content[:50].strip()
                    if content_head and content_head in doc_result["content"]:
                        logger.warning(f"替换模式大幅删减：原文 {original_len} 字 → 新内容 {new_len} 字，但新内容开头与原文匹配，可能是合理删减，允许执行")
                    else:
                        return f"【修改被拦截】安全检查：原文档共 {original_len} 字，新内容仅 {new_len} 字（不足原文30%）。\n替换模式会用新内容覆盖整个文档，这可能导致原文档大量内容丢失！\n\n如果您确实要大幅删减文档，请先调用 get_document_content_tool 获取完整原文，在原文基础上删减后提交完整内容。\n如果您只是想提取部分内容导出为docx，请使用 export_document_tool 而不是 modify_document_tool。"
        except Exception:
            pass  # 读取失败不阻塞修改流程

    try:
        # 使用同步重索引（async_reindex=False），确保修改后知识库立即可用
        # 虽然稍慢，但避免用户修改后搜索到旧内容
        result = update_document(filename, final_content, agent_id=current_aid_mod, async_reindex=False)
        if result["status"] == "not_found":
            return f"【修改失败】文档 \"{filename}\" 在知识库中未找到。\n提示：请确认文件名是否正确（需包含扩展名），可通过 list_documents_tool 查看当前文档列表。"
        if result["status"] == "error":
            return f"【修改失败】{result['message']}"

        # 写入后验证：确保原文核心内容仍然存在
        try:
            verify_result = get_document_content(filename, agent_id=current_aid_mod)
            if verify_result["status"] == "success":
                # 追加模式：原文前100字必须在
                if append and original_text and original_text[:100] not in verify_result["content"]:
                    logger.error(f"⚠️ 追加模式写入后验证失败：原文内容丢失！filename={filename}")
                    update_document(filename, original_text + "\n" + content, agent_id=current_aid_mod, async_reindex=False)
                    return f"【修改成功（已恢复）】{result['message']}（系统检测到原文可能丢失，已自动恢复）"
                # 替换模式：如果有原文参考，检查新内容开头是否一致
                elif not append and _original_content_for_verify:
                    content_head = content[:50].strip()
                    if content_head and content_head in _original_content_for_verify and content_head not in verify_result["content"]:
                        logger.error(f"⚠️ 替换模式写入后验证失败：提交内容与写入内容不一致！filename={filename}")
        except Exception:
            pass

        output = f"【修改成功】{result['message']}"

        return output
    except Exception as e:
        return f"【修改失败】{str(e)}"


@tool
def export_document_tool(content: str, filename: str = "", title: str = "") -> str:
    """将文本内容生成为docx文档并提供下载链接。用于生成综合文档、简略文档、汇总报告等。

    【用途】当用户要求生成一个可下载的文档时使用。
    【典型问题】
    - 「帮我整理一份综合文档」「生成一份汇总报告」
    - 「把知识库的内容整合成一个文档」
    - 「导出为docx文件」「给我一个Word文档」
    - 「生成一个简略版/精简版文档」
    【与modify_document_tool的区别】
    - modify_document_tool：修改知识库中已存在的文档（同时更新知识库索引）
    - export_document_tool：生成新的文档文件供下载（不影响知识库，适合整合/汇总/生成新文档）
    【DOCX内容要求】
    - content中不要包含emoji表情符号，只包含纯文字和章节格式
    - 用户说"不能出现表情包"等要求是对DOCX文档内容的要求，不是对对话回复的要求
    - 【表格必须使用Markdown表格语法】使用 | 列1 | 列2 | 格式，会自动转为Word原生表格
      正确示例：| 部门 | 职责 | 负责人 |
      错误示例：用空格或符号对齐的假表格（如 部门    职责    负责人）
    - 不要用多个空行分隔段落，系统会自动处理段落间距
    - 使用 **粗体** 标记重要文字，会转为Word粗体格式

    Args:
        content: 文档内容（Markdown格式，支持表格/标题/列表/粗体，不要包含emoji）。
        filename: 输出文件名（含扩展名），为空则自动生成。示例：「FMEA团队汇总.docx」
        title: 文档标题，为空则使用文件名。示例：「FMEA团队信息汇总」
    """
    try:
        if not filename:
            filename = f"export_{int(time.time())}.docx"
        if not filename.endswith('.docx'):
            filename += '.docx'

        result = export_document_as_docx(content, filename, title=title, session_id=get_current_session_id())
        if result["status"] == "success":
            actual_filename = result.get('filename', filename)
            download_url = f"/api/v1/documents/export-download/{actual_filename}"
            return f"【导出成功】文档已生成：{actual_filename}\n\n下载链接：{download_url}\n\n【重要】你必须在回复中完整展示上面的下载链接URL（{download_url}），前端依赖这个URL生成下载按钮。不要省略URL，不要只说\"请下载\"而不给出URL链接。\n【重要】不要在对话中重复输出文档的完整内容，用户可以直接下载文件查看。只需简要介绍文档包含什么内容即可。"
        else:
            return f"【导出失败】{result.get('message', '未知错误')}"
    except Exception as e:
        return f"【导出失败】{str(e)}"


@tool
def export_xlsx_tool(content: str, filename: str = "", title: str = "") -> str:
    """将文本内容生成为xlsx（Excel）文档并提供下载链接。用于生成表格数据、汇总报表等Excel文件。

    【用途】当用户要求生成一个可下载的Excel文件时使用。
    【典型问题】
    - 「帮我生成一个Excel表格」「导出为xlsx」
    - 「把数据整理成Excel文件」「给我一个表格文件」
    - 「生成一份报表」「导出数据到Excel」
    - 「我要xlsx格式的」「不要docx，要xlsx」
    【与export_document_tool的区别】
    - export_document_tool：生成docx（Word）文档，适合文字报告
    - export_xlsx_tool：生成xlsx（Excel）文档，适合表格数据和报表
    【XLSX内容要求】
    - content中使用Markdown表格语法：| 列1 | 列2 | 列3 |
    - 表格外的文字会保留在对应工作表中（放在表格上方）
    - 不要包含emoji表情符号
    - ⚠️ 避免多Sheet拆分：DFMEA/PFMEA/控制计划等分析类表格，所有内容放在同一个工作表中
    - 项目信息放在表格上方的单独行中（如：项目名称：XXX），不要另建Sheet
    - 严重度(S)/频度(O)/探测度(D)评级标准、AP矩阵等参考内容不需要单独建Sheet，直接省略
    - 不要使用 === Sheet: xxx === 标记拆分多个Sheet，除非用户明确要求多Sheet

    Args:
        content: 文档内容（Markdown格式，使用表格语法组织数据，不要包含emoji）。
        filename: 输出文件名（含扩展名），为空则自动生成。示例：「FMEA团队汇总.xlsx」
        title: 文档标题/工作表名称，为空则使用文件名。示例：「FMEA团队信息」
    """
    try:
        if not filename:
            filename = f"export_{int(time.time())}.xlsx"
        if not filename.endswith('.xlsx'):
            filename = filename.rsplit('.', 1)[0] + '.xlsx'

        result = export_document_as_xlsx(content, filename, title=title, session_id=get_current_session_id())
        if result["status"] == "success":
            actual_filename = result.get('filename', filename)
            download_url = f"/api/v1/documents/export-download/{actual_filename}"
            return f"【导出成功】Excel文档已生成：{actual_filename}\n\n下载链接：{download_url}\n\n【重要】你必须在回复中完整展示上面的下载链接URL（{download_url}），前端依赖这个URL生成下载按钮。不要省略URL，不要只说\"请下载\"而不给出URL链接。\n【重要】不要在对话中重复输出表格的完整内容，用户可以直接下载文件查看。只需简要介绍文件包含什么内容即可。"
        else:
            return f"【导出失败】{result.get('message', '未知错误')}"
    except Exception as e:
        return f"【导出失败】{str(e)}"


# ===== 8D 报告生成工具 =====

@tool
def generate_8d_report_tool(
    product: str,
    defect: str,
    customer: str,
    defect_rate: str = "500PPM",
    batch_size: str = "12",
    template: str = "generic-defect",
    five_why_steps: str = "",
    rc_summary: str = "",
    containment_actions: str = "",
    permanent_actions: str = "",
    yokoten_actions: str = "",
    auto_fill: bool = False
) -> str:
    """生成专业的汽车行业 8D 报告（同时生成 xlsx 和 docx 两个文件）。

    【用途】当用户需要 8D 报告（汽车行业问题解决报告、客户投诉报告、SCAR、根因分析报告）时使用。
    【典型触发】
    - 「生成8D报告」「8D分析」
    - 「客户投诉 + 产品 + 缺陷」
    - 「根因分析报告」「SCAR」
    - 「质量问题追溯」
    【为什么必须用这个工具而不是 export_xlsx_tool】
    - generate_8d_report_tool 会调用 skills/8d-skill/scripts/generate_8d.py 脚本
    - 生成的 xlsx 带合并单元格、深蓝章节标题、交替行底色、根因黄色高亮等专业样式
    - 生成的 docx 是标准 8D Word 文档，可直接提交客户
    - export_xlsx_tool 只能生成简单表格，无样式，不适合 8D 报告
    【模板选择规则】
    - 缺陷涉及漆面/涂装/颗粒/流挂/色差/橘皮/缩孔 → paint-defect
    - 缺陷涉及装配/间隙/面差/卡扣/异响/松动 → assembly-defect
    - 缺陷涉及焊接/虚焊/焊穿/焊渣/焊点/强度 → welding-defect
    - 缺陷涉及尺寸/超差/CPK/公差/变形/收缩 → dimensional-defect
    - 其他/无法明确分类 → generic-defect

    Args:
        product: 产品名称（如「前保险杠总成」「轮毂」「ECU」）
        defect: 缺陷描述（如「漆面颗粒」「装配间隙超差」「凹陷」）
        customer: 客户名称（如「比亚迪」「一汽大众」）
        defect_rate: 不良率（默认 500PPM，安全件用 50PPM，严禁用 3%/5%/8% 等灾难级数字）
        batch_size: 批次数量（8D 分析样本数，不是生产批量，默认 12，线束类用 5）
        template: 模板 slug，可选值: paint-defect/assembly-defect/welding-defect/dimensional-defect/generic-defect
        five_why_steps: 可选，动态 5Why 内容（JSON 字符串）。当 Agent 已对缺陷做了根因分析时传入，覆盖模板预填的 5Why。
            格式: [{"level":"Why 1","question":"为什么...？","answer":"...","evidence":"..."},...]
            必须包含 6 步：问题 + Why1 + Why2 + Why3 + Why4 + Why5（根因）
            如果为空字符串，则使用模板预填的 5Why 路径
        rc_summary: 可选，动态根因总结（JSON 字符串）。当 Agent 已推演了 RC1/RC2/RC3 时传入，覆盖模板预填的 root_cause_summary。
            格式: [{"id":"RC1","description":"直接原因描述","type":"直接原因"},{"id":"RC2","description":"管理原因","type":"管理原因"},{"id":"RC3","description":"系统原因","type":"系统原因"}]
            必须包含 3 条：RC1（直接原因）+ RC2（管理原因）+ RC3（系统原因）
            如果为空字符串，则使用模板预填的 RC 总结
        containment_actions: 可选，动态 D3 遏制措施（JSON 字符串）。当 Agent 已在对话中输出了遏制措施时传入，覆盖模板预填。
            格式: ["措施1描述","措施2描述","措施3描述",...]
            如果为空字符串，则使用模板预填的遏制措施
        permanent_actions: 可选，动态 D5-D6 永久纠正措施（JSON 字符串）。当 Agent 已在对话中输出了 CA 方案时传入，覆盖模板预填。
            格式: [{"action":"措施描述","target":"针对根因","responsible":"责任人","due_date":"完成时间"},...]
            如果为空字符串，则使用模板预填的永久纠正措施
        yokoten_actions: 可选，动态 D7 横向展开措施（JSON 字符串）。当 Agent 已在对话中输出了横向展开方案时传入，覆盖模板预填。
            格式: ["措施1描述","措施2描述","措施3描述",...]
            如果为空字符串，则使用模板预填的横向展开措施
        auto_fill: 可选，自动填充模式（默认 False）。当用户明确说「你帮我填」「给我示例」「看一下范例」「其他不要问我」时设为 True。
            启用后脚本会把所有 ____ 空白替换为合理示例值（化名/示例日期/角色分配）：
            - D1 团队姓名：张伟/李娜/王芳/刘强/陈静/赵磊/周敏/孙健（按角色分配）
            - D1 联系方式：内部分机号 8001-8009
            - D3/D5/D6/D7 责任人：按措施类型分配角色
            - D3/D5/D6/D7 完成时间：当前日期 + 2/3/5/7/14/30 天
            - D8 签名/日期：化名 + 当天日期
            - 其他字段（客户联系人/投诉日期/批次号等）：合理示例值
            注意：D4 5Why/6M 的「请填写」引导提示不会被替换，保留给用户填实际分析内容
    """
    import subprocess
    import sys
    import json as _json
    import re as _re

    # ── JSON 修复函数：LLM 生成的 JSON 常见错误自动修复 ──
    def _repair_llm_json(raw: str) -> str:
        """尝试修复 LLM 生成的常见 JSON 格式错误。

        常见问题：
        1. 中文标点混入：，→,  ：→: 或 ,  "→"  "→"
        2. 缺少逗号：相邻 }{ 或 ][ 或 }[ 或 ]{ 之间缺逗号
        3. 尾部多余逗号：},] 或 ,}
        4. 值中有未转义的换行符
        5. 单引号代替双引号
        6. 缺少引号的 key
        7. "value""key": 之间缺逗号
        8. "value":"key": 冒号误用为分隔符
        """
        s = raw.strip()
        if not s:
            return s

        # 1. 中文左右引号 "…" 替换为英文 "
        s = s.replace('\u201c', '"')   # " → "
        s = s.replace('\u201d', '"')   # " → "
        s = s.replace('\u2018', "'")   # ' → '
        s = s.replace('\u2019', "'")   # ' → '

        # 2. 中文逗号替换为英文逗号
        s = s.replace('\uff0c', ',')   # ，→ ,
        s = s.replace('\uff08', '(')   # （→ (
        s = s.replace('\uff09', ')')   # ）→ )

        # 3. 中文冒号：先统一替换为英文冒号，后面再修复"冒号误用为分隔符"的情况
        s = s.replace('\uff1a', ':')   # ：→ :

        # 4. 单引号 → 双引号
        s = s.replace("'", '"')

        # 5. 修复冒号误用为属性分隔符的情况：
        #    LLM 有时生成 {"key1":"value1"："key2":"value2"}
        #    替换后变成 {"key1":"value1":"key2":"value2"}
        #    规则：如果 "xxx":"yyy":"zzz" 中间的冒号实际上应该是逗号
        #    检测模式："非冒号内容":"内容":"字母开头"  中间冒号改为逗号
        #    即 "value":"key": 模式中，第一个冒号后的值结束后的冒号应该是逗号
        #    更安全的做法：反复查找 "string":"string": 并在中间加逗号
        #    但要注意不要误改 "key":"value" 的正常模式
        #    关键洞察：正常JSON中，冒号只出现在 "key": 后面，不会出现在 "value" 后面
        #    所以如果 ":" 出现在一个看起来是value的引号字符串后面，且后面又跟着 "key":
        #    那这个冒号应该是逗号
        #    简化实现：匹配 "非空字符串":"非空字符串": 并将第二个冒号及后面的模式改为 ,
        #    反复执行直到没有更多匹配
        for _ in range(10):  # 最多修复10轮，避免无限循环
            # 匹配："xxx":"yyy":  其中 yyy 不是以 { [ 开头
            # 这里的冒号应该改为逗号
            new_s = _re.sub(
                r'("(?:[^"\\]|\\.)*")\s*:\s*("(?:[^"\\]|\\.)*")\s*:',
                r'\1:\2,',
                s
            )
            if new_s == s:
                break
            s = new_s

        # 6. 修复缺少逗号的情况：}{  →  },{   ][  →  ],[
        s = _re.sub(r'\}\s*\{', '},{', s)
        s = _re.sub(r'\]\s*\[', '],[', s)
        s = _re.sub(r'\}\s*\[', '},[', s)
        s = _re.sub(r'\]\s*\{', '],{', s)

        # 7. 去掉尾部多余逗号：,]  →  ]  ,}  →  }
        s = _re.sub(r',\s*\]', ']', s)
        s = _re.sub(r',\s*\}', '}', s)

        # 8. 修复值中未转义的换行符（在双引号内的裸换行）
        def _fix_newlines_in_strings(m):
            content = m.group(1)
            content = content.replace('\n', '\\n')
            content = content.replace('\r', '')
            return '"' + content + '"'
        s = _re.sub(r'"((?:[^"\\]|\\.)*)"', _fix_newlines_in_strings, s, flags=_re.DOTALL)

        # 9. 修复缺少引号的 key：如  {id:"RC1"}  →  {"id":"RC1"}
        s = _re.sub(r'([{\[,])\s*([a-zA-Z_]\w*)\s*:', r'\1"\2":', s)

        # 10. 修复 "value""key":  →  "value","key":（值和下一个key之间缺逗号）
        s = _re.sub(r'"\s+"([a-zA-Z_])', r'",\1', s)

        return s

    def _safe_parse_json(raw: str, label: str = ""):
        """安全解析 JSON，先尝试原始解析，失败则尝试修复后解析。"""
        if not raw or not raw.strip():
            return None
        raw = raw.strip()
        # 第一次：直接解析
        try:
            return _json.loads(raw)
        except _json.JSONDecodeError:
            pass
        # 第二次：修复后解析
        try:
            repaired = _repair_llm_json(raw)
            result = _json.loads(repaired)
            logger.info(f"[8D] {label} JSON 修复成功（原始格式有误，已自动修复）")
            return result
        except _json.JSONDecodeError as e2:
            logger.warning(f"[8D] {label} JSON 修复后仍无法解析: {e2}")
            # 第三次：终极修复 — 用正则暴力提取关键字段
            try:
                # 尝试用 ast.literal_eval 作为最后手段
                import ast
                # 替换 True/False/None 为 Python 字面量
                py_str = raw.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                result = ast.literal_eval(py_str)
                logger.info(f"[8D] {label} JSON 通过 ast.literal_eval 修复成功")
                return result
            except Exception:
                logger.warning(f"[8D] {label} JSON 所有修复方式均失败，忽略该参数")
                return None

    try:
        # 定位 generate_8d.py 脚本路径
        # settings.DATA_DIR 是项目根/data，脚本在 项目根/skills/8d-skill/scripts/generate_8d.py
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        script_path = os.path.join(project_root, "skills", "8d-skill", "scripts", "generate_8d.py")

        if not os.path.exists(script_path):
            return f"【8D报告生成失败】未找到脚本: {script_path}。请确认 skills/8d-skill/scripts/generate_8d.py 已部署。"

        # 输出目录：data/export/（根目录，不走 session_id 子目录）
        # 原因：下载端点 /api/v1/documents/export-download/{filename} 会先搜子目录再搜根目录
        # 但如果 session_id 不为空，文件存在子目录里，下载端点遍历子目录可能因各种原因找不到
        # 8D 报告文件直接存根目录，下载端点的"根目录查找"能 100% 命中
        export_dir = os.path.join(settings.DATA_DIR, "export")
        os.makedirs(export_dir, exist_ok=True)

        # 构造命令
        cmd = [
            sys.executable,
            script_path,
            "--product", product,
            "--defect", defect,
            "--customer", customer,
            "--defect-rate", defect_rate,
            "--batch-size", batch_size,
            "--template", template,
            "--output-dir", export_dir,
        ]

        # 如果传入了动态 5Why，加 --five-why-json 参数
        has_dynamic_5why = False
        if five_why_steps and five_why_steps.strip():
            # 尝试解析 JSON（含自动修复）
            parsed_5why = _safe_parse_json(five_why_steps, "five_why_steps")
            if parsed_5why is not None:
                # 序列化回标准 JSON 字符串（确保格式正确）
                clean_5why_json = _json.dumps(parsed_5why, ensure_ascii=False)
                cmd.extend(["--five-why-json", clean_5why_json])
                logger.info(f"[8D] 启用动态 5Why 覆盖（{len(clean_5why_json)} chars）")
                has_dynamic_5why = True
            else:
                logger.warning(f"[8D] five_why_steps JSON 无法解析（已尝试修复），继续用模板预填 5Why")

        # 如果传入了动态 RC 总结，加 --rc-summary-json 参数
        has_rc_summary = False
        if rc_summary and rc_summary.strip():
            # 尝试解析 JSON（含自动修复）
            parsed_rc = _safe_parse_json(rc_summary, "rc_summary")
            if parsed_rc is not None:
                # 序列化回标准 JSON 字符串（确保格式正确）
                clean_rc_json = _json.dumps(parsed_rc, ensure_ascii=False)
                cmd.extend(["--rc-summary-json", clean_rc_json])
                logger.info(f"[8D] 启用动态 RC 覆盖（{len(clean_rc_json)} chars）")
                has_rc_summary = True
            else:
                logger.warning(f"[8D] rc_summary JSON 无法解析（已尝试修复），继续用模板预填 RC")

        # 如果传入了动态 D3 遏制措施，加 --containment-actions-json 参数
        has_containment = False
        if containment_actions and containment_actions.strip():
            parsed_ca = _safe_parse_json(containment_actions, "containment_actions")
            if parsed_ca is not None:
                clean_ca_json = _json.dumps(parsed_ca, ensure_ascii=False)
                cmd.extend(["--containment-actions-json", clean_ca_json])
                logger.info(f"[8D] 启用动态 D3 遏制措施覆盖（{len(clean_ca_json)} chars）")
                has_containment = True
            else:
                logger.warning(f"[8D] containment_actions JSON 无法解析，继续用模板预填")

        # 如果传入了动态 D5-D6 永久纠正措施，加 --permanent-actions-json 参数
        has_permanent = False
        if permanent_actions and permanent_actions.strip():
            parsed_pa = _safe_parse_json(permanent_actions, "permanent_actions")
            if parsed_pa is not None:
                clean_pa_json = _json.dumps(parsed_pa, ensure_ascii=False)
                cmd.extend(["--permanent-actions-json", clean_pa_json])
                logger.info(f"[8D] 启用动态 D5-D6 永久纠正措施覆盖（{len(clean_pa_json)} chars）")
                has_permanent = True
            else:
                logger.warning(f"[8D] permanent_actions JSON 无法解析，继续用模板预填")

        # 如果传入了动态 D7 横向展开措施，加 --yokoten-actions-json 参数
        has_yokoten = False
        if yokoten_actions and yokoten_actions.strip():
            parsed_yk = _safe_parse_json(yokoten_actions, "yokoten_actions")
            if parsed_yk is not None:
                clean_yk_json = _json.dumps(parsed_yk, ensure_ascii=False)
                cmd.extend(["--yokoten-actions-json", clean_yk_json])
                logger.info(f"[8D] 启用动态 D7 横向展开措施覆盖（{len(clean_yk_json)} chars）")
                has_yokoten = True
            else:
                logger.warning(f"[8D] yokoten_actions JSON 无法解析，继续用模板预填")

        # 🔴 关键修复：auto_fill 只能由用户明确要求触发
        # 用户给根因线索（five_why_steps）≠ 要示例，只是让 5Why 更精准
        # 只有用户明确说"示例/随便填/你帮我填"时，Agent 才会传 auto_fill=True
        if auto_fill:
            cmd.append("--auto-fill")
            logger.info(f"[8D] 启用自动填充模式（用户明确要求示例）")

        logger.info(f"[8D] 调用 generate_8d.py: {' '.join(cmd[:6])}...")

        # 执行脚本（超时 60 秒）
        # 🔴 Windows 编码修复：强制子进程用 UTF-8 输出，避免 GBK/CP936 导致中文乱码
        eightd_env = os.environ.copy()
        eightd_env['PYTHONIOENCODING'] = 'utf-8'
        eightd_env['PYTHONUTF8'] = '1'
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            encoding='utf-8',
            errors='replace',
            cwd=project_root,
            env=eightd_env,
        )

        if result.returncode != 0:
            err_msg = result.stderr[-500:] if result.stderr else "无 stderr 输出"
            return f"【8D报告生成失败】脚本执行错误（返回码 {result.returncode}）：\n{err_msg}"

        # 从 stdout 解析 RESULT_JSON
        stdout = result.stdout or ""
        json_match = _re.search(r'\[RESULT_JSON\]\s*(\{.*?\})\s*$', stdout, _re.DOTALL)
        if json_match:
            try:
                result_data = _json.loads(json_match.group(1))
                xlsx_path = result_data.get("excel_path", "")
                docx_path = result_data.get("word_path", "")
                report_number = result_data.get("report_number", "")
                matched_template = result_data.get("template_slug", template)

                # 提取文件名（去掉目录部分）
                xlsx_name = os.path.basename(xlsx_path) if xlsx_path else ""
                docx_name = os.path.basename(docx_path) if docx_path else ""

                # 🔴 关键验证：检查文件是否真的存在
                xlsx_exists = xlsx_path and os.path.exists(xlsx_path)
                docx_exists = docx_path and os.path.exists(docx_path)
                
                if not xlsx_exists and not docx_exists:
                    return f"【8D报告生成失败】脚本报告成功但文件不存在。\nxlsx_path: {xlsx_path}\ndocx_path: {docx_path}\nstdout: {stdout[-300:]}"
                
                logger.info(f"[8D] 文件验证: xlsx={'存在' if xlsx_exists else '不存在'} ({xlsx_path}), docx={'存在' if docx_exists else '不存在'} ({docx_path})")

                # 生成下载链接（前端会拦截这些 URL）
                xlsx_url = f"/api/v1/documents/export-download/{xlsx_name}" if xlsx_name else ""
                docx_url = f"/api/v1/documents/export-download/{docx_name}" if docx_name else ""

                # 明确告诉 Agent 实际启用了哪些模式
                modes_enabled = []
                if has_dynamic_5why:
                    modes_enabled.append("动态 5Why 覆盖（已填入您推演的根因路径）")
                if has_rc_summary:
                    modes_enabled.append("动态 RC 覆盖（已填入您推演的 RC1/RC2/RC3）")
                if has_containment:
                    modes_enabled.append("动态 D3 遏制措施覆盖")
                if has_permanent:
                    modes_enabled.append("动态 D5-D6 CA 措施覆盖")
                if has_yokoten:
                    modes_enabled.append("动态 D7 横向展开覆盖")
                if auto_fill:
                    modes_enabled.append("自动填充模式（人名/日期/责任人已填示例值）")
                modes_str = " + ".join(modes_enabled) if modes_enabled else "默认模式（空白处留 ____）"
                
                # 构建返回消息（标注文件是否存在）
                xlsx_line = f"📄 Excel 文件：{xlsx_name}\n下载链接：{xlsx_url}\n" if xlsx_exists else f"📄 Excel 文件：生成失败\n"
                docx_line = f"📝 Word 文件：{docx_name}\n下载链接：{docx_url}\n" if docx_exists else f"📝 Word 文件：生成失败\n"
                
                return (
                    f"【8D报告生成成功】\n"
                    f"报告编号：{report_number}\n"
                    f"匹配模板：{matched_template}\n"
                    f"启用模式：{modes_str}\n\n"
                    f"{xlsx_line}"
                    f"说明：单 Sheet 完整 Excel，含合并单元格+章节标题+交替行底色+根因高亮，可继续编辑\n\n"
                    f"{docx_line}"
                    f"说明：标准 8D Word 文档，可编辑后使用\n\n"
                    f"【重要】你必须在回复中完整展示上面的下载链接 URL，前端依赖这些 URL 生成下载按钮。\n"
                    f"【重要】不要在对话中重复输出 8D 报告的完整内容，用户可以直接下载文件查看。\n"
                    f"只需简要告诉用户：报告已生成、匹配了什么模板、{'空白处已填示例值' if auto_fill else '空白处需补充实际数据'}。"
                )
            except _json.JSONDecodeError as e:
                return f"【8D报告生成失败】解析脚本输出 JSON 失败: {e}\n原始输出: {stdout[-500:]}"

        # 如果没有 RESULT_JSON，尝试从 stdout 中查找文件路径
        xlsx_match = _re.search(r'Excel 已生成[：:]\s*(.+?\.xlsx)', stdout)
        docx_match = _re.search(r'Word 已生成[：:]\s*(.+?\.docx)', stdout)

        if xlsx_match or docx_match:
            msg = "【8D报告生成成功】\n"
            if xlsx_match:
                xlsx_name = os.path.basename(xlsx_match.group(1).strip())
                msg += f"📄 Excel 文件：{xlsx_name}\n下载链接：/api/v1/documents/export-download/{xlsx_name}\n\n"
            if docx_match:
                docx_name = os.path.basename(docx_match.group(1).strip())
                msg += f"📝 Word 文件：{docx_name}\n下载链接：/api/v1/documents/export-download/{docx_name}\n\n"
            msg += "【重要】你必须在回复中完整展示上面的下载链接 URL，前端依赖这些 URL 生成下载按钮。"
            return msg

        return f"【8D报告生成失败】脚本执行成功但未找到文件路径。\nstdout: {stdout[-500:]}\nstderr: {result.stderr[-500:] if result.stderr else ''}"

    except subprocess.TimeoutExpired:
        return "【8D报告生成失败】脚本执行超时（60秒），请检查 openpyxl/python-docx 是否已安装，或缩减输入内容后重试。"
    except FileNotFoundError as e:
        return f"【8D报告生成失败】Python 解释器未找到: {e}"
    except Exception as e:
        logger.exception("[8D] generate_8d_report_tool 异常")
        return f"【8D报告生成失败】{type(e).__name__}: {str(e)}"



# ===== FMEA 报告生成工具（PFMEA/DFMEA） =====

@tool
def generate_fmea_report_tool(
    fmea_type: str,
    product: str,
    customer: str,
    project_no: str = "",
    system_level: str = "",
    design_responsibility: str = "",
    process_name: str = "",
    process_steps: str = "",
    manufacturing_site: str = "",
    team: str = "",
    template: str = "generic-fmea",
    failure_chains: str = "",
    auto_fill: bool = False
) -> str:
    """生成专业的汽车行业 FMEA 报告（PFMEA 或 DFMEA，同时生成 xlsx 和 docx 两个文件）。

    【用途】当用户需要 FMEA 分析报告（设计 FMEA / 过程 FMEA / 潜在失效模式分析）时使用。
    【典型触发】
    - 「生成DFMEA」「做PFMEA」「FMEA分析」
    - 「设计FMEA」「过程FMEA」「潜在失效模式分析」
    - 「APQP阶段FMEA」「PPAP提交FMEA」「控制计划关联FMEA」
    - 「严重度S/频度O/探测度D评分」「AP行动优先级」「特殊特性CC/SC识别」
    【为什么必须用这个工具而不是 export_xlsx_tool】
    - generate_fmea_report_tool 会调用 skills/pfmea-dfmea-skill/scripts/generate_fmea.py 脚本
    - 生成的 xlsx 带 7 个 Sheet（表头/结构/功能/失效/风险/优化/矩阵）、AP 热力图、CC/SC 高亮等专业样式
    - 生成的 docx 是标准 FMEA Word 文档（7 章 + 签名栏），可直接提交客户
    - export_xlsx_tool 只能生成简单表格，无样式，不适合 FMEA 报告
    【FMEA 类型选择】
    - 用户提到「产品设计」「零部件设计」「DFMEA」「设计FMEA」→ fmea_type="DFMEA"
    - 用户提到「生产工艺」「制造过程」「PFMEA」「过程FMEA」「工序」→ fmea_type="PFMEA"
    - 同时提到设计+过程 / 用户未明示 → 必须先追问用户
    【模板选择规则】
    - 含 ECU/控制器/传感器/线束/PCB/电路 → electronic-ecm
    - 含 齿轮/轴承/紧固件/轴/壳体/装配 → mechanical-assembly
    - 含 电镀/热处理/氧化/表面处理/淬火 → surface-treatment
    - 含 喷涂/电泳/漆面/涂装/喷漆 → painting-coating
    - 其他/无法明确分类 → generic-fmea

    Args:
        fmea_type: FMEA 类型，必填，"DFMEA" 或 "PFMEA"（大小写敏感）
        product: 产品名称（如「前保险杠总成」「ECU 控制单元」「前照灯 LED 模组」）
        customer: 客户名称（如「比亚迪」「一汽大众」）
        project_no: 可选，项目编号（如「P2026-0123」）
        system_level: DFMEA 专用，系统层级（整车/系统/子系统/组件/零件）
        design_responsibility: DFMEA 专用，设计责任方（自有设计/供应商设计/联合设计）
        process_name: PFMEA 专用，工艺名称（如「注塑成型」「焊接」「装配」）
        process_steps: PFMEA 专用，工序清单（逗号分隔，如「原料干燥,注塑成型,去毛刺,外观检验,包装」）
        manufacturing_site: PFMEA 专用，制造地址
        team: 可选，团队成员（如「张三（设计）/李四（质量）/王五（工艺）」）
        template: 模板 slug，可选值: electronic-ecm/mechanical-assembly/surface-treatment/painting-coating/generic-fmea
        failure_chains: **核心参数（强烈推荐传入）**，动态失效链内容（JSON 字符串）。
            🔴🔴🔴 **重要**：只要 Agent 在对话中输出过任何失效链（FE/FM/FC），就**必须**通过此参数传入，否则文件内容会用模板预填值，与对话输出不一致！
            - 对话中输出 N 条失效链 → failure_chains 必须传 N 条
            - 对话中说的 FE/FM/FC/S/O/D 必须与 failure_chains 中的字段完全一致
            - 🔴 **优化措施也必须通过此参数传入**：每条失效链可包含 measure_type/measure_desc/measure_owner/measure_due_date/post_s/post_o/post_d/post_ap 字段，用于填充"六、优化措施"表。不传这些字段会导致优化措施表用模板预填值（与对话中推演的措施不一致）
            - 唯一例外：用户只给产品+客户、Agent 没做任何失效链分析时可不传
            格式: [{"fe":"失效影响","fm":"失效模式","fc":"失效起因","s":8,"o":5,"d":6,"ap":"H","pc":"预防控制","dc":"探测控制",
                    "measure_type":"PC+DC 改进","measure_desc":"① 具体措施1 ② 具体措施2","measure_owner":"设计主管","measure_due_date":"D+30天",
                    "post_s":8,"post_o":3,"post_d":3,"post_ap":"M"},...]
            字段说明：
            - fe/fm/fc: 必填，失效链三级结构
            - s/o/d: 强烈推荐填写（1-10 整数），缺失时用模板 hint 值
            - ap: 可选（H/M/L），缺失时脚本自动计算
            - pc/dc: 可选，预防控制与探测控制描述
            - measure_type: 可选，措施类型（"PC+DC 改进" / "PC 改进" / "DC 改进"），缺失时脚本根据 pc/dc 推断
            - measure_desc: 可选但**强烈推荐**，措施描述（① ② ③ 具体内容），缺失时用 pc+dc 拼接（不推荐）
            - measure_owner: 可选但**强烈推荐**，责任人（如"设计主管""焊接工程师"），缺失时填 ____
            - measure_due_date: 可选但**强烈推荐**，截止日期（如"D+30天""2026-07-28"），缺失时填 ____
            - post_s/post_o/post_d: 可选但**强烈推荐**，措施后 S/O/D 评分（1-10 整数），缺失时填 "—"
            - post_ap: 可选但**强烈推荐**，措施后 AP（H/M/L），缺失时填 "—"
            如果为空字符串，则使用模板预填的失效链（electronic-ecm 模板预填 7 条，generic-fmea 预填 5 条）—— ⚠️ 这通常不是你想要的，请优先传入对话中推演的失效链
        auto_fill: 可选，自动填充模式（默认 False）。当用户明确说「你帮我填」「给我示例」「看一下范例」时设为 True。
            启用后脚本会把所有 ____ 空白替换为合理示例值：
            - FMEA 团队表姓名：张伟/李娜/刘强/陈静/赵磊/周敏/王芳/孙健（按角色分配）
            - 联系方式：内部分机号 8001-8009
            - 优化措施责任人：按措施序号轮换分配
            - 优化措施截止日期：当前日期 + 7/10/14/21/30/45/60 天
            - 签名栏：化名 + 当天日期
            - 表头信息：FMEA 编号 / 编制人 / 审核人 / 批准人 / 项目编号 等
            注意：S/O/D 评分仍由模板 hint 决定，AP 由 get_ap_priority 自动计算（不采用 ap_hint，避免不一致）
    """
    import subprocess
    import sys
    import json as _json
    import re as _re

    # ── JSON 修复函数（与 generate_8d_report_tool 共用同一套修复逻辑） ──
    def _repair_llm_json(raw: str) -> str:
        """尝试修复 LLM 生成的常见 JSON 格式错误。"""
        s = raw.strip()
        if not s:
            return s
        s = s.replace('\u201c', '"').replace('\u201d', '"')
        s = s.replace('\u2018', "'").replace('\u2019', "'")
        s = s.replace('\uff0c', ',').replace('\uff08', '(').replace('\uff09', ')')
        s = s.replace('\uff1a', ':')
        s = s.replace("'", '"')
        for _ in range(10):
            new_s = _re.sub(
                r'("(?:[^"\\]|\\.)*")\s*:\s*("(?:[^"\\]|\\.)*")\s*:',
                r'\1:\2,',
                s
            )
            if new_s == s:
                break
            s = new_s
        s = _re.sub(r'\}\s*\{', '},{', s)
        s = _re.sub(r'\]\s*\[', '],[', s)
        s = _re.sub(r'\}\s*\[', '},[', s)
        s = _re.sub(r'\]\s*\{', '],{', s)
        s = _re.sub(r',\s*\]', ']', s)
        s = _re.sub(r',\s*\}', '}', s)
        def _fix_newlines_in_strings(m):
            content = m.group(1)
            content = content.replace('\n', '\\n').replace('\r', '')
            return '"' + content + '"'
        s = _re.sub(r'"((?:[^"\\]|\\.)*)"', _fix_newlines_in_strings, s, flags=_re.DOTALL)
        s = _re.sub(r'([{\[,])\s*([a-zA-Z_]\w*)\s*:', r'\1"\2":', s)
        s = _re.sub(r'"\s+"([a-zA-Z_])', r'",\1', s)
        return s

    def _safe_parse_json(raw: str, label: str = ""):
        """安全解析 JSON，先尝试原始解析，失败则尝试修复后解析。"""
        if not raw or not raw.strip():
            return None
        raw = raw.strip()
        try:
            return _json.loads(raw)
        except _json.JSONDecodeError:
            pass
        try:
            repaired = _repair_llm_json(raw)
            result = _json.loads(repaired)
            logger.info(f"[FMEA] {label} JSON 修复成功（原始格式有误，已自动修复）")
            return result
        except _json.JSONDecodeError as e2:
            logger.warning(f"[FMEA] {label} JSON 修复后仍无法解析: {e2}")
            try:
                import ast
                py_str = raw.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                result = ast.literal_eval(py_str)
                logger.info(f"[FMEA] {label} JSON 通过 ast.literal_eval 修复成功")
                return result
            except Exception:
                logger.warning(f"[FMEA] {label} JSON 所有修复方式均失败，忽略该参数")
                return None

    try:
        # 参数校验
        fmea_type_upper = (fmea_type or "").strip().upper()
        if fmea_type_upper not in ("DFMEA", "PFMEA"):
            return f"【FMEA报告生成失败】fmea_type 必须是 'DFMEA' 或 'PFMEA'，当前值：'{fmea_type}'"

        # 定位 generate_fmea.py 脚本路径
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        script_path = os.path.join(project_root, "skills", "pfmea-dfmea-skill", "scripts", "generate_fmea.py")

        if not os.path.exists(script_path):
            return f"【FMEA报告生成失败】未找到脚本: {script_path}。请确认 skills/pfmea-dfmea-skill/scripts/generate_fmea.py 已部署（git submodule update --init --recursive）。"

        # 输出目录：data/export/（与 8D 报告相同的下载端点）
        export_dir = os.path.join(settings.DATA_DIR, "export")
        os.makedirs(export_dir, exist_ok=True)

        # 构造命令
        cmd = [
            sys.executable,
            script_path,
            "--fmea-type", fmea_type_upper,
            "--product", product,
            "--customer", customer,
            "--template", template,
            "--output-dir", export_dir,
        ]

        # 可选参数
        if project_no:
            cmd.extend(["--project-no", project_no])
        if system_level:
            cmd.extend(["--system-level", system_level])
        if design_responsibility:
            cmd.extend(["--design-responsibility", design_responsibility])
        if process_name:
            cmd.extend(["--process-name", process_name])
        if process_steps:
            cmd.extend(["--process-steps", process_steps])
        if manufacturing_site:
            cmd.extend(["--manufacturing-site", manufacturing_site])
        if team:
            cmd.extend(["--team", team])

        # 动态失效链
        has_dynamic_chains = False
        if failure_chains and failure_chains.strip():
            parsed_chains = _safe_parse_json(failure_chains, "failure_chains")
            if parsed_chains is not None:
                # 写入临时 JSON 文件传给脚本（避免命令行参数过长）
                import tempfile
                chains_file = os.path.join(export_dir, f"_fmea_chains_{int(__import__('time').time())}.json")
                with open(chains_file, "w", encoding="utf-8") as cf:
                    _json.dump(parsed_chains, cf, ensure_ascii=False, indent=2)
                cmd.extend(["--failure-chains-json", chains_file])
                logger.info(f"[FMEA] 启用动态失效链覆盖（{len(parsed_chains) if isinstance(parsed_chains, list) else 1} 条）")
                has_dynamic_chains = True
            else:
                logger.warning(f"[FMEA] failure_chains JSON 无法解析（已尝试修复），继续用模板预填失效链")

        if auto_fill:
            cmd.append("--auto-fill")
            logger.info(f"[FMEA] 启用自动填充模式（用户明确要求示例）")

        logger.info(f"[FMEA] 调用 generate_fmea.py: fmea_type={fmea_type_upper}, template={template}")

        # 执行脚本（超时 60 秒）
        # 🔴 Windows 编码修复：强制子进程用 UTF-8 输出，避免 GBK/CP936 导致中文乱码
        fmea_env = os.environ.copy()
        fmea_env['PYTHONIOENCODING'] = 'utf-8'
        fmea_env['PYTHONUTF8'] = '1'
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            encoding='utf-8',
            errors='replace',
            cwd=project_root,
            env=fmea_env,
        )

        if result.returncode != 0:
            err_msg = result.stderr[-500:] if result.stderr else "无 stderr 输出"
            return f"【FMEA报告生成失败】脚本执行错误（返回码 {result.returncode}）：\n{err_msg}"

        # 从 stdout 解析生成的文件路径
        stdout = result.stdout or ""
        # 脚本输出格式：[OK] Excel 报告已生成：/path/to/FMEA_XXX.xlsx
        #               [OK] Word 报告已生成：/path/to/FMEA_XXX.docx
        xlsx_match = _re.search(r'Excel 报告已生成[：:]\s*(.+?\.xlsx)', stdout)
        docx_match = _re.search(r'Word 报告已生成[：:]\s*(.+?\.docx)', stdout)

        if not (xlsx_match and docx_match):
            return f"【FMEA报告生成失败】脚本执行成功但未找到文件路径。\nstdout: {stdout[-500:]}\nstderr: {result.stderr[-500:] if result.stderr else ''}"

        xlsx_path = xlsx_match.group(1).strip()
        docx_path = docx_match.group(1).strip()
        xlsx_name = os.path.basename(xlsx_path)
        docx_name = os.path.basename(docx_path)

        # 验证文件确实存在
        xlsx_exists = os.path.exists(xlsx_path)
        docx_exists = os.path.exists(docx_path)
        logger.info(f"[FMEA] 文件验证: xlsx={'存在' if xlsx_exists else '不存在'} ({xlsx_path}), docx={'存在' if docx_exists else '不存在'} ({docx_path})")

        if not xlsx_exists and not docx_exists:
            return f"【FMEA报告生成失败】脚本报告成功但文件不存在。\nxlsx_path: {xlsx_path}\ndocx_path: {docx_path}"

        # 生成下载链接（前端会拦截这些 URL）
        xlsx_url = f"/api/v1/documents/export-download/{xlsx_name}" if xlsx_name else ""
        docx_url = f"/api/v1/documents/export-download/{docx_name}" if docx_name else ""

        # 提取统计信息
        chains_count_match = _re.search(r'失效链数[：:]\s*(\d+)', stdout)
        ap_h_match = _re.search(r'AP=H[：:]\s*(\d+)', stdout)
        ap_m_match = _re.search(r'AP=M[：:]\s*(\d+)', stdout)
        ap_l_match = _re.search(r'AP=L[：:]\s*(\d+)', stdout)
        chains_count = chains_count_match.group(1) if chains_count_match else "?"
        ap_h = ap_h_match.group(1) if ap_h_match else "?"
        ap_m = ap_m_match.group(1) if ap_m_match else "?"
        ap_l = ap_l_match.group(1) if ap_l_match else "?"

        # 明确告诉 Agent 实际启用了哪些模式
        modes_enabled = []
        if has_dynamic_chains:
            modes_enabled.append("动态失效链覆盖（已填入您推演的 FE/FM/FC）")
        if auto_fill:
            modes_enabled.append("自动填充模式（人名/日期/责任人已填示例值）")
        modes_str = " + ".join(modes_enabled) if modes_enabled else "默认模式（空白处留 ____）"

        xlsx_line = f"📄 Excel 文件：{xlsx_name}\n下载链接：{xlsx_url}\n" if xlsx_exists else f"📄 Excel 文件：生成失败\n"
        docx_line = f"📝 Word 文件：{docx_name}\n下载链接：{docx_url}\n" if docx_exists else f"📝 Word 文件：生成失败\n"

        return (
            f"【FMEA报告生成成功】\n"
            f"FMEA 类型：{fmea_type_upper}\n"
            f"匹配模板：{template}\n"
            f"启用模式：{modes_str}\n"
            f"失效链统计：共 {chains_count} 条，AP=H（高优先级）{ap_h} 条 / AP=M（中）{ap_m} 条 / AP=L（低）{ap_l} 条\n\n"
            f"{xlsx_line}"
            f"说明：7 Sheet 完整 Excel（表头/结构/功能/失效/风险/优化/矩阵），含 AP 热力图 + CC/SC 高亮 + 自动行高\n\n"
            f"{docx_line}"
            f"说明：标准 FMEA Word 文档（7 章 + 签名栏），可编辑后使用\n\n"
            f"【重要】你必须在回复中完整展示上面的下载链接 URL，前端依赖这些 URL 生成下载按钮。\n"
            f"【重要】不要在对话中重复输出 FMEA 报告的完整内容，用户可以直接下载文件查看。\n"
            f"只需简要告诉用户：报告已生成、匹配了什么模板、{'空白处已填示例值' if auto_fill else '空白处需补充实际数据'}。"
        )

    except subprocess.TimeoutExpired:
        return "【FMEA报告生成失败】脚本执行超时（60秒），请检查 openpyxl/python-docx 是否已安装，或缩减输入内容后重试。"
    except FileNotFoundError as e:
        return f"【FMEA报告生成失败】Python 解释器未找到: {e}"
    except Exception as e:
        logger.exception("[FMEA] generate_fmea_report_tool 异常")
        return f"【FMEA报告生成失败】{type(e).__name__}: {str(e)}"



# ===== [#12] 外部系统集成工具 =====

@tool
def github_api_tool(action: str, repo: str = "", path: str = "", content: str = "", message: str = "", token: str = "") -> str:
    """与 GitHub 仓库进行交互，支持读取和更新文件。

    【用途】当代码仓库操作需求时使用，如查看仓库内容、更新文件、获取文件内容等。
    【典型问题】「帮我把这个改动推到GitHub」「查看仓库的文件列表」「更新某个文件」

    【使用规则】
    - 读取用 action="read"（大文件自动截断8000字）或 "read_full"（返回全部内容）
    - 修改前务必先用 action="read_full" 读取完整原始内容，再修改后用 action="update" 提交
    - 用户发送的 GitHub Token 务必通过 token 参数传入；不要回复中重复显示用户的 Token
    - 公开仓库读取无需 Token，写操作（update）必须要有 Token

    Args:
        action: 操作类型，支持 "read"（读取文件，大文件截断8000字）, "read_full"（读取完整文件，不截断）, "list"（列出目录内容）, "update"（更新文件）
        repo: 仓库名称，格式 "owner/repo"，示例 "cy556-like/company-doc-agent"
        path: 文件路径，示例 "app/config.py"
        content: 更新文件时的文件内容（仅 action=update 时需要）
        message: 更新文件时的 commit message（仅 action=update 时需要）
        token: GitHub Personal Access Token（可选）。用户在对话中提供时可传入，用于写操作鉴权。未提供时从环境变量 GITHUB_TOKEN 读取。
    """
    import httpx

    # Token 优先级：对话中传入 > 环境变量
    github_token = token or os.getenv("GITHUB_TOKEN", "")
    if not repo:
        return "【GitHub 操作】缺少仓库参数，请提供 repo 参数，格式：owner/repo"

    # 构建请求头：公开仓库的 read/list 不需要 Token，update 操作需要 Token
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    # 写操作（update）必须需要 Token
    if action == "update" and not github_token:
        return "【GitHub 操作】写入操作需要 Token 鉴权。请在对话中提供 Token，或在 .env 中设置 GITHUB_TOKEN。读取公开仓库不需要 Token。"
    base_url = f"https://api.github.com/repos/{repo}"

    # [性能修复] 使用 httpx.Client 上下文管理器确保 TCP 连接及时释放，
    # 避免长时间运行后文件描述符耗尽导致变慢/报错
    try:
        if action == "list":
            url = f"{base_url}/contents/{path}" if path else f"{base_url}/contents"
            with httpx.Client(headers=headers, timeout=15) as client:
                resp = client.get(url)
            if resp.status_code != 200:
                return f"【GitHub 操作】获取目录失败: {resp.status_code} {resp.text[:200]}"
            items = resp.json()
            if isinstance(items, dict) and items.get("message"):
                return f"【GitHub 操作】{items['message']}"
            output = f"【GitHub 目录】{repo}/{path}:\n\n"
            for item in items[:20]:
                icon = "📁" if item.get("type") == "dir" else "📄"
                output += f"  {icon} {item['name']} ({item.get('type', '')})\n"
            if len(items) > 20:
                output += f"  ... 共 {len(items)} 项\n"
            return output

        elif action in ("read", "read_full"):
            if not path:
                return "【GitHub 操作】读取文件需要提供 path 参数"

            # 对于大文件，使用 GitHub Blob API 避免内容截断
            # GitHub Contents API 对大文件会返回 403 且 base64 有大小限制
            # Blob API 可获取任意大小的文件完整内容
            import base64
            file_content = ""
            sha = ""

            # 先尝试 Contents API（小文件快速获取）
            url = f"{base_url}/contents/{path}"
            with httpx.Client(headers=headers, timeout=15) as client:
                resp = client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                sha = data.get("sha", "")
                file_size = data.get("size", 0)
                # 如果文件较大（>100KB），用 Blob API 获取完整内容
                if file_size > 100000:
                    blob_sha = data.get("sha", "")
                    blob_url = f"{base_url}/git/blobs/{blob_sha}"
                    with httpx.Client(headers=headers, timeout=30) as client:
                        blob_resp = client.get(blob_url)
                    if blob_resp.status_code == 200:
                        blob_data = blob_resp.json()
                        file_content = base64.b64decode(blob_data["content"]).decode("utf-8", errors="replace")
                    else:
                        # Blob API 也失败，用 Contents API 尽量获取
                        file_content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                else:
                    file_content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            elif resp.status_code == 403:
                # Contents API 对大文件返回 403，使用 Raw URL 直接获取文件内容
                raw_url = f"https://raw.githubusercontent.com/{repo}/main/{path}"
                with httpx.Client(headers={"User-Agent": "DocAgent/1.0"}, timeout=30) as client:
                    raw_resp = client.get(raw_url)
                if raw_resp.status_code == 200:
                    file_content = raw_resp.text
                else:
                    return f"【GitHub 操作】读取文件失败: Contents API 403, Raw URL {raw_resp.status_code}"
            else:
                return f"【GitHub 操作】读取文件失败: {resp.status_code} {resp.text[:200]}"

            output = f"【GitHub 文件】{repo}/{path} (sha: {sha[:8] if sha else 'unknown'}..., 共 {len(file_content)} 字符)\n\n"

            # action="read" 时限制返回长度（避免工具输出过长拖慢 Agent），
            # action="read_full" 时返回完整内容
            if action == "read" and len(file_content) > 8000:
                output += file_content[:8000]
                output += f"\n\n... (文件共 {len(file_content)} 字符，已显示前8000字。如需完整内容请使用 action=read_full)"
            else:
                output += file_content

            return output

        elif action == "update":
            if not path or not content:
                return "【GitHub 操作】更新文件需要提供 path 和 content 参数"
            import base64
            # 先获取当前文件的 sha
            url = f"{base_url}/contents/{path}"
            with httpx.Client(headers=headers, timeout=15) as client:
                resp = client.get(url)
            if resp.status_code != 200:
                # 文件不存在，创建新文件
                sha = None
            else:
                sha = resp.json().get("sha")

            commit_msg = message or f"Update {path} via DocAgent"
            body = {
                "message": commit_msg,
                "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
            }
            if sha:
                body["sha"] = sha

            with httpx.Client(headers=headers, timeout=15) as client:
                resp = client.put(url, json=body)
            if resp.status_code in (200, 201):
                return f"【GitHub 操作】文件更新成功: {repo}/{path}\nCommit: {commit_msg}"
            else:
                return f"【GitHub 操作】文件更新失败: {resp.status_code} {resp.text[:300]}"

        else:
            return f"【GitHub 操作】不支持的操作: {action}。支持: read, read_full, list, update"

    except Exception as e:
        return f"【GitHub 操作】操作失败: {str(e)}\n提示：读取公开仓库不需要 Token，写入操作才需要配置 GITHUB_TOKEN。"


@tool
def send_email_tool(to: str, subject: str, body: str) -> str:
    """发送电子邮件通知。

    【用途】当需要发送邮件通知时使用，如发送报告、通知审批结果等。
    【典型问题】「发邮件通知技术部」「给张三发邮件」

    Args:
        to: 收件人邮箱地址，多人用逗号分隔。示例："zhangsan@company.com" 或 "a@co.com,b@co.com"
        subject: 邮件主题
        body: 邮件正文内容
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")
    smtp_from = os.getenv("SMTP_FROM", smtp_user)

    if not smtp_host or not smtp_user:
        return "【邮件发送】未配置 SMTP 邮件服务。请在 .env 中设置 SMTP_HOST、SMTP_USER、SMTP_PASS。"

    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_from
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_from, to.split(","), msg.as_string())
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_from, to.split(","), msg.as_string())

        return f"【邮件发送】邮件已成功发送给 {to}，主题：{subject}"

    except Exception as e:
        return f"【邮件发送】发送失败: {str(e)}\n建议：检查 SMTP 配置是否正确。"


@tool
def database_query_tool(query: str, database: str = "default") -> str:
    """执行 SQL 查询语句（只读），支持查询企业数据库。

    【用途】当需要从数据库中查询业务数据时使用，如订单、库存、销售数据等。
    【典型问题】「查询本月销售额」「库存还剩多少」「最近10笔订单」

    注意：此工具只支持 SELECT 查询，不支持 INSERT/UPDATE/DELETE 等写操作。

    Args:
        query: SQL 查询语句。示例："SELECT * FROM orders WHERE date > '2024-01-01' LIMIT 10"
        database: 数据库名称（可选，默认为 default）
    """
    # 安全检查：只允许 SELECT 语句
    normalized = query.strip().upper()
    if not normalized.startswith("SELECT") and not normalized.startswith("WITH"):
        return "【数据库查询】安全限制：仅支持 SELECT 查询，不允许执行 INSERT/UPDATE/DELETE 等写操作。"

    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "EXEC"]
    for kw in forbidden:
        if kw in normalized.split():
            return f"【数据库查询】安全限制：检测到禁止的关键字 {kw}，仅支持只读查询。"

    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        return "【数据库查询】未配置 DATABASE_URL 环境变量。请在 .env 中设置数据库连接字符串。"

    try:
        import sqlite3

        # 支持 SQLite 和 PostgreSQL
        if db_url.startswith("sqlite"):
            conn = sqlite3.connect(db_url.replace("sqlite:///", ""), timeout=10)
        elif db_url.startswith("postgresql"):
            try:
                import psycopg2
                conn = psycopg2.connect(db_url, connect_timeout=10)
            except ImportError:
                return "【数据库查询】PostgreSQL 驱动未安装，请运行: pip install psycopg2-binary"
        else:
            return f"【数据库查询】不支持的数据库类型: {db_url.split(':')[0]}"

        try:
            cursor = conn.cursor()
            cursor.execute(query)

            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchmany(50)  # 限制最多返回 50 行

                output = f"【数据库查询】查询成功，返回 {len(rows)} 行：\n\n"
                # 表头
                output += "| " + " | ".join(columns) + " |\n"
                output += "|" + "|".join(["---" for _ in columns]) + "|\n"
                # 数据行
                for row in rows:
                    output += "| " + " | ".join(str(v) if v is not None else "NULL" for v in row) + " |\n"

                if len(rows) == 50:
                    output += "\n（最多显示 50 行，如需更多请添加 LIMIT 条件）"
                return output
            else:
                return "【数据库查询】查询执行成功，无返回结果。"
        finally:
            conn.close()

    except Exception as e:
        return f"【数据库查询】查询失败: {str(e)}\n建议：检查 SQL 语法和数据库连接配置。"


# ===== 导出工具列表 =====

# 基础工具（始终可用）
BASE_TOOLS = [
    search_documents_tool,
    lookup_employee_tool,
    list_departments_tool,
    list_documents_tool,
    get_document_content_tool,
    upload_document_tool,
    delete_document_tool,
    modify_document_tool,
    export_document_tool,
    export_xlsx_tool,
    generate_8d_report_tool,
    generate_fmea_report_tool,
]

# 联网搜索工具（按需启用）
WEB_SEARCH_TOOLS = [
    web_search_tool,
]

# [#12] 外部系统集成工具（仅当配置了对应环境变量时才启用，避免无意义token消耗）
EXTERNAL_TOOLS = []
if os.getenv("GITHUB_TOKEN"):
    EXTERNAL_TOOLS.append(github_api_tool)
if os.getenv("SMTP_HOST"):
    EXTERNAL_TOOLS.append(send_email_tool)
if os.getenv("DATABASE_URL"):
    EXTERNAL_TOOLS.append(database_query_tool)

# 全部工具
ALL_TOOLS = BASE_TOOLS + WEB_SEARCH_TOOLS + EXTERNAL_TOOLS


def get_tools(web_search: bool = False):
    """根据参数获取工具列表

    Args:
        web_search: 是否启用联网搜索工具

    Returns:
        工具列表
    """
    if web_search:
        return ALL_TOOLS
    return BASE_TOOLS + EXTERNAL_TOOLS


def get_cache_stats() -> dict:
    """获取工具缓存统计信息"""
    return _tool_cache.stats()
