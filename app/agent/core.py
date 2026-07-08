"""
Agent 核心逻辑模块
使用 LangGraph 构建 ReAct 模式的 Agent
ReAct = Reasoning(推理) + Acting(行动) → 边思考边行动
支持流式输出（Streaming SSE）
支持多步骤任务编排、工具并行执行、自省纠错

性能优化:
- 流式首Token优化：使用 astream_events v2 减少首Token延迟
- 意图路由：简单问题跳过Agent工具调用，直接LLM回答
- 提示词精简：减少系统提示词Token数，加速推理
- 历史消息窗口：限制上下文长度，避免过长上下文拖慢推理
- Agent单例复用：避免每次请求重建Agent图
"""
import asyncio
import copy
import time
import logging
import hashlib
import contextvars
import threading
from datetime import datetime
from typing import Annotated, AsyncGenerator
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.config import settings, VISION_MODELS, DEFAULT_VISION_MODEL, VISION_API_KEY, VISION_BASE_URL, FAST_MODELS, DEEPSEEK_MODELS, VOLCENGINE_MODELS, QWEN_MODELS, MIMO_MODELS, GLM_MODELS
from app.agent.tools import ALL_TOOLS, get_tools, set_current_agent_id, set_current_session_id, get_current_session_id, reset_search_count
from app.agent.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_WEB_SEARCH, CHAT_SYSTEM_PROMPT, get_agent_keywords_section
from app.memory.manager import get_session_history

logger = logging.getLogger(__name__)

# [BUG FIX v6] 会话取消信号：全局 dict 按 session_id 追踪取消状态
# v5 用 contextvars.ContextVar 有两个致命缺陷：
# 1. 跨 HTTP 请求隔离 → 新请求看不到旧请求的 cancel_event → 无法取消幽灵任务
# 2. CancelledError handler 从未调用 cancel_event.set() → _is_session_cancelled() 永远返回 False
# v6 改用全局 dict + threading.Lock + session_id 索引，并在取消时真正 set 事件
# [BUG FIX v7] 会话取消信号：全局 dict 按 session_id 追踪取消状态
# v6 用 contextvars.ContextVar 有两个致命缺陷：
# 1. 跨 HTTP 请求隔离 → 新请求看不到旧请求的 cancel_event → 无法取消幽灵任务
# 2. CancelledError handler 从未调用 cancel_event.set() → _is_session_cancelled() 永远返回 False
# v6 改用全局 dict + threading.Lock + session_id 索引，并在取消时真正 set 事件
# v7 增加 created_at 时间戳，定期清理超时条目，防止长时间运行后字典无限增长
_session_cancel_events: dict[str, tuple] = {}  # session_id -> (threading.Event, created_at)
_session_cancel_lock = threading.Lock()
_SESSION_CANCEL_TTL = 1800  # 30分钟，超过此时间的条目在定期清理时删除

def _get_or_create_cancel_event(session_id: str) -> threading.Event:
    """为 session 获取或创建取消事件，同时取消同一 session 的上一个事件"""
    with _session_cancel_lock:
        old_entry = _session_cancel_events.pop(session_id, None)
        if old_entry is not None:
            old_entry[0].set()  # 取消同一 session 的上一个幽灵任务
            logger.info(f"[取消追踪] 已取消 session={session_id} 的上一个 Agent 任务")
        evt = threading.Event()
        _session_cancel_events[session_id] = (evt, time.time())
        return evt

def _set_session_cancelled(session_id: str):
    """标记 session 已取消，阻止 think() 发起新的 LLM 调用"""
    with _session_cancel_lock:
        entry = _session_cancel_events.get(session_id)
        if entry is not None:
            entry[0].set()

def _cleanup_session_cancel(session_id: str):
    """正常结束时清理取消事件"""
    with _session_cancel_lock:
        _session_cancel_events.pop(session_id, None)

def _is_session_cancelled(session_id: str = None) -> bool:
    """检查当前 session 是否已被取消（在 think() 中调用以避免无效 LLM 调用）
    
    通过 get_current_session_id() 获取 session_id，在 ThreadPoolExecutor 子线程中也能正确获取
    （contextvars 自动传播到子线程）
    """
    sid = session_id or get_current_session_id()
    if not sid:
        return False
    with _session_cancel_lock:
        entry = _session_cancel_events.get(sid)
    if entry is not None and entry[0].is_set():
        return True
    return False

def _cleanup_stale_cancel_events():
    """[v7] 清理超时的取消事件条目，防止长时间运行后字典无限增长
    
    清理策略：
    1. 超过 TTL（30分钟）的条目直接删除
    2. 已取消/已完成（is_set）的条目也删除（任务已结束，不再需要追踪）
    """
    now = time.time()
    with _session_cancel_lock:
        stale = []
        for sid, entry in _session_cancel_events.items():
            evt, created_at = entry
            # 已取消的事件 或 超时的条目，都可以清理
            if evt.is_set() or (now - created_at > _SESSION_CANCEL_TTL):
                stale.append(sid)
        for sid in stale:
            del _session_cancel_events[sid]
    if stale:
        logger.info(f"[缓存清理] 清理了 {len(stale)} 个过期取消事件条目，剩余 {len(_session_cancel_events)}")

# 最大历史消息数量（保留足够上下文保证多轮对话质量）
MAX_HISTORY_MESSAGES = 30

# [#6] 多步骤任务编排：最大工具调用轮数
# 从8降到5：大多数场景2-3次搜索+1次导出即完成，8轮导致LLM过度搜索
# 5轮仍足够处理复杂任务（3次搜索 + 2次其他操作）
MAX_TOOL_ROUNDS = 5

# [#11] 工具重试配置
MAX_TOOL_RETRIES = 2
RETRYABLE_TOOL_ERRORS = ["搜索失败", "未找到", "连接", "超时", "timeout", "error"]

# 意图路由：仅纯闲聊/打招呼才走 Chat 模式，其余一律走 Agent 保证质量
# [质量修复] 大幅收紧简单问题判定，避免专业问题被误路由导致降级
SIMPLE_QUERY_PATTERNS = [
    # 纯闲聊/打招呼（仅这些确定不需要工具调用）
    "你好", "嗨", "hello", "hi", "你是谁", "你叫什么", "介绍一下你自己",
    "谢谢", "感谢", "再见", "拜拜", "好的", "知道了",
]
# 简单问题的最大字符数（超过此长度认为不是简单问题）
SIMPLE_MAX_LENGTH = 8  # 仅极短的打招呼/闲聊才判定为简单问题

def _is_simple_query(query: str) -> bool:
    """判断用户输入是否为简单问题（不需要工具调用的纯闲聊/打招呼）
    
    [质量修复] 收紧判定逻辑：
    - 仅纯闲聊/打招呼走 Chat 模式，避免专业问题被误杀
    - 移除"是什么""什么是""为什么"等泛化关键词（这些可能是专业问题的开头）
    - 移除短文本回退（≤15字就判定简单），因为很多专业问题也很短
    """
    query_stripped = query.strip()
    query_lower = query_stripped.lower()
    
    # 1. 仅精确匹配纯闲聊关键词
    for pattern in SIMPLE_QUERY_PATTERNS:
        if pattern in query_lower:
            return True
    
    # 2. 极短且不含问号/专业词的纯打招呼（≤8字且无问号）
    if len(query_stripped) <= SIMPLE_MAX_LENGTH and '？' not in query_stripped and '?' not in query_stripped:
        # 再排除可能包含专业意图的短句
        professional_hints = ["怎么做", "怎么写", "帮我", "分析", "生成", "检查", "评估", "写", "画"]
        if not any(h in query_stripped for h in professional_hints):
            return True
    
    return False

def _inject_current_date(system_prompt: str) -> str:
    """将当前日期注入 system prompt 尾部
    
    [质量修复] 日期信息不再以假 HumanMessage 形式插入消息列表，
    而是直接追加到 system prompt 末尾。假 HumanMessage 会干扰模型对对话流的
    理解，模型可能将其视为用户输入的一部分，影响回答质量。
    
    虽然每天日期变化会导致 prompt caching 效率略降，但对话质量更重要。
    """
    now = datetime.now()
    date_text = f"\n\n[当前日期：{now.strftime('%Y年%m月%d日')}，星期{['一','二','三','四','五','六','日'][now.weekday()]}。请在回答中涉及时间信息时使用正确的当前日期，严禁编造日期。]"
    return system_prompt + date_text

def _load_8d_skill_context(skill: str, user_input: str) -> str:
    """加载 8D skill 的完整工作流上下文（SKILL.md + 匹配到的 template.json）。

    当用户在前端点击 8D Skill 按钮后，selectedSkill='8d-skill' 会被透传到此处。
    本函数：
      1. 读取 skills/8d-skill/SKILL.md 全文（约 264 行）
      2. 从 user_input 提取关键字，按模板匹配规则选择模板 slug
         (paint-defect / assembly-defect / welding-defect / dimensional-defect / generic-defect)
      3. 读取 skills/8d-skill/templates/<slug>/template.json
      4. 拼成上下文文本返回，用于追加到 system_prompt 末尾

    任何异常都返回空字符串，确保不阻断主对话流程。
    """
    if not skill or skill != "8d-skill":
        return ""

    try:
        import os
        import json
        import logging
        logger_ = logging.getLogger(__name__)

        # 定位 skills 目录（相对于本文件 app/agent/core.py）
        # app/agent/core.py -> 上两级 = app/ -> 再上一级 = 项目根
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        skill_md_path = os.path.join(project_root, "skills", "8d-skill", "SKILL.md")
        templates_dir = os.path.join(project_root, "skills", "8d-skill", "templates")

        if not os.path.isfile(skill_md_path):
            logger_.warning(f"8D skill SKILL.md not found at {skill_md_path}")
            return ""

        with open(skill_md_path, "r", encoding="utf-8") as f:
            skill_md_content = f.read()

        # ---------- 模板匹配 ----------
        # 规则：先按缺陷描述匹配，再按产品类别复核
        # 关键字 → 模板 slug 映射
        TEMPLATE_RULES = [
            (["漆面", "涂装", "颗粒", "流挂", "色差", "橘皮", "缩孔"], "paint-defect"),
            (["装配", "间隙", "面差", "卡扣", "异响", "松动"], "assembly-defect"),
            (["焊接", "虚焊", "焊穿", "焊渣", "焊点", "强度"], "welding-defect"),
            (["尺寸", "超差", "CPK", "cpk", "公差", "变形", "收缩"], "dimensional-defect"),
        ]

        matched_slug = None
        for keywords, slug in TEMPLATE_RULES:
            if any(kw in user_input for kw in keywords):
                matched_slug = slug
                break
        if matched_slug is None:
            matched_slug = "generic-defect"

        template_path = os.path.join(templates_dir, matched_slug, "template.json")
        if not os.path.isfile(template_path):
            logger_.warning(f"8D template.json not found at {template_path}")
            # 退而求其次：只注入 SKILL.md
            return (
                "\n\n## 🔧 8D Skill 完整工作流（已加载 SKILL.md）\n\n"
                + skill_md_content
                + "\n\n## ⚠️ 模板未找到\n"
                + f"未找到模板 {matched_slug} 的 template.json，请按 SKILL.md 通用流程执行 8D 报告。\n"
            )

        with open(template_path, "r", encoding="utf-8") as f:
            template_json = json.load(f)

        template_str = json.dumps(template_json, ensure_ascii=False, indent=2)

        # ---------- 加载 references/ 参考资料 ----------
        # SKILL.md 第六节明确指出这 3 个文件是 LLM 撰写报告时的参考资料，
        # 必须真正加载内容，否则 SKILL.md 中的引用就是死链。
        references_dir = os.path.join(project_root, "skills", "8d-skill", "references")
        reference_files = [
            ("8d_guide.md", "8D 方法论详细指南（D0-D8 每步关键活动、输出物、常见错误）"),
            ("5why_examples.md", "5Why 范例库（5 个行业/缺陷类型的完整 5Why 范例 + 常见断点）"),
            ("fishbone_guide.md", "鱼骨图 6M 分析指南（每个 M 含 5+ 条排查项 + 汽车行业清单）"),
        ]
        references_section = "\n### references/ 参考资料\n"
        for ref_name, ref_desc in reference_files:
            ref_path = os.path.join(references_dir, ref_name)
            if os.path.isfile(ref_path):
                try:
                    with open(ref_path, "r", encoding="utf-8") as rf:
                        ref_content = rf.read()
                    references_section += f"\n#### {ref_name}\n> {ref_desc}\n\n{ref_content}\n"
                    logger_.info(f"8D skill loaded reference: {ref_name} ({len(ref_content)} chars)")
                except Exception as ref_e:
                    logger_.warning(f"Failed to read reference {ref_name}: {ref_e}")
            else:
                logger_.warning(f"Reference file not found: {ref_path}")

        # ---------- 拼上下文 ----------
        context = (
            "\n\n## 🔧 8D Skill 完整工作流（已加载 SKILL.md + 匹配模板 + references/）\n\n"
            f"### 已匹配模板：{matched_slug}\n"
            f"\n### SKILL.md 完整内容\n\n{skill_md_content}\n\n"
            f"### 匹配模板 template.json（请直接使用预填的 5Why 路径、6M 排查项、CA 措施、Yokoten，不要凭经验编造）\n\n```json\n{template_str}\n```\n"
            f"{references_section}\n"
            "### 执行要求\n"
            "1. 严格按 SKILL.md 中的「五、工作流」执行，禁止跳过任何一步\n"
            "2. 5Why 路径必须使用 template.json 中 d4_template.5why_path.steps 的预填答案（可基于用户实际信息微调，但不得凭空编造）\n"
            "3. 6M 排查必须使用 template.json 中 d4_template.6m_analysis 的预填项（含 finding 和 judgment）\n"
            "4. CA 措施必须使用 template.json 中 d5_d6_template.permanent_actions 的预填项\n"
            "5. Yokoten 必须使用 template.json 中 d7_template.yokoten 的预填项\n"
            "6. 输出 D0-D8 完整 8 步内容到对话，最后调用 export_xlsx_tool + export_document_tool 生成文件\n"
        )
        logger_.info(f"8D skill context loaded: SKILL.md ({len(skill_md_content)} chars) + template {matched_slug} ({len(template_str)} chars) + references (3 files)")
        return context
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception(f"Failed to load 8D skill context: {e}")
        return ""


def _load_fmea_skill_context(skill: str, user_input: str) -> str:
    """加载 PFMEA/DFMEA skill 的完整工作流上下文（SKILL.md + 匹配到的 template.json + references/）。

    当用户在前端点击「PFMEA/DFMEA分析」按钮后，selectedSkill='pfmea-dfmea-skill' 会被透传到此处。
    本函数：
      1. 读取 skills/pfmea-dfmea-skill/SKILL.md 全文
      2. 从 user_input 提取关键字，按模板匹配规则选择模板 slug
         (electronic-ecm / mechanical-assembly / surface-treatment / painting-coating / generic-fmea)
      3. 读取 skills/pfmea-dfmea-skill/templates/<slug>/template.json
      4. 读取 skills/pfmea-dfmea-skill/references/ 下的 3 个参考文件
      5. 拼成上下文文本返回，用于追加到 system_prompt 末尾

    任何异常都返回空字符串，确保不阻断主对话流程。
    """
    if not skill or skill != "pfmea-dfmea-skill":
        return ""

    try:
        import os
        import json
        import logging
        logger_ = logging.getLogger(__name__)

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        skill_md_path = os.path.join(project_root, "skills", "pfmea-dfmea-skill", "SKILL.md")
        templates_dir = os.path.join(project_root, "skills", "pfmea-dfmea-skill", "templates")

        if not os.path.isfile(skill_md_path):
            logger_.warning(f"FMEA skill SKILL.md not found at {skill_md_path}")
            return ""

        with open(skill_md_path, "r", encoding="utf-8") as f:
            skill_md_content = f.read()

        # ---------- 模板匹配 ----------
        # 规则：先按产品类别关键字匹配，无法明确分类时用 generic-fmea 兜底
        TEMPLATE_RULES = [
            (["ECU", "控制器", "传感器", "线束", "PCB", "电路", "电子", "PCBA", "IC", "LED", "模组"], "electronic-ecm"),
            (["齿轮", "轴承", "紧固件", "螺栓", "轴", "壳体", "装配", "机械", "传动"], "mechanical-assembly"),
            (["电镀", "热处理", "氧化", "表面处理", "淬火", "渗碳", "氮化"], "surface-treatment"),
            (["喷涂", "电泳", "漆面", "涂装", "喷漆", "漆膜"], "painting-coating"),
        ]

        matched_slug = None
        for keywords, slug in TEMPLATE_RULES:
            if any(kw in user_input for kw in keywords):
                matched_slug = slug
                break
        if matched_slug is None:
            matched_slug = "generic-fmea"

        template_path = os.path.join(templates_dir, matched_slug, "template.json")
        if not os.path.isfile(template_path):
            logger_.warning(f"FMEA template.json not found at {template_path}")
            return (
                "\n\n## 🔧 PFMEA/DFMEA Skill 完整工作流（已加载 SKILL.md）\n\n"
                + skill_md_content
                + "\n\n## ⚠️ 模板未找到\n"
                + f"未找到模板 {matched_slug} 的 template.json，请按 SKILL.md 通用流程执行 FMEA 报告。\n"
            )

        with open(template_path, "r", encoding="utf-8") as f:
            template_json = json.load(f)

        template_str = json.dumps(template_json, ensure_ascii=False, indent=2)

        # ---------- 加载 references/ 参考资料 ----------
        references_dir = os.path.join(project_root, "skills", "pfmea-dfmea-skill", "references")
        reference_files = [
            ("fmea_seven_step_guide.md", "FMEA 七步法详细指南（每步 DFMEA+PFMEA 双视角、目的、关键活动、输出物、常见错误、IATF 关联）"),
            ("sod_scoring_tables.md", "S/O/D 评分表（10 级）+ 完整 1000 种 AP 组合矩阵 + 评分一致性检查"),
            ("failure_chain_examples.md", "失效链案例库（5 个行业 × 3 个案例 = 15 个 FE→FM→FC 真实案例）"),
        ]
        references_section = "\n### references/ 参考资料\n"
        for ref_name, ref_desc in reference_files:
            ref_path = os.path.join(references_dir, ref_name)
            if os.path.isfile(ref_path):
                try:
                    with open(ref_path, "r", encoding="utf-8") as rf:
                        ref_content = rf.read()
                    references_section += f"\n#### {ref_name}\n> {ref_desc}\n\n{ref_content}\n"
                    logger_.info(f"FMEA skill loaded reference: {ref_name} ({len(ref_content)} chars)")
                except Exception as ref_e:
                    logger_.warning(f"Failed to read reference {ref_name}: {ref_e}")
            else:
                logger_.warning(f"Reference file not found: {ref_path}")

        # ---------- 拼上下文 ----------
        context = (
            "\n\n## 🔧 PFMEA/DFMEA Skill 完整工作流（已加载 SKILL.md + 匹配模板 + references/）\n\n"
            f"### 已匹配模板：{matched_slug}\n"
            f"\n### SKILL.md 完整内容\n\n{skill_md_content}\n\n"
            f"### 匹配模板 template.json（请直接使用预填的失效链 FE/FM/FC、PC/DC 控制措施，不要凭经验编造）\n\n```json\n{template_str}\n```\n"
            f"{references_section}\n"
            "### 执行要求\n"
            "1. 严格按 SKILL.md 中的「四、七步法详细说明」执行，禁止跳过任何一步（规划准备/结构分析/功能分析/失效分析/风险分析/优化/结果文件化）\n"
            "2. 失效链必须使用 template.json 中 failure_chains_template 的预填内容（FE→FM→FC 三级结构），可基于用户实际信息微调，但不得凭空编造\n"
            "3. S/O/D 评分必须基于 references/sod_scoring_tables.md 的标准评分表，不可主观臆断\n"
            "4. AP 必须由 S/O/D 组合查表得出（references/sod_scoring_tables.md 的 1000 种组合矩阵），严禁使用 RPN\n"
            "5. PC（预防控制）与 DC（探测控制）必须分离，不可混写\n"
            "6. S=9-10 自动识别为 CC（关键特性），S=8 且 AP=H/M 自动识别为 SC（特殊特性），同步到控制计划\n"
            "7. 输出七步法完整内容到对话（结构树/功能树/失效链/风险评级/优化措施），最后调用 generate_fmea_report_tool 生成 xlsx + docx 文件\n"
        )
        logger_.info(f"FMEA skill context loaded: SKILL.md ({len(skill_md_content)} chars) + template {matched_slug} ({len(template_str)} chars) + references (3 files)")
        return context
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception(f"Failed to load FMEA skill context: {e}")
        return ""




def _get_date_message() -> HumanMessage:
    """已废弃：日期信息现在通过 _inject_current_date() 注入 system prompt 尾部。
    保留此函数但不再使用，避免其他模块调用时报错。
    """
    # 返回一个无害的空 HumanMessage（不再包含日期信息）
    # 调用方会在后续统一清理
    return HumanMessage(content="")

# ===== 1. 定义 Agent 状态 =====
class AgentState(TypedDict):
    """
    Agent 的状态定义
    messages 使用 add_messages 策略：新消息追加而非覆盖
    retry_count: [#11] 工具重试计数
    """
    messages: Annotated[list, add_messages]
    retry_count: int

# ===== 2. 创建 LLM =====
# 主Key是否已确认失效（运行时标记，避免每次都重试失败的Key）
_primary_key_failed = False
_primary_key_lock = threading.Lock()  # [BUG FIX] 并发安全

# [优化1] LLM Client 缓存：按 (model, api_key, base_url, temperature) 缓存 ChatOpenAI 实例
# 避免每次请求新建 HTTP 连接，减少 500ms-3s 的连接建立开销
# [BUG FIX v8] 缓存加 TTL：空闲后 API 服务端关闭 TCP 连接，复用缓存实例会导致请求卡 5-30s
_llm_cache = {}  # cache_key -> {"instance": ChatOpenAI, "created_at": float}
_LLM_CACHE_TTL = 900  # 15分钟，短于代理/服务端典型空闲超时（60-120s）

def create_llm(deep_think: bool = False, fast_mode: bool = False, model_override: str = None, 
               short_response: bool = False):
    """创建 LLM 实例（启用 streaming 支持，支持备用Key自动切换）
    
    [优化1] 使用缓存：按 (model, api_key, base_url, temperature) 作为缓存 key，
    相同参数复用同一个 ChatOpenAI 实例，避免每次请求重建 HTTP 连接（TCP + TLS 握手）。
    每次新建连接耗时 500ms-3s，复用后降至 <50ms。
    
    Args:
        deep_think: 是否启用深度思考模式（使用更强的模型、先思考再回答）
        fast_mode: 是否使用快速模型（用于简单问题的快速响应）
        model_override: 强制指定模型（用于多模态等需要切换模型的场景）
        short_response: 是否为短回复场景（降低 max_tokens 加速推理）
    """
    global _primary_key_failed
    model = model_override or settings.LLM_MODEL
    
    if fast_mode and not model_override:
        # 从 FAST_MODELS 配置中选取快速模型（如当前模型已是快速模型则不切换）
        if model not in FAST_MODELS and FAST_MODELS:
            fast_model = next(iter(FAST_MODELS))
            logger.info(f"快速模式：模型从 {model} 切换到 {fast_model}")
            model = fast_model
        else:
            logger.info(f"快速模式：当前模型 {model} 已是快速模型，无需切换")
    # deep_think 不再切换模型：用户已主动选择模型，深度思考只需调整 temperature 和 max_tokens
    # 旧代码会强制切换到已失效的 glm-4-plus 等模型，导致 API 调用失败

    # 决定使用主Key还是备用Key（用锁保护并发读写）
    with _primary_key_lock:
        use_backup = _primary_key_failed and bool(settings.LLM_API_KEY_BACKUP)
    
    # [DeepSeek] 检测是否为 DeepSeek 模型，自动切换火山引擎 API
    is_deepseek = model in DEEPSEEK_MODELS
    # [豆包] 检测是否为火山引擎模型（豆包Auto/DeepSeek/豆包），使用火山引擎 Coding API
    is_volcengine = model in VOLCENGINE_MODELS
    # [千问] 检测是否为千问模型，使用阿里云 DashScope API
    is_qwen = model in QWEN_MODELS
    # [MiMo] 检测是否为MiMo模型，使用小米API
    is_mimo = model in MIMO_MODELS
    # [GLM] 检测是否为GLM模型，使用阿里云百炼平台（兼容模式代理智谱模型）
    is_glm = model in GLM_MODELS
    
    if is_volcengine and settings.DEEPSEEK_API_KEY:
        api_key = settings.DEEPSEEK_API_KEY
        base_url = settings.DEEPSEEK_BASE_URL
        logger.info(f"火山引擎模型检测到（{model}），使用火山引擎 Coding API: {base_url}")
    elif is_qwen and settings.QWEN_API_KEY:
        api_key = settings.QWEN_API_KEY
        base_url = settings.QWEN_BASE_URL
        logger.info(f"千问模型检测到（{model}），使用阿里云 DashScope API: {base_url}")
    elif is_mimo and settings.MIMO_API_KEY:
        api_key = settings.MIMO_API_KEY
        base_url = settings.MIMO_BASE_URL
        logger.info(f"MiMo模型检测到（{model}），使用小米MiMo API: {base_url}")
    elif is_glm and settings.GLM_API_KEY:
        api_key = settings.GLM_API_KEY
        base_url = settings.GLM_BASE_URL
        logger.info(f"GLM模型检测到（{model}），使用火山引擎Ark: {base_url}")
    # [视觉模型] 无论当前选什么模型，视觉理解始终走智谱AI专用配置
    elif model in VISION_MODELS:
        api_key = VISION_API_KEY
        base_url = VISION_BASE_URL
        logger.info(f"视觉模型检测到（{model}），使用智谱AI视觉专用API: {base_url}")
    else:
        api_key = settings.LLM_API_KEY_BACKUP if use_backup else settings.LLM_API_KEY
        base_url = settings.LLM_BASE_URL_BACKUP if use_backup else settings.LLM_BASE_URL
    temperature = 0.7 if deep_think else 0.6
    
    # 智能 max_tokens：保证模型有足够输出空间，避免回答被截断
    if short_response:
        max_tokens = 4096   # 短回复场景（闲聊等），4096 足够且不会过度截断
    elif deep_think:
        max_tokens = 16384  # 深度思考需要充足输出空间
    else:
        max_tokens = 16384  # Agent模式（8D/DFMEA复杂报告，原8192）
    
    # [性能优化] request_timeout 分档：
    # - 短回复 45s（足够且不会让用户等太久）
    # - 正常 120s（复杂任务如DFMEA需要长时间生成）
    # - 深度思考 180s
    if short_response:
        request_timeout = 45
    elif deep_think:
        request_timeout = 180
    else:
        request_timeout = 120

    # [优化1] 检查缓存，复用已有的 ChatOpenAI 实例（带 TTL 检查）
    cache_key = (model, api_key, base_url, temperature)
    if cache_key in _llm_cache:
        entry = _llm_cache[cache_key]
        if time.time() - entry["created_at"] < _LLM_CACHE_TTL:
            logger.debug(f"LLM Client 缓存命中: model={model}")
            return entry["instance"]
        else:
            # [BUG FIX v8] TTL 过期，丢弃旧实例（TCP 连接已死），下面创建新的
            logger.info(f"LLM Client 缓存过期（>{_LLM_CACHE_TTL}s），重新创建: model={model}")
            del _llm_cache[cache_key]

    if use_backup:
        logger.info(f"使用备用API Key（主Key已失效）: {base_url}")
    else:
        logger.info(f"使用主API Key: {base_url}")

    # [429 自动重试] 添加 2s 初始超时用于连接检测，配合 openai 库内置重试
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        streaming=True,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        # [重要] 不设置 max_retries，避免超时时指数退避重试放大响应时间
        # 复杂任务（DFMEA等）LLM生成需要60-120s，重试会导致200-300s的卡死
    )
    _llm_cache[cache_key] = {"instance": llm, "created_at": time.time()}
    logger.info(f"LLM Client 已创建并缓存: model={model}, max_tokens={max_tokens}, timeout={request_timeout}s, 缓存数量={len(_llm_cache)}")
    return llm

def _check_and_switch_to_backup(error_exception):
    """检测到401错误时，自动切换到备用Key"""
    global _primary_key_failed
    error_str = str(error_exception).lower()
    if ("401" in error_str or "authentication" in error_str or "令牌" in error_str) and settings.LLM_API_KEY_BACKUP:
        with _primary_key_lock:
            if not _primary_key_failed:
                _primary_key_failed = True
                logger.warning(f"⚠️ 主API Key认证失败(401)，已自动切换到备用Key: {settings.LLM_BASE_URL_BACKUP}")
        return True
    return False

def reset_primary_key():
    """重置主Key状态（更换Key后调用）"""
    global _primary_key_failed
    with _primary_key_lock:
        _primary_key_failed = False

# ===== 3. 构建 Agent 图 =====

# [性能优化 2] 并行工具执行节点
# 标准 ToolNode 顺序执行每个 tool_call，当 LLM 一次返回多个互不依赖的工具调用时
# （如同时查文档+查员工），串行执行浪费了网络等待时间。
# ParallelToolNode 使用 asyncio.gather 并行执行所有 tool_call，
# 多工具轮次延迟降低 40-50%。
class ParallelToolNode:
    """并行工具执行节点，替代 LangGraph 默认的 ToolNode"""
    
    def __init__(self, tools, messages_key="messages"):
        self._tool_node = ToolNode(tools, messages_key=messages_key)
    
    async def __call__(self, state):
        from langchain_core.messages import AIMessage
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        
        if not (isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls):
            return await self._tool_node.ainvoke(state)
        
        tool_calls = last_message.tool_calls
        
        # 单个工具调用：直接走标准 ToolNode
        if len(tool_calls) <= 1:
            return await self._tool_node.ainvoke(state)
        
        # 多个工具调用：并行执行
        logger.info(f"[性能优化 2] 并行执行 {len(tool_calls)} 个工具: {[tc.get('name', '?') for tc in tool_calls]}")
        
        async def _invoke_single(call):
            """对单个 tool_call 执行 ToolNode"""
            single_state = copy.deepcopy(state)
            # 构造只有当前 tool_call 的 AIMessage
            single_ai = AIMessage(
                content="",
                tool_calls=[call],
                id=last_message.id,
            )
            single_state["messages"][-1] = single_ai
            try:
                result = await self._tool_node.ainvoke(single_state)
                return result.get("messages", [])
            except Exception as e:
                from langchain_core.messages import ToolMessage
                logger.error(f"工具 {call.get('name', '?')} 并行执行失败: {e}")
                return [ToolMessage(
                    content=f"工具执行失败: {str(e)}",
                    tool_call_id=call.get("id", ""),
                    name=call.get("name", "unknown"),
                )]
        
        # asyncio.gather 并行执行所有工具
        results = await asyncio.gather(*[_invoke_single(call) for call in tool_calls])
        
        # 按 tool_calls 原始顺序展平结果
        all_tool_messages = []
        for msgs in results:
            all_tool_messages.extend(msgs)
        
        return {"messages": all_tool_messages}


def create_agent_graph(web_search: bool = False):
    """
    构建 LangGraph Agent 执行图

    Args:
        web_search: 是否启用联网搜索工具

    流程：用户输入 → LLM 思考 → 是否调用工具？
           ├─ 是 → 执行工具 → 回到 LLM 思考（循环，最多8轮）
           └─ 否 → 输出回答 → 结束
    """
    llm = create_llm()
    tools = get_tools(web_search=web_search)
    llm_with_tools = llm.bind_tools(tools)
    system_prompt = SYSTEM_PROMPT_WITH_WEB_SEARCH if web_search else SYSTEM_PROMPT
    # [Prompt Caching] 不在 system prompt 尾部注入日期（会破坏前缀缓存）
    # 改为在消息中插入日期条，system prompt 前缀 100% 稳定

    async def think(state: AgentState):
        """LLM 思考：分析用户问题，决定是否调用工具
        
        [BUG FIX v7] 改回 async + ainvoke()：
        - sync invoke() 在 ThreadPoolExecutor 中执行时，on_chat_model_stream 事件跨线程转发
        - 长回复（DFMEA 5000+ token）大量流式事件丢失 → full_response 为空
        - 触发 fallback agent.ainvoke() 重新执行整轮 → 做两遍，60-120s 卡死
        - async ainvoke() 事件直接在事件循环触发，token逐个输出，不走 fallback
        - 多轮工具场景每轮多 3-5s 事件循环开销，但远小于 fallback 的 60-120s
        """
        if _is_session_cancelled():
            logger.warning("检测到会话已取消，跳过 LLM 调用")
            raise RuntimeError("Session cancelled by user")
        messages = state["messages"]
        # [质量修复] 日期已通过 _inject_current_date() 注入 system_prompt 尾部，不再插入假 HumanMessage
        system_msg = SystemMessage(content=system_prompt)
        response = await llm_with_tools.ainvoke([system_msg] + messages)
        return {"messages": [response]}

    tool_node = ParallelToolNode(tools, messages_key="messages")

    def should_continue(state: AgentState):
        """判断是否需要继续调用工具"""
        messages = state["messages"]
        retry_count = state.get("retry_count", 0)
        tool_message_count = sum(1 for m in messages if isinstance(m, ToolMessage))

        if tool_message_count >= MAX_TOOL_ROUNDS:
            logger.info(f"Agent 工具调用已达上限 {MAX_TOOL_ROUNDS} 轮，强制结束")
            return END

        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            if tool_message_count > 0:
                for msg in reversed(messages):
                    if isinstance(msg, ToolMessage):
                        tool_result = msg.content if isinstance(msg.content, str) else str(msg.content)
                        if any(err in tool_result for err in RETRYABLE_TOOL_ERRORS):
                            if retry_count < MAX_TOOL_RETRIES:
                                logger.info(f"Agent 检测到工具错误，第 {retry_count + 1} 次重试")
                                return "act"
                            else:
                                logger.info(f"Agent 工具重试已达上限 {MAX_TOOL_RETRIES} 次，继续执行")
                        break
            return "act"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("think", think)
    graph.add_node("act", tool_node)
    graph.set_entry_point("think")
    graph.add_conditional_edges("think", should_continue, {"act": "act", END: END})
    graph.add_edge("act", "think")
    return graph.compile()

# ===== 4. Agent 实例管理 =====
_agent_graph = None
_agent_web_search = False

def get_agent(web_search: bool = False):
    """获取 Agent 实例（懒加载，根据 web_search 参数决定是否包含联网搜索工具）"""
    global _agent_graph, _agent_web_search
    if _agent_graph is None or _agent_web_search != web_search:
        _agent_graph = create_agent_graph(web_search=web_search)
        _agent_web_search = web_search
    return _agent_graph

def reset_agent():
    """重置 Agent 实例（切换模型后调用，下次对话会自动重建）
    
    [优化] 不再 _llm_cache.clear()，保留其他模型的 LLM Client 缓存：
    - LLM 缓存按 (model, api_key, base_url, temperature) 隔离，切换模型不影响其他模型的缓存
    - 保留缓存后，用户在多个模型间切换时无需重复建立 TCP+TLS 连接（省 500ms~3s）
    - 过期清理由 cleanup_stale_caches() 按 TTL=15min 自动处理
    """
    global _agent_graph, _agent_prompt_graph_cache
    _agent_graph = None
    # [优化] 不再 clear _llm_cache：其他模型的缓存应保留，切回时直接命中
    # 旧代码 _llm_cache.clear() 导致每次切换模型都清空全部缓存，命中率仅~50%
    _agent_prompt_graph_cache.clear()  # Agent Graph 绑定模型，必须清空重建
    _agent_prompt_graph_timestamps.clear()  # 清空缓存时间戳


def cleanup_stale_caches():
    """[性能修复] 定期清理过期的缓存，防止长时间运行后内存增长
    
    由 main.py 的定期清理任务每5分钟调用一次。
    清理内容：
    1. 超过30分钟未使用的 Agent Graph 缓存
    2. [v7] 超过 TTL 或已完成的取消事件条目
    3. [v8] 超过 TTL 的 LLM Client 缓存（TCP连接空闲后被服务端关闭，必须重建）
    """
    _cleanup_stale_graph_cache()
    _cleanup_stale_cancel_events()  # [v7] 清理超时/已完成的取消事件
    
    # [v8 修复] 清理超过 TTL 的 LLM Client 缓存
    # 旧代码 guard len(_llm_cache) > 2 在典型单模型场景下永远不执行
    # 现在按 TTL 逐项清理
    global _llm_cache
    now = time.time()
    stale = [k for k, v in _llm_cache.items() if now - v["created_at"] > _LLM_CACHE_TTL]
    for k in stale:
        del _llm_cache[k]
    if stale:
        logger.info(f"[缓存清理] LLM Client 缓存清理了 {len(stale)} 个过期实例（>{_LLM_CACHE_TTL}s），剩余 {len(_llm_cache)}")

# [性能修复] Agent Graph 缓存过期检查：超过30分钟未使用的缓存自动清理
_AGENT_GRAPH_CACHE_TTL = 1800  # 30分钟
_agent_prompt_graph_timestamps = {}  # cache_key -> last_access_time

def _cleanup_stale_graph_cache():
    """清理超过 TTL 未使用的 Agent Graph 缓存，防止长时间运行后内存增长"""
    now = time.time()
    stale_keys = [k for k, t in _agent_prompt_graph_timestamps.items() if now - t > _AGENT_GRAPH_CACHE_TTL]
    for k in stale_keys:
        if k in _agent_prompt_graph_cache:
            del _agent_prompt_graph_cache[k]
        del _agent_prompt_graph_timestamps[k]
    if stale_keys:
        logger.info(f"[缓存清理] 清理了 {len(stale_keys)} 个过期 Agent Graph 缓存（>{_AGENT_GRAPH_CACHE_TTL}s未使用）")

def _build_agent_prompt(agent_task: str, web_search: bool = False, agent_id: str = None) -> str:
    """根据智能体的任务描述构建专属系统提示词
    
    智能体的任务描述将作为角色定义的优先内容，
    覆盖默认的「小智」角色，但保留工具使用指南和安全边界。
    
    关键改进：当智能体有自定义任务描述时，强制要求优先检索知识库，
    避免LLM将专业问题误判为"通用问题"而直接回答。
    
    新增：注入该智能体的关键词问题列表，让模型知道何时自动导出文件。
    """
    base_prompt = SYSTEM_PROMPT_WITH_WEB_SEARCH if web_search else SYSTEM_PROMPT
    
    # 注入该智能体的关键词问题列表
    keywords_section = get_agent_keywords_section(agent_id) if agent_id else ""
    
    custom_header = f"""# 角色

{agent_task}

## 身份
- 你的角色由上述定义决定，请严格按照任务描述中的角色定位和行为规则来行动
- 你的核心职责是完成上述任务描述中定义的工作
- 语气与风格应与角色定位保持一致

## 重要原则：不要拒绝合理请求
在符合角色定位的前提下，用户提出的合理请求你应当尽力帮助。
**绝对不要**说"这不属于我的服务范围"或"我无法帮你"这类话——只要你能做到，就给出回答。

## 知识库优先规则（最高优先级！必须严格遵守）
作为专属智能体，你**必须优先检索知识库**来回答用户的任何专业问题，而不是直接用自身知识回答。

### 强制检索规则
1. **凡是与你角色定义（上述任务描述）相关的专业问题，必须先调用 search_documents_tool 检索知识库**，基于检索结果回答
2. **即使用户没有明确说"根据知识库回答"，你也必须自动检索知识库**——用户默认期望你从知识库获取专业信息
3. 如果知识库检索无结果，可以补充自身知识，但**必须明确标注**：「以下内容非来自知识库，仅供参考」
4. **绝对禁止**在未检索知识库的情况下，直接用自己的知识回答专业问题

### 搜索效率规则
- 同一主题只搜1次，用组合关键词，不要拆成多次搜索
- 每轮最多搜索3次，信息足够就回答
- 生成文档时搜1次拿模板后直接生成

### 判断标准
- 必须检索知识库的问题：与你的角色定义、专业领域、公司制度/流程/规范/标准相关的问题
  - 示例：「FMEA成员有哪些」「质量方针是什么」「VDA6.4有什么要求」「乌龟图怎么画」
- 也必须检索：看起来简单但可能知识库有专门记载的问题
  - 示例：「团队有哪些人」「流程是什么」「有哪些文件」
- 可以直接回答的问题：纯编程、数学计算、翻译、闲聊等与你专业领域无关的通用问题
  - 示例：「Python怎么写」「1+1等于几」「帮我翻译一下」

### 标准回答流程
```
用户提问 → 判断是否与专业领域相关？
├─ 是 → 1. 先调用 search_documents_tool 检索知识库
│       2. 基于检索结果回答，标注来源
│       3. 如果无结果，补充自身知识并标注
└─ 否 → 直接回答（通用问题）
```
{keywords_section}
"""
    
    tools_section_marker = "## 工具选择指南"
    tools_idx = base_prompt.find(tools_section_marker)
    
    if tools_idx > 0:
        preserved_section = base_prompt[tools_idx:]
        result = custom_header + preserved_section
        
        old_general_rule = """### 通用问题处理
- 编程、知识问答、写作、翻译等通用问题，**直接用自己的知识回答**，不要拒绝
- 不要说"这不是我的服务范围"——回答时依然保持专业、清晰"""
        
        new_general_rule = """### 通用问题处理规则
- **与你的专业领域相关的问题**（如质量体系、公司制度、流程规范等）：必须先检索知识库，详见上方「知识库优先规则」
- **纯通用问题**（编程、数学计算、翻译、闲聊等与专业领域无关的问题）：可以直接用自己的知识回答
- 不要说"这不是我的服务范围"、"我只处理企业事务"之类的话
- 回答通用问题时，依然保持专业、清晰的风格"""
        
        result = result.replace(old_general_rule, new_general_rule)
        
        # [Token优化] 用自定义角色替换默认"小智"身份，避免双份角色文本
        old_identity = "你是一位名为「小智」的智能助手，在企业场景下专精于文档和员工信息查询，同时也能回答通用问题，并具备 GitHub 操作、邮件发送、数据库查询等能力。"
        if old_identity in result:
            result = result.replace(old_identity, custom_task)
        return result
    else:
        return custom_header + base_prompt

def _build_chat_prompt(agent_task: str) -> str:
    """根据智能体任务描述构建Chat模式的系统提示词"""
    return f"""{agent_task}

## 核心原则
- 严格按照上述角色定义来回答问题
- 专业、简洁、友好，使用规范中文回答
- 不拒绝合理的用户请求，尽力提供有价值的帮助
- 回答要有深度和细节，不要过于简略
- 适时使用结构化格式（编号、分段、表格）组织回答

## 知识库优先规则
- 与你的专业领域相关的问题，必须优先基于知识库内容回答
- 如果知识库中没有相关信息，可以补充自身知识，但必须标注：「以下内容非来自知识库，仅供参考」
- 只有纯通用问题（编程、数学、翻译、闲聊等与专业领域无关的问题）可以直接回答

## 回答规则
- 编程问题：给出完整代码，附上关键注释和运行说明
- 知识问答：准确、详细地回答，必要时补充背景信息
- 写作任务：根据需求撰写，保持风格一致
- 翻译任务：准确翻译，保留原文的语气和风格
- 闲聊：轻松自然地回应

## 格式要求
- 使用Markdown格式组织回答
- 代码使用代码块，标注语言类型
- 涉及流程时使用有序列表
- 涉及对比时使用表格
"""

# [优化2] Agent Graph 缓存：自定义 prompt 的图按 hash 缓存，避免重复编译
# 同一个 agent_task + web_search 组合只需编译一次 LangGraph
_agent_prompt_graph_cache = {}  # cache_key -> compiled graph
_AGENT_PROMPT_CACHE_MAX_SIZE = 8  # 最多缓存 8 个不同的自定义 Agent 图

def get_agent_with_prompt(custom_system_prompt: str, web_search: bool = False):
    """获取带有自定义系统提示词的 Agent 实例
    
    [优化2] 按 prompt hash + web_search 缓存编译后的 Agent Graph，
    避免每次 agent_task 请求都重新编译 LangGraph（减少 200ms-1s）。
    同一个智能体连续对话时，直接复用已编译的图。
    """
    # 生成缓存 key
    prompt_hash = hashlib.md5(custom_system_prompt.encode()).hexdigest()[:16]
    cache_key = f"{prompt_hash}:{web_search}"
    
    if cache_key in _agent_prompt_graph_cache:
        logger.debug(f"Agent Graph 缓存命中: prompt_hash={prompt_hash}, web_search={web_search}")
        _agent_prompt_graph_timestamps[cache_key] = time.time()  # [性能修复] 更新访问时间
        return _agent_prompt_graph_cache[cache_key]
    
    llm = create_llm()
    tools = get_tools(web_search=web_search)
    llm_with_tools = llm.bind_tools(tools)

    async def think(state: AgentState):
        """LLM 思考：分析用户问题，决定是否调用工具
        
        [BUG FIX v7] 改回 async + ainvoke()，原因同上（长回复流式事件跨线程丢失 → fallback 重复执行）
        """
        if _is_session_cancelled():
            logger.warning("检测到会话已取消，跳过 LLM 调用（自定义智能体）")
            raise RuntimeError("Session cancelled by user")
        messages = state["messages"]
        # [质量修复] 日期已通过 _inject_current_date() 注入 system_prompt 尾部，不再插入假 HumanMessage
        system_msg = SystemMessage(content=custom_system_prompt)
        response = await llm_with_tools.ainvoke([system_msg] + messages)
        return {"messages": [response]}

    tool_node = ParallelToolNode(tools, messages_key="messages")

    def should_continue(state: AgentState):
        messages = state["messages"]
        retry_count = state.get("retry_count", 0)
        tool_message_count = sum(1 for m in messages if isinstance(m, ToolMessage))

        if tool_message_count >= MAX_TOOL_ROUNDS:
            return END

        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            if tool_message_count > 0:
                for msg in reversed(messages):
                    if isinstance(msg, ToolMessage):
                        tool_result = msg.content if isinstance(msg.content, str) else str(msg.content)
                        if any(err in tool_result for err in RETRYABLE_TOOL_ERRORS):
                            if retry_count < MAX_TOOL_RETRIES:
                                return "act"
                        break
            return "act"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("think", think)
    graph.add_node("act", tool_node)
    graph.set_entry_point("think")
    graph.add_conditional_edges("think", should_continue, {"act": "act", END: END})
    graph.add_edge("act", "think")
    compiled = graph.compile()
    
    # [优化2] 缓存编译结果（LRU：超过上限时移除最早的）
    if len(_agent_prompt_graph_cache) >= _AGENT_PROMPT_CACHE_MAX_SIZE:
        oldest_key = next(iter(_agent_prompt_graph_cache))
        del _agent_prompt_graph_cache[oldest_key]
        logger.debug(f"Agent Graph 缓存已满，淘汰: {oldest_key}")
    _agent_prompt_graph_cache[cache_key] = compiled
    _agent_prompt_graph_timestamps[cache_key] = time.time()  # [性能修复] 记录缓存时间
    _cleanup_stale_graph_cache()  # [性能修复] 顺便清理过期缓存
    logger.info(f"Agent Graph 已编译并缓存: prompt_hash={prompt_hash}, web_search={web_search}, 缓存数量={len(_agent_prompt_graph_cache)}")
    
    return compiled

def chat(user_input: str, session_id: str = "default", web_search: bool = False, mode: str = "agent", deep_think: bool = False, agent_id: str = None, agent_task: str = None, skill: str = None) -> str:
    """非流式对话（保留兼容）"""
    set_current_agent_id(agent_id)
    set_current_session_id(session_id)
    reset_search_count()  # 每轮新对话重置搜索计数
    
    if agent_task:
        custom_prompt = _inject_current_date(_build_agent_prompt(agent_task, web_search=web_search, agent_id=agent_id))
    elif web_search:
        custom_prompt = _inject_current_date(SYSTEM_PROMPT_WITH_WEB_SEARCH)
    else:
        custom_prompt = _inject_current_date(SYSTEM_PROMPT)
    # [方案B] 8D skill 注入：把 SKILL.md + 匹配模板追加到 system prompt 末尾
    if skill:
        # 8D/FMEA skill 注入：尝试两个加载器，命中哪个就用哪个
        skill_ctx = _load_8d_skill_context(skill, user_input) or _load_fmea_skill_context(skill, user_input)
        if skill_ctx:
            custom_prompt = custom_prompt + skill_ctx
    
    if mode == "chat":
        llm = create_llm(deep_think=deep_think)
        history = get_session_history(session_id)
        recent_messages = history.messages[-MAX_HISTORY_MESSAGES:]
        all_messages = recent_messages + [HumanMessage(content=user_input)]
        chat_prompt = _inject_current_date(_build_chat_prompt(agent_task) if agent_task else CHAT_SYSTEM_PROMPT)
        result = llm.invoke([SystemMessage(content=chat_prompt)] + all_messages)
        full_response = result.content
        history.add_message(HumanMessage(content=user_input))
        history.add_message(AIMessage(content=full_response))
        return full_response

    agent = get_agent(web_search=web_search)
    history = get_session_history(session_id)
    recent_messages = history.messages[-MAX_HISTORY_MESSAGES:]
    all_messages = recent_messages + [HumanMessage(content=user_input)]
    result = agent.invoke({"messages": all_messages, "retry_count": 0})
    ai_message = result["messages"][-1]
    history.add_message(HumanMessage(content=user_input))
    history.add_message(ai_message)
    return ai_message.content

# ===== 5. 流式对话 =====

TOOL_DISPLAY_NAMES = {
    "search_documents_tool": "搜索文档",
    "lookup_employee_tool": "查询员工",
    "list_departments_tool": "部门列表",
    "list_documents_tool": "文档列表",
    "get_document_content_tool": "获取文档全文",
    "upload_document_tool": "上传文档",
    "delete_document_tool": "删除文档",
    "modify_document_tool": "修改文档",
    "export_document_tool": "导出文档",
    "export_xlsx_tool": "导出Excel",
    "web_search_tool": "联网搜索",
    "github_api_tool": "GitHub操作",
    "send_email_tool": "发送邮件",
    "database_query_tool": "数据库查询",
}

def _extract_content(chunk) -> str:
    """从流式 chunk 中提取文本内容，处理字符串和列表两种格式
    
    某些LLM返回的content是列表格式（如包含tool_calls时），
    需要安全地提取文本部分，避免拼接错误。
    """
    content = getattr(chunk, 'content', '')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                text_parts.append(item.get('text', ''))
            elif isinstance(item, str):
                text_parts.append(item)
        return ''.join(text_parts)
    return ''

# [BUG FIX] 整体超时保护：Agent 对话最大允许时长（秒）
# 超过此时间强制结束，避免 LLM API 挂起导致服务器无响应需 Ctrl+C
AGENT_STREAM_TIMEOUT = 180  # 3分钟

async def chat_stream_generator(user_input: str, session_id: str = "default", web_search: bool = False, mode: str = "agent", deep_think: bool = False, agent_id: str = None, agent_task: str = None, skill: str = None) -> AsyncGenerator[dict, None]:
    """流式对话：逐token输出，同时显示工具调用进度
    
    性能优化：
    - 意图路由：简单问题自动走Chat模式，跳过Agent工具循环
    - 流式首Token：使用 astream_events v2 实现毫秒级首Token输出
    - 非流式回退：流式失败时自动回退到非流式，确保总能得到回复
    
    BUG FIX：
    - 添加整体超时保护，防止 LLM API 挂起导致服务器无响应
    - 确保 tool_done 和 done 事件总是发送，避免前端工具标签一直转圈
    - 跟踪已启动的工具，异常时自动发送未完成的 tool_done 事件
    - [v5] 会话级取消追踪：终止对话时真正取消 Agent 执行，不再只是停止消费事件
    """
    set_current_agent_id(agent_id)
    set_current_session_id(session_id)
    reset_search_count()  # 每轮新对话重置搜索计数
    
    # [BUG FIX v6] 获取或创建 session 级取消事件（自动取消上一个幽灵任务）
    cancel_event = _get_or_create_cancel_event(session_id)
    
    # 性能优化：意图路由 - 简单问题走Chat模式（跳过Agent循环，减少3-5秒延迟）
    if mode == "agent" and _is_simple_query(user_input) and not web_search:
        logger.info(f"意图路由：检测到简单问题，自动走Chat模式加速响应")
        mode = "chat"
    
    if mode == "chat":
        async for chunk in _chat_mode_stream(user_input, session_id, deep_think=deep_think, web_search=web_search, agent_id=agent_id, agent_task=agent_task, skill=skill):
            yield chunk
        _cleanup_session_cancel(session_id)  # [v6] 正常结束清理
        return

    # Agent模式：走Agent工具调用
    if agent_task:
        custom_system_prompt = _inject_current_date(_build_agent_prompt(agent_task, web_search=web_search, agent_id=agent_id))
    else:
        custom_system_prompt = _inject_current_date(SYSTEM_PROMPT_WITH_WEB_SEARCH if web_search else SYSTEM_PROMPT)
    # [方案B] 8D/FMEA skill 注入
    if skill:
        skill_ctx = _load_8d_skill_context(skill, user_input) or _load_fmea_skill_context(skill, user_input)
        if skill_ctx:
            custom_system_prompt = custom_system_prompt + skill_ctx
    if agent_task:
        agent = get_agent_with_prompt(custom_system_prompt, web_search=web_search)
    else:
        agent = get_agent_with_prompt(custom_system_prompt, web_search=web_search)
    history = get_session_history(session_id)
    recent_messages = history.messages[-MAX_HISTORY_MESSAGES:]
    all_messages = recent_messages + [HumanMessage(content=user_input)]

    full_response = ""
    start_time = time.time()
    
    # [BUG FIX] 跟踪已启动但未完成的工具，异常时自动发送 tool_done
    pending_tools = {}  # tool_name -> display_name

    try:
        yield {"type": "thinking", "content": "正在思考..."}

        # [性能修复 v2] 直接使用 astream_events + 每轮超时检查
        # - 去掉了无效的 _stream_with_timeout 嵌套包装（该包装名为超时保护但实际未加 wait_for）
        # - 在每轮事件循环中检查总耗时，超过 AGENT_STREAM_TIMEOUT 则抛 TimeoutError
        # - 配合 async think() + ainvoke()，流式事件直接在事件循环中触发，不走跨线程转发
        async for event in agent.astream_events(
            {"messages": all_messages, "retry_count": 0},
            version="v2",
        ):
            # [BUG FIX] 整体超时保护：每个事件都检查总耗时
            if time.time() - start_time > AGENT_STREAM_TIMEOUT:
                raise asyncio.TimeoutError()

            kind = event["event"]

            if kind == "on_chat_model_start":
                # LLM 开始生成：发送思考进度反馈
                # 在多轮工具调用场景中，每轮 think 开始都会触发此事件
                yield {"type": "thinking", "content": "正在思考..."}

            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                content = _extract_content(chunk)
                if content:
                    full_response += content
                    yield {"type": "token", "content": content}

            elif kind == "on_tool_start":
                tool_name = event.get("name", "")
                display_name = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
                pending_tools[tool_name] = display_name  # [BUG FIX] 跟踪未完成工具
                yield {"type": "tool", "name": tool_name, "display": display_name}

            elif kind == "on_tool_end":
                tool_name = event.get("name", "")
                display_name = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
                pending_tools.pop(tool_name, None)  # [BUG FIX] 标记工具已完成
                yield {"type": "tool_done", "name": tool_name, "display": display_name}
            
            # [BUG FIX] 处理工具执行出错的情况：on_tool_end 可能不会触发
            elif kind == "on_tool_error":
                tool_name = event.get("name", "")
                display_name = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
                pending_tools.pop(tool_name, None)  # 标记工具已完成（出错也算完成）
                yield {"type": "tool_done", "name": tool_name, "display": display_name}

    except asyncio.TimeoutError:
        # [BUG FIX v6] 超时时：设置取消信号 + 清理 + error + done
        _set_session_cancelled(session_id)
        logger.warning(f"Agent 流式输出超时（{AGENT_STREAM_TIMEOUT}s），强制结束，已标记 session={session_id} 为取消")
        for tool_name, display_name in pending_tools.items():
            yield {"type": "tool_done", "name": tool_name, "display": display_name}
        pending_tools.clear()
        yield {"type": "error", "content": f"请求超时（{AGENT_STREAM_TIMEOUT}秒），LLM服务响应过慢，请稍后重试"}
        # 确保保存已有的部分回复
        if full_response:
            try:
                history.add_message(HumanMessage(content=user_input))
                history.add_message(AIMessage(content=full_response))
            except Exception:
                pass
        yield {"type": "done"}
        _cleanup_session_cancel(session_id)
        return
    except asyncio.CancelledError:
        # [BUG FIX v6] 取消时：设置取消信号 → think() 的下一轮会跳过 LLM 调用
        _set_session_cancelled(session_id)
        logger.info(f"Agent 流式输出被取消（客户端断开），已标记 session={session_id} 为取消")
        for tool_name, display_name in pending_tools.items():
            yield {"type": "tool_done", "name": tool_name, "display": display_name}
        pending_tools.clear()
        yield {"type": "done"}
        _cleanup_session_cancel(session_id)
        return
    except Exception as e:
        # [BUG FIX v6] 异常时也设置取消信号，避免后续 think() 继续浪费调用
        _set_session_cancelled(session_id)
        logger.error(f"Agent 流式输出异常: {e}", exc_info=True)
        # [BUG FIX] 异常时：先发送未完成工具的 tool_done
        for tool_name, display_name in pending_tools.items():
            yield {"type": "tool_done", "name": tool_name, "display": display_name}
        pending_tools.clear()
        # 检测401认证错误，自动切换备用Key
        if _check_and_switch_to_backup(e):
            yield {"type": "error", "content": "主API Key已失效，已自动切换到备用Key，请重新提问"}
            yield {"type": "done"}
            _cleanup_session_cancel(session_id)
            return
        try:
            result = await asyncio.wait_for(
                agent.ainvoke({"messages": all_messages, "retry_count": 0}),
                timeout=60.0  # [BUG FIX] 非流式回退也加超时
            )
            ai_message = result["messages"][-1]
            full_response = ai_message.content or ""
            if full_response:
                for i in range(0, len(full_response), 3):
                    yield {"type": "token", "content": full_response[i:i+3]}
                    await asyncio.sleep(0.02)
        except asyncio.TimeoutError:
            yield {"type": "error", "content": "非流式回退也超时，请稍后重试"}
            yield {"type": "done"}
            return
        except Exception as e2:
            yield {"type": "error", "content": f"处理失败: {str(e2)}"}
            yield {"type": "done"}
            return

    # 流式输出为空时回退到非流式
    if not full_response:
        try:
            result = await asyncio.wait_for(
                agent.ainvoke({"messages": all_messages, "retry_count": 0}),
                timeout=60.0  # [BUG FIX] 非流式回退也加超时
            )
            ai_message = result["messages"][-1]
            full_response = ai_message.content or ""
            if full_response:
                for i in range(0, len(full_response), 3):
                    yield {"type": "token", "content": full_response[i:i+3]}
                    await asyncio.sleep(0.02)
            else:
                yield {"type": "error", "content": "未能获取到回复，请重试"}
        except asyncio.TimeoutError:
            yield {"type": "error", "content": "获取回复超时，请稍后重试"}
        except Exception as e3:
            yield {"type": "error", "content": f"处理失败: {str(e3)}"}

    # 保存到会话历史
    if full_response:
        try:
            history.add_message(HumanMessage(content=user_input))
            history.add_message(AIMessage(content=full_response))
        except Exception:
            pass

    elapsed = time.time() - start_time
    tool_rounds = sum(1 for m in all_messages if isinstance(m, ToolMessage))
    logger.info(f"Agent 对话完成 | 耗时={elapsed:.2f}s | 模型={settings.LLM_MODEL} | 工具轮数={tool_rounds}")

    yield {"type": "done"}
    _cleanup_session_cancel(session_id)  # [v6] 正常结束清理

async def _chat_mode_stream(user_input: str, session_id: str = "default", deep_think: bool = False, web_search: bool = False, agent_id: str = None, agent_task: str = None, skill: str = None) -> AsyncGenerator[dict, None]:
    """Chat模式：直接LLM流式对话，不经过Agent工具调用，可选联网搜索
    
    性能优化：Chat模式跳过了Agent的 Think→Act→Observe 循环，
    直接让LLM流式输出，首Token延迟从3-5秒降到0.5-1秒。
    """
    set_current_agent_id(agent_id)
    set_current_session_id(session_id)
    chat_system_prompt = _inject_current_date(_build_chat_prompt(agent_task) if agent_task else CHAT_SYSTEM_PROMPT)
    # [方案B] 8D/FMEA skill 注入
    if skill:
        skill_ctx = _load_8d_skill_context(skill, user_input) or _load_fmea_skill_context(skill, user_input)
        if skill_ctx:
            chat_system_prompt = chat_system_prompt + skill_ctx
    # [质量修复] 不再因简单问题降级模型（fast_mode 会切换到更弱的模型）
    # 保留 short_response 仅调整 max_tokens，但用户选择的模型不再被替换
    is_simple = _is_simple_query(user_input)
    llm = create_llm(deep_think=deep_think, fast_mode=False, short_response=is_simple)
    history = get_session_history(session_id)
    recent_messages = history.messages[-MAX_HISTORY_MESSAGES:]
    
    # 联网搜索：先搜索再将结果注入消息
    search_context = ""
    if web_search:
        try:
            yield {"type": "thinking", "content": "正在联网搜索..."}
            yield {"type": "tool", "name": "web_search_tool", "display": "联网搜索"}
            from app.agent.tools import web_search_tool
            # [性能修复] 使用 asyncio.to_thread 在线程池中执行同步HTTP调用，避免阻塞事件循环
            # 原代码直接调用 web_search_tool.invoke() 最多阻塞15秒，期间整个服务器无法处理任何请求
            search_result = await asyncio.to_thread(web_search_tool.invoke, user_input)
            yield {"type": "tool_done", "name": "web_search_tool", "display": "联网搜索"}
            search_context = f"\n\n【联网搜索结果】\n{search_result}\n\n请根据以上联网搜索结果回答用户问题。如果搜索结果没有相关信息，请根据自身知识回答。"
        except Exception as e:
            yield {"type": "tool_done", "name": "web_search_tool", "display": "联网搜索"}
            search_context = f"\n\n【联网搜索失败：{str(e)}】请根据自身知识回答。"
    
    enhanced_input = user_input + search_context
    all_messages = recent_messages + [HumanMessage(content=enhanced_input)]

    full_response = ""

    try:
        yield {"type": "thinking", "content": "深度思考中..." if deep_think else "正在思考..."}

        async for chunk in llm.astream([SystemMessage(content=chat_system_prompt)] + all_messages):
            content = _extract_content(chunk)
            if content:
                full_response += content
                yield {"type": "token", "content": content}

    except asyncio.TimeoutError:
        yield {"type": "error", "content": "请求超时，LLM服务响应过慢，请稍后重试"}
        return
    except Exception as e:
        # 检测401认证错误，自动切换备用Key
        if _check_and_switch_to_backup(e):
            yield {"type": "error", "content": "主API Key已失效，已自动切换到备用Key，请重新提问"}
            return
        yield {"type": "error", "content": f"处理失败: {str(e)}"}
        return

    if full_response:
        try:
            history.add_message(HumanMessage(content=user_input))
            history.add_message(AIMessage(content=full_response))
        except Exception:
            pass

    yield {"type": "done"}

async def chat_stream_generator_multimodal(multimodal_content: list, session_id: str = "default", agent_id: str = None, agent_task: str = None, skill: str = None) -> AsyncGenerator[dict, None]:
    """多模态流式对话：支持图片+文本的混合消息"""
    set_current_agent_id(agent_id)
    set_current_session_id(session_id)
    current_model = settings.LLM_MODEL
    use_model = current_model
    if current_model not in VISION_MODELS:
        use_model = DEFAULT_VISION_MODEL

    # [BUG FIX] 使用 create_llm(model_override=) 复用缓存，而非每次新建 ChatOpenAI
    # 原代码绕过缓存每次新建 HTTP 连接池，损失 500ms-3s 连接建立时间
    llm = create_llm(model_override=use_model)

    history = get_session_history(session_id)
    recent_messages = history.messages[-MAX_HISTORY_MESSAGES:]

    # 修复：[human_msg] → [human_msg]
    human_msg = HumanMessage(content=multimodal_content)
    all_messages = recent_messages + [human_msg]

    system_prompt = _inject_current_date(SYSTEM_PROMPT)
    if agent_task:
        system_prompt = _inject_current_date(f"{SYSTEM_PROMPT}\n\n## 你的专属任务\n{agent_task}\n\n请优先围绕上述任务回答用户问题，并在需要时调用相关工具搜索知识库。")
    # [方案B] 8D skill 注入（multimodal 路径）：从 multimodal_content 提取文本作为 user_input 用于模板匹配
    if skill:
        try:
            mm_text = " ".join([p.get("text", "") for p in multimodal_content if isinstance(p, dict)])
        except Exception:
            mm_text = ""
        skill_ctx = _load_8d_skill_context(skill, mm_text) or _load_fmea_skill_context(skill, mm_text)
        if skill_ctx:
            system_prompt = system_prompt + skill_ctx

    full_response = ""

    try:
        yield {"type": "thinking", "content": f"正在分析图片（使用{use_model}）..."}

        # [质量修复] 日期已通过 _inject_current_date() 注入 system_prompt 尾部
        async for chunk in llm.astream([SystemMessage(content=system_prompt)] + all_messages):
            content = _extract_content(chunk)
            if content:
                full_response += content
                yield {"type": "token", "content": content}

    except Exception as e:
        try:
            text_parts = [p["text"] for p in multimodal_content if p["type"] == "text"]
            fallback_text = "\n".join(text_parts) + "\n\n[注意：图片分析失败，请用文字描述你的问题]"
            async for event in chat_stream_generator(fallback_text, session_id):
                yield event
            return
        except Exception as e2:
            yield {"type": "error", "content": f"图片分析失败: {str(e2)}"}
            return

    if full_response:
        try:
            text_summary = " ".join([p["text"] for p in multimodal_content if p["type"] == "text"])
            history.add_message(HumanMessage(content=text_summary))
            history.add_message(AIMessage(content=full_response))
        except Exception:
            pass

    yield {"type": "done"}
