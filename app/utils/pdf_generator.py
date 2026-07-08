"""
PDF 生成工具模块
使用 reportlab 生成 PDF，支持中文字体
如果 reportlab 不可用，回退到 fpdf2，再不可用则降级为 txt
"""
import os
import re
import platform
import logging

logger = logging.getLogger(__name__)

CHINESE_FONT_PATHS = {
    "Windows": [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ],
    "Linux": [
        "/usr/share/fonts/truetype/chinese/SarasaMonoSC-Regular.ttf",
        "/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ],
    "Darwin": [
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ],
}

# Emoji 过滤正则：移除 PDF 字体无法渲染的字符
# 注意：范围必须精确，不能包含中文字符区间(U+4E00-U+9FFF)
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # 表情符号
    "\U0001F300-\U0001F5FF"  # 杂项符号和象形文字
    "\U0001F680-\U0001F6FF"  # 交通和地图符号
    "\U0001F1E0-\U0001F1FF"  # 旗帜
    "\U00002702-\U000027B0"  # Dingbats (✅✓✗等)
    "\U00002460-\U000024FF"  # Enclosed Alphanumerics (⑀⑁⑂等)
    "\U0001F100-\U0001F1FF"  # Enclosed Alphanumeric Supplement
    "\U0001F200-\U0001F2FF"  # Enclosed Ideographic Supplement (🉐🉑等)
    "\U0001f900-\U0001f9FF"  # 补充表情符号
    "\U0001fa00-\U0001fa6F"  # Chess symbols
    "\U0001fa70-\U0001faFF"  # Symbols and Pictographs Extended-A
    "\U00002600-\U000026FF"  # Misc symbols (☀☁☂等)
    "\U00002700-\U000027BF"  # Dingbats
    "\U0000FE00-\U0000FE0F"  # Variation Selectors
    "\U0000200D"             # Zero Width Joiner
    "\U00002B50"             # ⭐
    "\U000023E9-\U000023F3"  # Misc symbols
    "\U000023F0-\U000023FA"  # ⏰⏱⏲⏳
    "\U000025AA-\U000025FE"  # 几何形状
    "\U00002934-\U00002935"  # ⤴⤵
    "\U00002B05-\U00002B07"  # ⬅⬆⬇
    "]+"
)


def _strip_emoji(text: str) -> str:
    """移除文本中的 emoji 和字体不支持的特殊字符"""
    cleaned = _EMOJI_RE.sub("", text)
    cleaned = "".join(c for c in cleaned if ord(c) <= 0xFFFF)
    return cleaned


def find_chinese_font():
    """查找系统中可用的中文字体"""
    system = platform.system()
    for path in CHINESE_FONT_PATHS.get(system, []):
        if os.path.exists(path):
            return path
    return None


def generate_pdf(text, output_path, title="修改后的文档"):
    """
    生成 PDF 文件

    Args:
        text: 文本内容
        output_path: 输出路径
        title: 文档标题

    Returns:
        tuple: (success: bool, actual_path: str)
    """
    try:
        return _generate_pdf_reportlab(text, output_path, title)
    except ImportError:
        logger.warning("reportlab 未安装，回退到 fpdf2")
    except Exception as e:
        logger.warning(f"reportlab PDF 生成失败: {e}，回退到 fpdf2")

    # 回退到 fpdf2
    try:
        return _generate_pdf_fpdf(text, output_path, title)
    except ImportError:
        logger.warning("fpdf2 也未安装，降级为 txt 文件")
        return _save_as_txt(text, output_path)
    except Exception as e:
        logger.error(f"PDF 生成失败: {e}，降级为 txt 文件")
        return _save_as_txt(text, output_path)


def _generate_pdf_reportlab(text, output_path, title="修改后的文档"):
    """使用 reportlab 生成 PDF"""
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_path = find_chinese_font()
    if not font_path:
        raise RuntimeError("未找到中文字体")

    font_name = "ChineseFont"
    pdfmetrics.registerFont(TTFont(font_name, font_path))

    text = _strip_emoji(text)

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    title_style = ParagraphStyle('Title', fontName=font_name, fontSize=16, leading=22, alignment=1)
    body_style = ParagraphStyle('Body', fontName=font_name, fontSize=10, leading=14)

    story = []
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 6 * mm))

    for line in text.split("\n"):
        if not line.strip():
            story.append(Spacer(1, 3 * mm))
            continue
        safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe_line, body_style))

    doc.build(story)
    return True, output_path


def _generate_pdf_fpdf(text, output_path, title="修改后的文档"):
    """使用 fpdf2 生成 PDF（回退方案）"""
    from fpdf import FPDF

    font_path = find_chinese_font()
    if not font_path:
        raise RuntimeError("未找到中文字体")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("ChineseFont", "", font_path)
    pdf.set_font("ChineseFont", "", 12)

    text = _strip_emoji(text)
    for line in text.split("\n"):
        if not line.strip():
            pdf.ln(6)
            continue
        try:
            pdf.multi_cell(0, 7, line)
        except Exception:
            pass

    pdf.output(output_path)
    return True, output_path


def generate_chat_pdf(messages: list, session_id: str, agent_name: str = "") -> bytes:
    """
    生成对话导出 PDF（返回 bytes），使用 reportlab

    Args:
        messages: 对话消息列表 [{"role": "user"/"assistant", "content": "..."}]
        session_id: 会话 ID
        agent_name: 当前智能体名称（用于标题）

    Returns:
        PDF 文件的 bytes 内容
    """
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from io import BytesIO

    font_path = find_chinese_font()
    if not font_path:
        raise RuntimeError("未找到中文字体，无法生成 PDF")

    font_name = "ChineseFont"
    pdfmetrics.registerFont(TTFont(font_name, font_path))

    # 定义样式
    title_style = ParagraphStyle(
        'Title', fontName=font_name, fontSize=16, leading=22,
        alignment=1, spaceAfter=4 * mm
    )
    info_style = ParagraphStyle(
        'Info', fontName=font_name, fontSize=9, leading=13,
        alignment=1, textColor=HexColor('#888888'), spaceAfter=8 * mm
    )
    role_user_style = ParagraphStyle(
        'RoleUser', fontName=font_name, fontSize=11, leading=16,
        textColor=HexColor('#1a1a1a'), spaceAfter=2 * mm
    )
    role_assistant_style = ParagraphStyle(
        'RoleAssistant', fontName=font_name, fontSize=11, leading=16,
        textColor=HexColor('#1976D2'), spaceAfter=2 * mm
    )
    content_style = ParagraphStyle(
        'Content', fontName=font_name, fontSize=10, leading=14,
        spaceAfter=2 * mm
    )

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=15 * mm, bottomMargin=15 * mm)

    story = []
    # 标题
    display_title = f"{agent_name} 对话记录" if agent_name else "东风科技研发智能体 对话记录"
    story.append(Paragraph(display_title, title_style))
    story.append(Paragraph(f"Session: {session_id[:12]}", info_style))

    for msg in messages:
        role = "用户" if msg["role"] == "user" else "助手"
        content = msg.get("content", "")

        # 预处理：移除 emoji
        content = _strip_emoji(content)

        # 转义 XML 特殊字符（reportlab Paragraph 使用 XML 标记）
        role_safe = role.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # 角色标签
        role_style = role_user_style if msg["role"] == "user" else role_assistant_style
        story.append(Paragraph(f"<b>{role_safe}：</b>", role_style))

        # 内容：按行分段，避免超长段落
        for line in content.split("\n"):
            if not line.strip():
                story.append(Spacer(1, 2 * mm))
                continue
            safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            try:
                story.append(Paragraph(safe_line, content_style))
            except Exception:
                # 最终降级：逐字符过滤
                very_safe = "".join(
                    c for c in line if c.isprintable() or c in " \t"
                )
                very_safe = very_safe.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                if very_safe.strip():
                    try:
                        story.append(Paragraph(very_safe, content_style))
                    except Exception:
                        pass

        # 分隔线
        story.append(Spacer(1, 2 * mm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#cccccc')))
        story.append(Spacer(1, 2 * mm))

    doc.build(story)
    return buf.getvalue()


def _save_as_txt(text, output_path):
    """降级方案：保存为 txt 文件"""
    if output_path.lower().endswith(".pdf"):
        output_path = output_path[:-4] + ".txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return True, output_path
