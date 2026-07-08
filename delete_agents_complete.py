"""
QZAGENT 智能体彻底清理脚本
============================
功能：删除指定智能体的所有数据（配置 + 知识库 + 文档 + 索引）
运行：python delete_agents_complete.py
"""
import os
import sys
import json
import shutil

# 添加项目根目录到 path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from app.config import settings
    from app.rag.document import delete_agent_collection, list_all_collections
    from app.agent.storage import load_agents, save_agents, AGENTS_DIR
except ImportError as e:
    print(f"❌ 无法导入项目模块: {e}")
    print("请确认此脚本放在项目根目录（C:\\beifen\\QZAGENT）下运行")
    sys.exit(1)


# 要删除的 4 个智能体 ID
AGENTS_TO_DELETE = [
    "embedded-software-agent",
    "test-verification-agent",
    "equipment-production-agent",
    "standards-innovation-agent",
]


def main():
    print("=" * 60)
    print("  QZAGENT 智能体彻底清理工具")
    print("=" * 60)
    print()

    print(f"要删除的智能体 ID:")
    for aid in AGENTS_TO_DELETE:
        print(f"  - {aid}")
    print()

    # === 1. 显示当前状态 ===
    print("【1】当前知识库状态：")
    print(f"  ChromaDB 目录: {settings.CHROMA_DIR}")
    print(f"  文档目录: {settings.DOCUMENTS_DIR}")
    print()

    # 列出所有 ChromaDB collections
    print("  当前 ChromaDB collections:")
    try:
        collections = list_all_collections()
        chroma_collections = [c for c in collections if c.get('type') != 'keyword']
        if chroma_collections:
            for c in chroma_collections:
                print(f"    - {c['name']} (文档数: {c.get('count', '?')})")
        else:
            print("    (无)")
    except Exception as e:
        print(f"    读取失败: {e}")

    print()

    # 列出文档目录
    print("  当前文档目录内容:")
    if os.path.exists(settings.DOCUMENTS_DIR):
        items = os.listdir(settings.DOCUMENTS_DIR)
        if items:
            for item in items:
                item_path = os.path.join(settings.DOCUMENTS_DIR, item)
                if os.path.isdir(item_path):
                    files = os.listdir(item_path)
                    print(f"    - {item}/ ({len(files)} 个文件)")
        else:
            print("    (空)")
    else:
        print("    (目录不存在)")

    print()
    print("=" * 60)

    # === 2. 确认删除 ===
    confirm = input("\n⚠️  确认删除这 4 个智能体的所有数据？此操作不可恢复！(输入 yes 确认): ")
    if confirm.lower() != 'yes':
        print("已取消。")
        return

    print()
    print("=" * 60)
    print("【2】开始删除...")
    print("=" * 60)

    # === 3. 删除每个智能体的知识库 ===
    print("\n【3】删除智能体知识库（ChromaDB + 文档 + 索引）...")
    for agent_id in AGENTS_TO_DELETE:
        print(f"\n  处理智能体: {agent_id}")
        try:
            result = delete_agent_collection(agent_id)
            if result.get('status') == 'success':
                print(f"    ✓ {result.get('message', '已删除')}")
            else:
                print(f"    ⚠ {result.get('message', '删除失败')}")
        except Exception as e:
            print(f"    ❌ 异常: {e}")

    # === 4. 从所有用户的 agent 配置中删除这 4 个智能体 ===
    print("\n【4】从用户配置中删除智能体...")
    if os.path.exists(AGENTS_DIR):
        for fname in os.listdir(AGENTS_DIR):
            if fname.endswith('.json'):
                fpath = os.path.join(AGENTS_DIR, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        agents = json.load(f)

                    original_count = len(agents)
                    new_agents = [a for a in agents if a.get('id') not in AGENTS_TO_DELETE]
                    deleted_count = original_count - len(new_agents)

                    if deleted_count > 0:
                        with open(fpath, 'w', encoding='utf-8') as f:
                            json.dump(new_agents, f, ensure_ascii=False, indent=2)
                        print(f"  ✓ {fname}: 删除了 {deleted_count} 个智能体（剩余 {len(new_agents)} 个）")
                    else:
                        print(f"  - {fname}: 无需删除（配置里没有这 4 个智能体）")
                except Exception as e:
                    print(f"  ❌ {fname}: {e}")
    else:
        print(f"  ( agents 目录不存在: {AGENTS_DIR})")

    # === 5. 验证删除结果 ===
    print()
    print("=" * 60)
    print("【5】删除后状态验证：")
    print("=" * 60)

    # 检查 ChromaDB
    print("\n  ChromaDB 剩余 collections:")
    try:
        collections = list_all_collections()
        chroma_collections = [c for c in collections if c.get('type') != 'keyword']
        if chroma_collections:
            for c in chroma_collections:
                # 检查是否还有被删除智能体的 collection
                marker = " ❌" if any(aid in c['name'] for aid in AGENTS_TO_DELETE) else ""
                print(f"    - {c['name']} (文档数: {c.get('count', '?')}){marker}")
        else:
            print("    (空)")
    except Exception as e:
        print(f"    读取失败: {e}")

    # 检查文档目录
    print("\n  文档目录剩余:")
    if os.path.exists(settings.DOCUMENTS_DIR):
        items = os.listdir(settings.DOCUMENTS_DIR)
        if items:
            for item in items:
                # 检查是否是被删除智能体的目录
                marker = " ❌" if any(f"agent_{aid}" == item for aid in AGENTS_TO_DELETE) else ""
                print(f"    - {item}{marker}")
        else:
            print("    (空)")
    else:
        print("    (目录不存在)")

    # 检查用户配置
    print("\n  用户智能体配置:")
    if os.path.exists(AGENTS_DIR):
        for fname in os.listdir(AGENTS_DIR):
            if fname.endswith('.json'):
                fpath = os.path.join(AGENTS_DIR, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        agents = json.load(f)
                    print(f"    {fname}: {len(agents)} 个智能体")
                    for a in agents:
                        print(f"      - {a.get('name', '?')} (id={a.get('id', '?')})")
                except Exception as e:
                    print(f"    {fname}: 读取失败 ({e})")

    print()
    print("=" * 60)
    print("  ✅ 清理完成！")
    print("=" * 60)
    print()
    print("提示：")
    print("  1. 4 个智能体的知识库、文档、索引已彻底删除")
    print("  2. 所有用户的 agent 配置中已移除这 4 个智能体")
    print("  3. 重启服务后，界面只显示 6 个智能体")
    print("  4. 被删除的智能体对应的会话记录（如有）不会被删除（不影响使用）")


if __name__ == '__main__':
    main()
