"""
QZAGENT 知识库全量清理脚本
============================
功能：删除所有智能体的知识库（ChromaDB + 文档文件 + 关键词索引）
运行：python delete_all_knowledge_bases.py
"""
import os
import sys
import shutil
import json

# 添加项目根目录到 path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from app.config import settings
    from app.rag.document import list_all_collections, _get_collection_name
except ImportError as e:
    print(f"❌ 无法导入项目模块: {e}")
    print("请确认此脚本放在项目根目录（C:\\beifen\\QZAGENT）下运行")
    sys.exit(1)


def main():
    print("=" * 60)
    print("  QZAGENT 知识库全量清理工具")
    print("=" * 60)
    print()

    # === 1. 显示当前状态 ===
    print("【1】当前知识库状态：")
    print(f"  ChromaDB 目录: {settings.CHROMA_DIR}")
    print(f"  文档目录: {settings.DOCUMENTS_DIR}")
    print(f"  DATA_DIR: {settings.DATA_DIR}")
    print()

    # 列出所有 ChromaDB collections
    print("  ChromaDB collections:")
    try:
        collections = list_all_collections()
        chroma_collections = [c for c in collections if c.get('type') != 'keyword']
        keyword_indexes = [c for c in collections if c.get('type') == 'keyword']

        if chroma_collections:
            for c in chroma_collections:
                print(f"    - {c['name']} (文档数: {c.get('count', '?')})")
        else:
            print("    (无)")

        print()
        print("  关键词索引:")
        if keyword_indexes:
            for k in keyword_indexes:
                print(f"    - {k['name']} (条目数: {k.get('count', '?')})")
        else:
            print("    (无)")
    except Exception as e:
        print(f"    读取失败: {e}")

    print()

    # 列出文档目录
    print("  文档目录内容:")
    if os.path.exists(settings.DOCUMENTS_DIR):
        items = os.listdir(settings.DOCUMENTS_DIR)
        if items:
            for item in items:
                item_path = os.path.join(settings.DOCUMENTS_DIR, item)
                if os.path.isdir(item_path):
                    files = os.listdir(item_path)
                    print(f"    - {item}/ ({len(files)} 个文件)")
                else:
                    print(f"    - {item}")
        else:
            print("    (空)")
    else:
        print("    (目录不存在)")

    print()
    print("=" * 60)

    # === 2. 确认删除 ===
    confirm = input("\n⚠️  确认删除所有智能体知识库？此操作不可恢复！(输入 yes 确认): ")
    if confirm.lower() != 'yes':
        print("已取消。")
        return

    print()
    print("=" * 60)
    print("【2】开始删除...")
    print("=" * 60)

    # === 3. 删除 ChromaDB collections ===
    print("\n【3】删除 ChromaDB collections...")
    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.CHROMA_DIR)

        # 重新列出所有 collections
        all_collections = client.list_collections()
        deleted_count = 0
        for c in all_collections:
            try:
                name = c.name
                client.delete_collection(name)
                print(f"  ✓ 已删除 collection: {name}")
                deleted_count += 1
            except Exception as e:
                print(f"  ✗ 删除失败 {c.name}: {e}")

        if deleted_count == 0:
            print("  (无 collection 需要删除)")
        else:
            print(f"  共删除 {deleted_count} 个 collection")
    except Exception as e:
        print(f"  ❌ ChromaDB 操作失败: {e}")

    # === 4. 删除文档文件 ===
    print("\n【4】删除文档文件...")
    if os.path.exists(settings.DOCUMENTS_DIR):
        try:
            # 列出所有 agent 子目录
            for item in os.listdir(settings.DOCUMENTS_DIR):
                item_path = os.path.join(settings.DOCUMENTS_DIR, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"  ✓ 已删除目录: {item}/")
                else:
                    os.remove(item_path)
                    print(f"  ✓ 已删除文件: {item}")
        except Exception as e:
            print(f"  ❌ 删除文档失败: {e}")
    else:
        print("  (文档目录不存在)")

    # === 5. 删除关键词索引 ===
    print("\n【5】删除关键词索引...")
    keyword_index_dir = os.path.join(os.path.dirname(settings.CHROMA_DIR), 'keyword_index')
    if not os.path.exists(keyword_index_dir):
        # 尝试其他路径
        keyword_index_dir = os.path.join(settings.DATA_DIR, 'keyword_index')

    if os.path.exists(keyword_index_dir):
        try:
            for fname in os.listdir(keyword_index_dir):
                fpath = os.path.join(keyword_index_dir, fname)
                if fname.startswith('index_') and fname.endswith('.json'):
                    os.remove(fpath)
                    print(f"  ✓ 已删除: {fname}")
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath)
                    print(f"  ✓ 已删除目录: {fname}/")
                else:
                    os.remove(fpath)
                    print(f"  ✓ 已删除: {fname}")
        except Exception as e:
            print(f"  ❌ 删除关键词索引失败: {e}")
    else:
        print("  (关键词索引目录不存在)")

    # === 6. 验证删除结果 ===
    print()
    print("=" * 60)
    print("【6】删除后状态验证：")
    print("=" * 60)

    # 检查 ChromaDB
    try:
        client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        remaining = client.list_collections()
        print(f"  ChromaDB 剩余 collections: {len(remaining)}")
        if remaining:
            for c in remaining:
                print(f"    - {c.name}")
        else:
            print("    ✓ 已清空")
    except Exception as e:
        print(f"  ChromaDB 检查失败: {e}")

    # 检查文档目录
    if os.path.exists(settings.DOCUMENTS_DIR):
        remaining_docs = os.listdir(settings.DOCUMENTS_DIR)
        print(f"  文档目录剩余: {len(remaining_docs)} 项")
        if remaining_docs:
            for d in remaining_docs:
                print(f"    - {d}")
        else:
            print("    ✓ 已清空")
    else:
        print("  文档目录: 不存在 ✓")

    # 检查关键词索引
    if os.path.exists(keyword_index_dir):
        remaining_kw = [f for f in os.listdir(keyword_index_dir) if f.startswith('index_')]
        print(f"  关键词索引剩余: {len(remaining_kw)} 项")
        if remaining_kw:
            for k in remaining_kw:
                print(f"    - {k}")
        else:
            print("    ✓ 已清空")
    else:
        print("  关键词索引目录: 不存在 ✓")

    print()
    print("=" * 60)
    print("  ✅ 清理完成！")
    print("=" * 60)
    print()
    print("提示：")
    print("  1. 智能体配置（app/data/agents/*.json）未删除，智能体本身还在")
    print("  2. 重启服务后，各智能体的知识库将是空的")
    print("  3. 用户需要重新上传文档来重建知识库")


if __name__ == '__main__':
    main()
