"""
用户认证管理模块
支持用户注册、登录验证，密码使用 bcrypt 加盐加密存储
支持角色管理（admin/user），管理员可在后端查看明文密码
"""
import os
import json
import hashlib

from app.config import settings


def _get_users_file() -> str:
    """获取用户数据文件路径"""
    return os.path.join(settings.DATA_DIR, "users", "users.json")


def _load_users() -> dict:
    """加载用户数据"""
    users_file = _get_users_file()
    if os.path.exists(users_file):
        with open(users_file, "r", encoding="utf-8") as f:
            users = json.load(f)
        # 兼容旧数据：为没有 role 和 password_plain 的用户补全字段
        changed = False
        for username, info in users.items():
            if "role" not in info:
                # admin 用户默认为管理员角色，其他为普通用户
                info["role"] = "admin" if username == "admin" else "user"
                changed = True
            if "password_plain" not in info:
                info["password_plain"] = ""
                changed = True
        if changed:
            _save_users(users)
        return users
    # 默认管理员账号（bcrypt hash of "admin123"）
    default_users = {
        "admin": {
            "password_hash": _hash_password("admin123"),
            "password_plain": "admin123",
            "role": "admin",
        },
        "dfsr1": {
            "password_hash": _hash_password("dfsruser1"),
            "password_plain": "dfsruser1",
            "role": "user",
        }
    }
    _save_users(default_users)
    return default_users


def _save_users(users: dict) -> None:
    """保存用户数据"""
    users_file = _get_users_file()
    os.makedirs(os.path.dirname(users_file), exist_ok=True)
    with open(users_file, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def _hash_password(password: str) -> str:
    """密码哈希 - 使用 bcrypt 加盐加密"""
    try:
        import bcrypt
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    except ImportError:
        # fallback: 如果 bcrypt 未安装，使用 SHA256 + 固定盐（不推荐，但比无盐好）
        return hashlib.sha256(f"xfagent-salt-{password}".encode()).hexdigest()


def _verify_password(password: str, password_hash: str) -> bool:
    """验证密码"""
    try:
        import bcrypt
        # 尝试 bcrypt 验证
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ImportError, ValueError):
        # fallback: SHA256 验证（兼容旧密码，需同时检查带盐和不带盐的格式）
        return password_hash == hashlib.sha256(f"xfagent-salt-{password}".encode()).hexdigest() or password_hash == hashlib.sha256(password.encode()).hexdigest()


def register_user(username: str, password: str, role: str = "user") -> dict:
    """
    注册新用户（仅后端/管理员可调用，前端不提供注册入口）

    Args:
        username: 用户名
        password: 明文密码
        role: 角色，默认为 "user"，仅管理员可设置为 "admin"

    Returns:
        dict: {"success": Bool, "message": str}
    """
    if len(username) < 2:
        return {"success": False, "message": "用户名至少2个字符"}
    if len(password) < 4:
        return {"success": False, "message": "密码至少4个字符"}
    if role not in ("admin", "user"):
        return {"success": False, "message": "无效的角色，只支持 admin 或 user"}

    users = _load_users()
    if username in users:
        return {"success": False, "message": "用户名已存在"}

    users[username] = {
        "password_hash": _hash_password(password),
        "password_plain": password,
        "role": role,
    }
    _save_users(users)
    return {"success": True, "message": f"用户 {username} 创建成功"}


def login_user(username: str, password: str) -> dict:
    """
    用户登录验证

    Returns:
        dict: {"success": Bool, "message": str, "role": str}
    """
    users = _load_users()
    if username not in users:
        return {"success": False, "message": "用户名或密码错误"}

    if not _verify_password(password, users[username]["password_hash"]):
        return {"success": False, "message": "用户名或密码错误"}

    # 如果密码是旧的 SHA256 格式，自动升级为 bcrypt
    if not users[username]["password_hash"].startswith("$2"):
        users[username]["password_hash"] = _hash_password(password)
        _save_users(users)

    user_role = users[username].get("role", "user")
    return {"success": True, "message": "登录成功", "role": user_role}


def get_user_role(username: str) -> str:
    """获取用户角色"""
    users = _load_users()
    if username in users:
        return users[username].get("role", "user")
    return ""


def is_admin(username: str) -> bool:
    """判断用户是否为管理员"""
    return get_user_role(username) == "admin"


def list_all_users() -> list:
    """
    列出所有用户信息（含明文密码），仅管理员可用

    Returns:
        list: [{"username": str, "role": str, "password_plain": str}, ...]
    """
    users = _load_users()
    result = []
    for username, info in users.items():
        result.append({
            "username": username,
            "role": info.get("role", "user"),
            "password_plain": info.get("password_plain", ""),
            "created_at": info.get("created_at", ""),
        })
    return result


def delete_user(username: str) -> dict:
    """
    删除用户（不允许删除 admin 账号）

    Returns:
        dict: {"success": Bool, "message": str}
    """
    if username == "admin":
        return {"success": False, "message": "不允许删除管理员账号"}

    users = _load_users()
    if username not in users:
        return {"success": False, "message": f"用户 {username} 不存在"}

    del users[username]
    _save_users(users)
    return {"success": True, "message": f"用户 {username} 已删除"}


def update_user_role(username: str, role: str) -> dict:
    """
    修改用户角色（仅管理员可用）

    Returns:
        dict: {"success": Bool, "message": str}
    """
    if role not in ("admin", "user"):
        return {"success": False, "message": "无效的角色，只支持 admin 或 user"}

    users = _load_users()
    if username not in users:
        return {"success": False, "message": f"用户 {username} 不存在"}

    users[username]["role"] = role
    _save_users(users)
    return {"success": True, "message": f"用户 {username} 角色已更新为 {role}"}


def reset_user_password(username: str, new_password: str) -> dict:
    """
    重置用户密码（仅管理员可用）

    Returns:
        dict: {"success": Bool, "message": str}
    """
    if len(new_password) < 4:
        return {"success": False, "message": "密码至少4个字符"}

    users = _load_users()
    if username not in users:
        return {"success": False, "message": f"用户 {username} 不存在"}

    users[username]["password_hash"] = _hash_password(new_password)
    users[username]["password_plain"] = new_password
    _save_users(users)
    return {"success": True, "message": f"用户 {username} 密码已重置"}
