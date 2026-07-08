/**
 * ForgeAgent 前端应用
 * 主脚本 - 处理认证、聊天、会话管理、导出等功能
 */

// ===== [BUG FIX] Browser Back Button Support =====
// The app uses CSS-based SPA navigation (show/hide elements) without updating
// browser history. This causes the back button to exit the app entirely
// instead of returning to the login page. Fix: use history.pushState to
// record page transitions, and listen for popstate to handle back/forward.

let currentUser = null;
let userRole = null;
let authToken = null;
let selectedFile = null;
let selectedFileBase64 = null;
let isLoading = false;
let currentChatId = null;
let allChats = [];
let renamingChatId = null;
let currentAbortController = null;
let userScrolledUp = false;
let lastMessageText = '';
let webSearchEnabled = false;
let deepThinkEnabled = false;
let currentMode = 'agent';
let selectedSkill = null;  // 当前选中的技能（如 '8d-skill'）
const MAX_FILE_SIZE = 50 * 1024 * 1024;

// [#12] 同步防抖锁：避免短时间内重复调用 syncAgentsFromServer
let _syncAgentsLock = false;
let _syncAgentsLastTime = 0;
const _SYNC_AGENTS_COOLDOWN = 5000;  // 5秒内不重复同步
// [#12] 上次同步到服务器的智能体数据指纹（用于检测数据是否真变了）
let _lastSyncedAgentsHash = '';

// ===== Agent Management =====
// 允许的智能体ID白名单（与后端 storage.py 保持一致）
// 顺序即侧边栏固定显示顺序，点击等操作不会改变
const ALLOWED_AGENT_IDS = [
    'dfmea-risk-agent',            // 唯一智能体：填写体系调研
];

// 按 ALLOWED_AGENT_IDS 定义的顺序排序智能体列表（保证侧边栏顺序永远固定）
function sortAgentsByFixedOrder(agents) {
    const orderMap = {};
    ALLOWED_AGENT_IDS.forEach((id, idx) => { orderMap[id] = idx; });
    return agents.sort((a, b) => {
        const oa = orderMap[a.id] !== undefined ? orderMap[a.id] : 9999;
        const ob = orderMap[b.id] !== undefined ? orderMap[b.id] : 9999;
        return oa - ob;
    });
}

// 每个智能体的欢迎页配置（名称、描述、推荐问题）
const AGENT_WELCOME_CONFIG = {
    'dfmea-risk-agent': {
        name: '全质体系智能体',
        desc: '体系调研完毕，请点击左边侧边栏的按钮（如：一键生成智能手册），系统将自动生成需要的文件',
        questions: []
    },
};

// 注意：AGENT_WELCOME_CONFIG 的键顺序无关，显示顺序由 sortAgentsByFixedOrder 控制

// 获取智能体欢迎页配置（内置+自定义智能体）
function getAgentWelcomeConfig(agentId) {
    if (AGENT_WELCOME_CONFIG[agentId]) return AGENT_WELCOME_CONFIG[agentId];
    const agent = myAgents.find(a => a.id === agentId);
    if (agent) {
        return {
            name: agent.name,
            desc: agent.task || '专属AI智能体',
            questions: ['介绍一下你的能力', '帮我分析一个问题', '给我一些建议', '常见的注意事项有哪些？']
        };
    }
    return null;
}

function forceCorrectAgents() {
    let existing = [];
    try { existing = JSON.parse(localStorage.getItem('forgeAgents') || '[]'); } catch(e) { existing = []; }
    const existingMap = {};
    existing.forEach(a => { existingMap[a.id] = a; });

    const defaults = {
        'dfmea-risk-agent': { name: '全质体系智能体', task: '你是体系智能体的填写体系调研模块。用户通过填写企业信息，系统自动生成规范化的体系调研报告，作为后续一键生成手册、程序文件、三层次文件、记录表格和不合格项整改的基础数据。', summary: '全质体系智能体' }
    };

    const correctAgents = Object.keys(defaults).map(id => {
        const def = defaults[id];
        const ex = existingMap[id];
        return {
            id: id,
            name: ex ? (ex.name || def.name) : def.name,
            task: ex ? (ex.task || def.task) : def.task,
            summary: ex ? (ex.summary || def.summary) : def.summary,
            mode: 'agent',
            created_at: ex ? (ex.created_at || 0) : 0,
            updated_at: ex ? (ex.updated_at || null) : null,
            chat_ids: ex ? (ex.chat_ids || []) : []
        };
    });

    localStorage.setItem('forgeAgents', JSON.stringify(correctAgents));
    return correctAgents;
}

function filterAgents(agents) {
    if (!agents || !Array.isArray(agents)) return sortAgentsByFixedOrder(forceCorrectAgents());
    // 保留内置智能体 + 用户动态创建的智能体（agent_ 开头）
    const filtered = agents.filter(a => ALLOWED_AGENT_IDS.includes(a.id) || (a.id && a.id.startsWith('agent_')));
    // 确保内置智能体一定存在
    const hasBuiltIn = ALLOWED_AGENT_IDS.every(id => filtered.some(a => a.id === id));
    if (!hasBuiltIn) return sortAgentsByFixedOrder(forceCorrectAgents());
    return sortAgentsByFixedOrder(filtered);
}

let myAgents = filterAgents((function() { try { return JSON.parse(localStorage.getItem('forgeAgents') || 'null'); } catch(e) { return null; } })());
let currentAgentId = null;
let agentKbUploadMode = false;

function _resolveMergeDirection(local, serverAgent) {
    // BUG FIX: Improved timestamp-based merge logic for prompt sync across browsers
    // If server has updated_at but local doesn't, prefer server data
    if (serverAgent.updated_at && !local.updated_at) return true;
    // If local has updated_at but server doesn't, prefer local data
    if (local.updated_at && !serverAgent.updated_at) return false;
    // Otherwise compare timestamps
    const localTime = local.updated_at || local.created_at || 0;
    const serverTime = serverAgent.updated_at || serverAgent.created_at || 0;
    return serverTime > localTime;
}

async function saveAgents() {
    // 过滤：只保留允许的智能体
    myAgents = filterAgents(myAgents);
    localStorage.setItem('forgeAgents', JSON.stringify(myAgents));
    // [#12] 同步到服务器：检测数据是否真变了（chat_ids变化不算，服务端不存chat_ids）
    if (currentUser && authToken) {
        try {
            const agentsForServer = myAgents.map(a => ({
                id: a.id, name: a.name, task: a.task, mode: a.mode, created_at: a.created_at, updated_at: a.updated_at
            }));
            const newHash = JSON.stringify(agentsForServer);
            if (newHash === _lastSyncedAgentsHash) {
                console.log('[saveAgents] 数据未变化，跳过POST');
                return;
            }
            _lastSyncedAgentsHash = newHash;
            const resp = await fetch('/api/v1/agents/sync', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + authToken },
                body: JSON.stringify({ agents: agentsForServer })
            });
            const data = await resp.json();
            if (data.success && data.agents && data.agents.length > 0) {
                // Merge: preserve local chat_ids, use timestamp-based comparison for name/task/updated_at
                const localAgents = JSON.parse(localStorage.getItem('forgeAgents') || '[]');
                const localMap = {};
                localAgents.forEach(a => { localMap[a.id] = a; });
                const mergedAgents = data.agents.map(serverAgent => {
                    const local = localMap[serverAgent.id];
                    if (!local) return { ...serverAgent, chat_ids: [] };
                    // 强制使用服务器名称和task，不用本地的（防止旧名称覆盖新名称）
                    return {
                        ...serverAgent,
                        name: serverAgent.name,
                        task: serverAgent.task,
                        summary: local.summary || serverAgent.summary || '',
                        updated_at: serverAgent.updated_at || null,
                        chat_ids: local.chat_ids || []
                    };
                });
                myAgents = filterAgents(mergedAgents);
                localStorage.setItem('forgeAgents', JSON.stringify(myAgents));
            }
        } catch (e) {
            console.warn('[智能体同步失败]', e);
        }
    }
}

async function syncAgentsFromServer(force = false) {
    // [#12] 防抖锁：5秒内不重复同步（除非 force=true）
    if (!force && _syncAgentsLock) return;
    const now = Date.now();
    if (!force && (now - _syncAgentsLastTime) < _SYNC_AGENTS_COOLDOWN) return;
    _syncAgentsLock = true;
    _syncAgentsLastTime = now;

    // 从服务器拉取最新智能体数据并合并（保留本地 chat_ids）
    // 修复跨浏览器同步：先GET服务器数据，再与本地比较，只有本地更新时才POST
    if (!currentUser || !authToken) { _syncAgentsLock = false; return; }
    try {
        // Step 1: GET 服务器最新数据（不发送本地数据，避免旧数据覆盖服务器）
        const getResp = await fetch('/api/v1/agents', {
            method: 'GET',
            headers: { 'Authorization': 'Bearer ' + authToken }
        });
        const getData = await getResp.json();
        
        if (getData.success && getData.agents && getData.agents.length > 0) {
            const serverAgents = getData.agents;
            const localAgents = JSON.parse(localStorage.getItem('forgeAgents') || '[]');
            const localMap = {};
            localAgents.forEach(a => { localMap[a.id] = a; });
            
            // Step 2: 比较时间戳，合并数据
            let localHasNewer = false;
            const mergedAgents = serverAgents.map(serverAgent => {
                const local = localMap[serverAgent.id];
                if (!local) return { ...serverAgent, chat_ids: [] };
                // 强制使用服务器名称和task，不用本地的（防止旧名称覆盖新名称）
                return {
                    ...serverAgent,
                    name: serverAgent.name,
                    task: serverAgent.task,
                    summary: local.summary || serverAgent.summary || '',
                    updated_at: serverAgent.updated_at || null,
                    chat_ids: local.chat_ids || []
                };
            });
            
            myAgents = filterAgents(mergedAgents);
            localStorage.setItem('forgeAgents', JSON.stringify(myAgents));
            
            // Step 3: 只有本地有更新数据时才POST到服务器
            if (localHasNewer) {
                const agentsForServer = myAgents.map(a => ({
                    id: a.id, name: a.name, task: a.task, mode: a.mode, 
                    created_at: a.created_at, updated_at: a.updated_at
                }));
                // [#12] 计算数据指纹，检测是否真变了（避免无变化的写操作）
                const newHash = JSON.stringify(agentsForServer);
                if (newHash !== _lastSyncedAgentsHash) {
                    _lastSyncedAgentsHash = newHash;
                    try {
                        await fetch('/api/v1/agents/sync', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + authToken },
                            body: JSON.stringify({ agents: agentsForServer })
                        });
                    } catch (postErr) {
                        console.warn('[智能体POST同步失败]', postErr);
                    }
                } else {
                    console.log('[sync] 数据未变化，跳过POST');
                }
            }
        }

        // Rebuild chat_ids from server data
        await rebuildChatIdsFromServer();
        renderMyAgents();
    } catch (e) {
        console.warn('[智能体同步失败]', e);
    } finally {
        _syncAgentsLock = false;
    }
}
// BUG FIX: Rebuild agent.chat_ids from server chat data to restore agent-chat associations
// after refresh/cross-browser where local chat_ids are lost
async function rebuildChatIdsFromServer() {
    if (!currentUser || !authToken) return;
    try {
        const resp = await fetch(`/api/v1/chats?username=${encodeURIComponent(currentUser)}`, { headers: apiHeaders() });
        const data = await resp.json();
        console.log('[rebuildChatIds] server chats:', data);
        if (data.success && data.chats) {
            const serverChats = data.chats;
            myAgents.forEach(agent => {
                // Find all chats where chat.agent_id matches this agent's id
                const matchingChatIds = serverChats
                    .filter(chat => chat.agent_id === agent.id)
                    .map(chat => chat.chat_id);
                console.log(`[rebuildChatIds] Agent ${agent.name} (${agent.id}): found ${matchingChatIds.length} chats`);
                // Merge: add any new server chat_ids
                const existingIds = new Set(agent.chat_ids || []);
                matchingChatIds.forEach(id => existingIds.add(id));
                agent.chat_ids = Array.from(existingIds);
            });
            localStorage.setItem('forgeAgents', JSON.stringify(myAgents));
            console.log('[rebuildChatIds] Rebuilt chat_ids from server');
        }
    } catch (e) {
        console.warn('[rebuildChatIds失败]', e);
    }
}

function generateAgentId() {
    return 'agent_' + Date.now().toString(36) + Math.random().toString(36).substr(2, 6);
}

function openAgentCreateModal() {
    document.getElementById('agentName').value = '';
    document.getElementById('agentTask').value = '';
    document.getElementById('agentCreateModal').classList.add('show');
    setTimeout(() => document.getElementById('agentName').focus(), 100);
}

function closeAgentCreateModal() {
    document.getElementById('agentCreateModal').classList.remove('show');
}

async function createAgent() {
    const name = document.getElementById('agentName').value.trim();
    const task = document.getElementById('agentTask').value.trim();
    if (!name) { showToast('请输入智能体名称'); return; }
    if (!task) { showToast('请输入任务描述'); return; }
    
    const agent = {
        id: generateAgentId(),
        name: name,
        task: task,
        mode: 'agent',
        created_at: Date.now() / 1000,
        chat_ids: []
    };
    myAgents.push(agent);
    saveAgents();
    closeAgentCreateModal();
    
    // Switch to the new agent
    await switchToAgent(agent.id);
    renderMyAgents();
    showToast(`智能体「${name}」锻造成功！`);
}

function deleteAgent(agentId) {
    const agent = myAgents.find(a => a.id === agentId);
    if (!agent) return;
    // 禁止删除内置智能体
    if (ALLOWED_AGENT_IDS.includes(agentId)) {
        showToast('内置智能体不可删除');
        return;
    }
    if (!confirm(`确定删除智能体「${agent.name}」？相关对话和知识库也将被删除。`)) return;
    
    // 先删除服务器端的知识库
    fetch(`/api/v1/agents/${encodeURIComponent(agentId)}/knowledge`, { method: 'DELETE', headers: apiHeaders() })
        .then(r => r.json())
        .then(data => console.log('[KB删除]', data))
        .catch(e => console.warn('[KB删除失败]', e));
    
    myAgents = sortAgentsByFixedOrder(myAgents.filter(a => a.id !== agentId));
    saveAgents();
    
    if (currentAgentId === agentId) {
        currentAgentId = null;
        agentKbUploadMode = false;
        document.getElementById('kbUploadToggle').classList.remove('active');
        document.getElementById('agentKbBar').style.display = 'none';
        modeChatId['agent'] = null;
        document.getElementById('chatTitle').textContent = '质量改进工程师助手';
        updateKbUploadVisibility();
        updateHeaderKbVisibility();
    }
    renderMyAgents();
    loadChatList();
    showToast('智能体已删除');
}

async function switchToAgent(agentId) {
    const agent = myAgents.find(a => a.id === agentId);
    if (!agent) return;

    // [BUG FIX #2] 切换智能体时中断正在进行的流式响应
    // 防止旧SSE流在后台继续运行导致 isLoading 锁死、新聊天无法发送消息
    stopGeneration();

    currentAgentId = agentId;

    // Force agent mode (智能体强制使用agent模式)
    if (currentMode !== 'agent') {
        switchMode('agent');
    }

    // 智能体模式默认开启联网搜索
    if (!webSearchEnabled) {
        webSearchEnabled = true;
        document.getElementById('webSearchToggle').classList.add('active');
        localStorage.setItem('webSearch', '1');
    }

    // Update header title
    document.getElementById('chatTitle').textContent = agent.name;

    // 更新知识库按钮可见性（选中智能体时显示📚）
    updateKbUploadVisibility();
    updateHeaderKbVisibility();

    // Render agents list
    renderMyAgents();
    updateGenButtonsVisibility();
    
    // 点击智能体：显示空白对话页面（含智能体欢迎信息）
    currentChatId = null;
    modeChatId['agent'] = null;
    clearChatUI();
    renderChatList();
    // 确保欢迎页可见（但如果没有调研数据，提示用户先填写）
    const welcomeEl = document.getElementById('welcomeCenter');
    if (welcomeEl) {
        const hasSurvey = localStorage.getItem('surveyData');
        if (hasSurvey) {
            // 已填写调研 → 显示正常欢迎页
            welcomeEl.style.display = '';
        } else {
            // 未填写调研 → 显示提示
            welcomeEl.style.display = '';
            welcomeEl.innerHTML = '<h2>体系智能体</h2><p>请先点击左侧"填写体系调研"填写企业信息</p><div class="quick-actions"></div>';
        }
    }
    const chatContent = document.getElementById('chatContent');
    if (chatContent) chatContent.classList.add('centered');
}

function renderMyAgents() {
    const list = document.getElementById('myAgentsList');
    if (!list) return;
    list.innerHTML = '';

    myAgents.forEach(agent => {
        const item = document.createElement('div');
        item.className = `agent-item${agent.id === currentAgentId ? ' active' : ''}`;
        item.setAttribute('data-agent-id', agent.id);
        const initial = (agent.name && agent.name[0] || '?').toUpperCase();
        item.innerHTML = `
            <div class="agent-item-info">
                <div class="agent-item-name">${escapeHtml(agent.name)}</div>
            </div>
            <button class="agent-action-btn new-chat" data-action="new-chat" data-agent-id="${agent.id}" title="新建对话" aria-label="新建对话"><svg width="22" height="22" viewBox="0 0 24 24" class="agent-new-chat-icon"><rect x="1" y="1" width="22" height="22" rx="6" ry="6" fill="#1051BF"/><path d="M9.5 6.5L18.5 12L9.5 17.5Z" fill="white"/></svg></button>
        `;
        list.appendChild(item);
    });

    // 事件委托：在列表容器上统一处理点击，避免 innerHTML 后事件丢失
    list.onclick = function(e) {
        const newChatBtn = e.target.closest('[data-action="new-chat"]');
        if (newChatBtn) {
            e.stopPropagation();
            e.preventDefault();
            const aid = newChatBtn.getAttribute('data-agent-id');
            console.log('[事件委托] 新建对话按钮点击, agentId=', aid);
            if (aid) {
                createNewChatForAgent(aid);
            }
            return;
        }
        const agentItem = e.target.closest('.agent-item');
        if (agentItem) {
            const aid = agentItem.getAttribute('data-agent-id');
            if (aid) {
                switchToAgent(aid);
                closeSidebarOnMobile();
            }
        }
    };
}

// ===== Agent Edit (disabled - prompt no longer user-editable) =====
let editingAgentId = null;

async function createNewChatForAgent(agentId) {
    console.log('[新建对话] 开始, agentId=', agentId, 'currentUser=', currentUser, 'currentMode=', currentMode);
    if (!currentUser) {
        console.warn('[新建对话] 未登录，跳过');
        showToast('请先登录');
        return;
    }

    // 切换到该智能体
    currentAgentId = agentId;
    currentMode = 'agent';
    localStorage.setItem('chatMode', 'agent');

    // 更新模式切换按钮样式
    const modeChatBtn = document.getElementById('modeChat');
    const modeAgentBtn = document.getElementById('modeAgent');
    if (modeChatBtn) modeChatBtn.classList.toggle('active', false);
    if (modeAgentBtn) modeAgentBtn.classList.toggle('active', true);

    try {
        const agent = myAgents.find(a => a.id === agentId);
        const chatTitle = agent ? agent.name : '新对话';
        console.log('[新建对话] 发送POST请求, title=', chatTitle, 'agent_id=', agentId);

        const resp = await fetch(`/api/v1/chats?username=${encodeURIComponent(currentUser)}&title=${encodeURIComponent(chatTitle)}&mode=agent&agent_id=${encodeURIComponent(agentId)}`, {
            method: 'POST',
            headers: apiHeaders()
        });
        const data = await resp.json();
        console.log('[新建对话] API返回:', JSON.stringify(data));

        if (data.success && data.chat) {
            currentChatId = data.chat.chat_id;
            modeChatId['agent'] = currentChatId;

            // 关联智能体
            if (agent) {
                if (!agent.chat_ids) agent.chat_ids = [];
                if (!agent.chat_ids.includes(data.chat.chat_id)) agent.chat_ids.push(data.chat.chat_id);
                agentActiveChatId[agentId] = data.chat.chat_id;
                saveAgentActiveChatIds();
                saveAgents();
            }

            // 刷新聊天列表
            await loadChatList();

            // 清空聊天区域，显示新对话界面
            clearChatUI();

            // 显示智能体专属欢迎页（居中模式）
            const welcomeEl = document.getElementById('welcomeCenter');
            if (welcomeEl) welcomeEl.style.display = '';
            const chatContent = document.getElementById('chatContent');
            if (chatContent) chatContent.classList.add('centered');
            updateWelcomeContent();

            // 刷新智能体列表高亮
            renderMyAgents();

            // 更新标题
            const titleEl = document.getElementById('chatTitle');
            if (titleEl && agent) titleEl.textContent = agent.name;

            // 更新知识库按钮
            updateKbUploadVisibility();
            updateHeaderKbVisibility();

            // 移动端关闭侧边栏
            closeSidebarOnMobile();

            showToast('已创建新对话');

            // 聚焦输入框
            setTimeout(() => {
                const input = document.getElementById('messageInput') || document.getElementById('msgInput');
                if (input) input.focus();
            }, 100);

            console.log('[新建对话] 完成, chatId=', currentChatId);
        } else {
            console.error('[新建对话] API返回失败:', data);
            showToast('创建对话失败');
        }
    } catch (e) {
        console.error('[新建对话] 异常:', e);
        showToast('创建对话异常: ' + e.message);
    }
}

function toggleMyAgents() {
    // No longer a collapsible section - agents are always visible in sidebar
    // This function kept for compatibility but does nothing
}

// ===== Agent KB Upload Toggle & Header KB Button Visibility =====
function updateHeaderKbVisibility() {
    const btn = document.getElementById('headerKbBtn');
    const externalKbBtn = document.getElementById('headerExternalKbBtn');
    const helpBtn = document.getElementById('headerHelpBtn');
    const skillsWrapper = document.getElementById('skillsWrapper');
    // 帮助按钮和外部知识库按钮始终显示
    if (helpBtn) helpBtn.style.display = 'inline-flex';
    if (externalKbBtn) externalKbBtn.style.display = 'inline-flex';
    // 企业内部体系文件按钮始终显示
    if (btn) btn.style.display = 'inline-flex';
    // Skills 按钮也始终显示
    if (skillsWrapper) skillsWrapper.style.display = '';
    if (currentAgentId) {
        btn.style.display = 'inline-flex';
        if (skillsWrapper) skillsWrapper.style.display = 'inline-block';
    } else {
        btn.style.display = 'none';
        if (skillsWrapper) skillsWrapper.style.display = 'none';
        // 同时关闭知识库页面
        const kbPage = document.getElementById('kbPage');
        if (kbPage && kbPage.style.display !== 'none') {
            hideKbPage();
        }
    }
}

function updateKbUploadVisibility() {
    const kbBtn = document.getElementById('kbUploadToggle');
    // 只在 agent 模式 且 选中了某个智能体 时才显示知识库上传按钮
    if (currentMode === 'agent' && currentAgentId) {
        kbBtn.style.display = '';
    } else {
        kbBtn.style.display = 'none';
        // 同时关闭知识库上传模式
        if (agentKbUploadMode) {
            agentKbUploadMode = false;
            kbBtn.classList.remove('active');
            document.getElementById('agentKbBar').style.display = 'none';
        }
    }
}

function toggleAgentKbUpload() {
    if (!currentAgentId) {
        showToast('请先选择或创建一个智能体');
        return;
    }
    agentKbUploadMode = !agentKbUploadMode;
    document.getElementById('kbUploadToggle').classList.toggle('active', agentKbUploadMode);
    document.getElementById('kbUploadToggle').setAttribute('aria-pressed', agentKbUploadMode);
    document.getElementById('agentKbBar').style.display = agentKbUploadMode ? 'flex' : 'none';
}

// 每个模式独立记录当前会话ID，切换模式时恢复
let modeChatId = { agent: null, chat: null };
// Per-agent active chat tracking for conversation isolation
let agentActiveChatId = {};
// 初始化所有允许智能体的活跃聊天ID
ALLOWED_AGENT_IDS.forEach(id => { agentActiveChatId[id] = null; });

function saveAgentActiveChatIds() {
    localStorage.setItem('agentActiveChatIds', JSON.stringify(agentActiveChatId));
}

function loadAgentActiveChatIds() {
    try {
        const saved = localStorage.getItem('agentActiveChatIds');
        if (saved) agentActiveChatId = JSON.parse(saved);
    } catch(e) {}
}

// Load per-agent active chat IDs at startup
loadAgentActiveChatIds();

// ===== API Helper (with JWT Token) =====
function apiHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    if (authToken) {
        headers['Authorization'] = 'Bearer ' + authToken;
    }
    return headers;
}

// ===== Theme =====
function toggleTheme() {
    const html = document.documentElement;
    const isDark = html.getAttribute('data-theme') === 'dark';
    html.setAttribute('data-theme', isDark ? 'light' : 'dark');
    localStorage.setItem('theme', isDark ? 'light' : 'dark');
    document.getElementById('themeBtn').textContent = isDark ? '🌙' : '☀️';
}

(function initTheme() {
    const saved = localStorage.getItem('theme');
    if (saved === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
    }
})();

// ===== Web Search Toggle =====
function toggleWebSearch() {
    webSearchEnabled = !webSearchEnabled;
    const btn = document.getElementById('webSearchToggle');
    btn.classList.toggle('active', webSearchEnabled);
    localStorage.setItem('webSearch', webSearchEnabled ? '1' : '0');
}

(function initWebSearch() {
    const saved = localStorage.getItem('webSearch');
    if (saved === '1') {
        webSearchEnabled = true;
        document.getElementById('webSearchToggle').classList.add('active');
    }
})();

// ===== Mode Switch =====
function switchMode(mode) {
    if (currentMode === mode) return;

    // Before switching away from agent mode, save the current agent's active chat
    if (currentMode === 'agent' && currentAgentId) {
        agentActiveChatId[currentAgentId] = currentChatId;
        saveAgentActiveChatIds();
    }

    // 保存当前模式的 chatId
    modeChatId[currentMode] = currentChatId;

    currentMode = mode;
    localStorage.setItem('chatMode', mode);

    document.getElementById('modeChat').classList.toggle('active', mode === 'chat');
    document.getElementById('modeAgent').classList.toggle('active', mode === 'agent');

    const webToggle = document.getElementById('webSearchToggle');
    const thinkToggle = document.getElementById('deepThinkToggle');

    if (mode === 'chat') {
        webToggle.style.display = '';
        thinkToggle.classList.add('visible');
    } else {
        webToggle.style.display = '';
        thinkToggle.classList.remove('visible');
        thinkToggle.classList.remove('active');
        deepThinkEnabled = false;
    }

    const titleEl = document.getElementById('chatTitle');
    if (titleEl) {
        if (mode === 'agent' && currentAgentId) {
            const agent = myAgents.find(a => a.id === currentAgentId);
            titleEl.textContent = agent ? agent.name : '质量改进工程师助手';
        } else {
            titleEl.textContent = mode === 'agent' ? '质量改进工程师助手' : 'Chat';
        }
    }
    // Reset agent when switching to chat mode
    if (mode === 'chat') {
        currentAgentId = null;
        renderMyAgents();
    }

    // After switching to agent mode, restore from agentActiveChatId
    if (mode === 'agent' && currentAgentId) {
        const lastChat = agentActiveChatId[currentAgentId];
        if (lastChat) {
            modeChatId['agent'] = lastChat;
        }
    }

    // 更新知识库上传按钮可见性
    updateKbUploadVisibility();
    updateHeaderKbVisibility();

    // 切换模式时更新欢迎页内容
    updateWelcomeContent();

    // 切换模式时：筛选该模式的历史对话，恢复该模式上次的会话
    renderChatList();
    restoreModeChat();
}

// 恢复当前模式上次的活跃会话，如果没有则新建
async function restoreModeChat() {
    const modeChats = getModeChats();
    const savedId = modeChatId[currentMode];
    if (modeChats.length === 0) {
        // 该模式没有会话，不新建，显示空白或调研表单
        const hasSurvey = localStorage.getItem('surveyData');
        if (!hasSurvey) {
            showSurveyForm();
        } else {
            clearChatUI();
            const welcomeEl = document.getElementById('welcomeCenter');
            if (welcomeEl) welcomeEl.style.display = '';
        }
    } else if (savedId && modeChats.some(c => c.chat_id === savedId)) {
        // 恢复上次该模式的会话
        currentChatId = savedId;
        renderChatList();
        await loadChatHistory(savedId);
    } else {
        // 选择该模式的第一个会话
        currentChatId = modeChats[0].chat_id;
        modeChatId[currentMode] = currentChatId;
        renderChatList();
        await loadChatHistory(currentChatId);
    }
}

// 判断对话是否属于某个智能体（同时参考本地 chat_ids 和服务端 agent_id）
function chatBelongsToAgent(chat, agentId) {
    // 1. 检查本地 localStorage 的 chat_ids
    const agent = myAgents.find(a => a.id === agentId);
    if (agent && agent.chat_ids && agent.chat_ids.includes(chat.chat_id)) {
        return true;
    }
    // 2. 检查服务端返回的 agent_id 字段（跨浏览器同步的关键）
    if (chat.agent_id && chat.agent_id === agentId) {
        return true;
    }
    return false;
}

// 判断对话是否属于任意智能体
function chatBelongsToAnyAgent(chat) {
    return myAgents.some(agent => chatBelongsToAgent(chat, agent.id));
}

// 获取当前模式的会话列表
function getModeChats() {
    // Chat mode: show chats with mode='chat'
    if (currentMode === 'chat') {
        return allChats.filter(chat => chat.mode === 'chat');
    }
    // Agent mode with specific agent: show that agent's chats
    if (currentMode === 'agent' && currentAgentId) {
        return allChats.filter(chat => chatBelongsToAgent(chat, currentAgentId));
    }
    // Agent mode but no specific agent: show agent-mode chats not belonging to any agent
    if (currentMode === 'agent' && !currentAgentId) {
        return allChats.filter(chat => {
            const modeMatch = chat.mode === 'agent' || (!chat.mode && currentMode === 'agent');
            if (!modeMatch) return false;
            return !chatBelongsToAnyAgent(chat);
        });
    }
    return [];
}

(function initMode() {
    const saved = localStorage.getItem('chatMode');
    if (saved === 'chat') {
        currentMode = 'chat';
        localStorage.setItem('chatMode', 'chat');
        document.getElementById('modeChat').classList.add('active');
        document.getElementById('modeAgent').classList.remove('active');
    }
    // 初始化时根据状态决定知识库按钮可见性
    updateKbUploadVisibility();
    updateHeaderKbVisibility();
})();

// ===== Deep Think Toggle =====
function toggleDeepThink() {
    deepThinkEnabled = !deepThinkEnabled;
    const btn = document.getElementById('deepThinkToggle');
    btn.classList.toggle('active', deepThinkEnabled);
    localStorage.setItem('deepThink', deepThinkEnabled ? '1' : '0');
}

(function initDeepThink() {
    const saved = localStorage.getItem('deepThink');
    if (saved === '1' && currentMode === 'chat') {
        deepThinkEnabled = true;
        document.getElementById('deepThinkToggle').classList.add('active');
    }
})();

// ===== Marked Config =====
if (typeof marked !== 'undefined') {
    marked.setOptions({
        highlight: function(code, lang) {
            if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                try { return hljs.highlight(code, { language: lang }).value; } catch (e) {}
            }
            if (typeof hljs !== 'undefined') {
                try { return hljs.highlightAuto(code).value; } catch (e) {}
            }
            return code;
        },
        breaks: true,
        gfm: true,
    });

    const renderer = new marked.Renderer();
    renderer.code = function(code, language, escaped) {
        let codeText = '', lang = '';
        if (typeof code === 'object') {
            codeText = code.text || '';
            lang = code.lang || '';
        } else {
            codeText = code;
            lang = language || '';
        }
        let highlighted;
        if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
            try { highlighted = hljs.highlight(codeText, { language: lang }).value; } catch (e) { highlighted = escapeHtml(codeText); }
        } else if (typeof hljs !== 'undefined') {
            try { highlighted = hljs.highlightAuto(codeText).value; } catch (e) { highlighted = escapeHtml(codeText); }
        } else {
            highlighted = escapeHtml(codeText);
        }
        const langLabel = lang ? lang : 'code';
        const codeId = 'code-' + Math.random().toString(36).substr(2, 9);
        return `<pre><div class="code-block-header"><span>${langLabel}</span><button class="code-copy-btn" onclick="copyCodeBlock('${codeId}', this)" aria-label="复制代码">复制</button></div><code id="${codeId}" class="hljs language-${lang}">${highlighted}</code></pre>`;
    };
    marked.setOptions({ renderer: renderer });
}

// ===== Toast =====
let _toastTimer = null;
function showToast(msg, duration) {
    duration = duration || 2000;
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.classList.add('show');
    if (_toastTimer) clearTimeout(_toastTimer);
    _toastTimer = setTimeout(() => { toast.classList.remove('show'); _toastTimer = null; }, duration);
}

// ===== Clipboard =====
function copyToClipboard(text, onSuccess, onFail) {
    // 优先尝试 Clipboard API（需要 HTTPS 或 localhost）
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(() => {
            if (onSuccess) onSuccess();
        }).catch(() => {
            if (!fallbackCopy(text)) { if (onFail) onFail(); } else { if (onSuccess) onSuccess(); }
        });
        return;
    }
    // HTTP 环境：使用 fallback
    if (!fallbackCopy(text)) { if (onFail) onFail(); } else { if (onSuccess) onSuccess(); }
}

function fallbackCopy(text) {
    try {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.left = '0';
        ta.style.top = '0';
        ta.style.opacity = '0';
        ta.style.pointerEvents = 'none';
        ta.setAttribute('readonly', '');
        ta.style.fontSize = '16px'; // 防止 iOS 缩放
        document.body.appendChild(ta);
        ta.focus();
        ta.setSelectionRange(0, ta.value.length);
        const ok = document.execCommand('copy');
        document.body.removeChild(ta);
        return ok;
    } catch (e) { return false; }
}

// ===== Code Block Copy =====
function copyCodeBlock(codeId, btn) {
    const codeEl = document.getElementById(codeId);
    if (!codeEl) return;
    const text = codeEl.textContent;
    copyToClipboard(text, () => {
        btn.textContent = '已复制';
        btn.classList.add('copied');
        setTimeout(() => { btn.textContent = '复制'; btn.classList.remove('copied'); }, 2000);
        showToast('代码已复制');
    }, () => { showToast('复制失败'); });
}

// ===== Model Management =====
async function loadModels() {
    try {
        const resp = await fetch('/api/v1/models');
        const data = await resp.json();
        const select = document.getElementById('modelSelect');
        select.innerHTML = '';
        data.models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.id; opt.textContent = m.name; opt.title = m.desc;
            if (m.id === data.current) opt.selected = true;
            select.appendChild(opt);
        });
    } catch (e) { console.error('加载模型列表失败', e); }
}

async function switchModel() {
    const modelId = document.getElementById('modelSelect').value;
    try {
        const resp = await fetch('/api/v1/models/set', { method: 'POST', headers: apiHeaders(), body: JSON.stringify({ model_id: modelId }) });
        const data = await resp.json();
        if (data.success) {
            const select = document.getElementById('modelSelect');
            const name = select.options[select.selectedIndex].textContent;
            addMessageToUI('assistant', `✅ 已切换到模型: ${name}`);
        }
    } catch (e) { console.error('切换模型失败', e); }
}

// ===== Auth =====
// ===== Login Modal =====
function openLoginModal() {
    document.getElementById('loginModalTitle').textContent = '用户登录';
    document.getElementById('loginModalSubtitle').textContent = 'USERS LOGIN';
    document.getElementById('loginModal').classList.add('show');
    setTimeout(() => document.getElementById('loginUser').focus(), 100);
}

function closeLoginModal() {
    document.getElementById('loginModal').classList.remove('show');
    const loginMsg = document.getElementById('loginMsg');
    if (loginMsg) { loginMsg.textContent = ''; loginMsg.className = 'msg-box'; }
    const regMsg = document.getElementById('regMsg');
    if (regMsg) { regMsg.textContent = ''; regMsg.className = 'msg-box'; }
}

function openTrialModal() {
    document.getElementById('loginModalTitle').textContent = '用户登录';
    document.getElementById('loginModalSubtitle').textContent = 'USERS LOGIN';
    document.getElementById('loginModal').classList.add('show');
    setTimeout(() => document.getElementById('loginUser').focus(), 100);
}

function switchTab(tab) {
    // Tab bar removed from login page, this function is kept for backward compat
    if (document.getElementById('loginForm')) {
        document.getElementById('loginForm').style.display = 'block';
    }
}

// 登录页作为首页：禁止点击背景关闭（已移除关闭按钮）
// 原逻辑：点击overlay背景会关闭登录弹窗，但现在登录页就是首页，不应被关闭
document.addEventListener('click', function(e) {
    // 不再允许通过点击背景关闭登录弹窗
});

// Close modals on Escape key — close the topmost active modal only
document.addEventListener('keydown', function(e) {
    if (e.key !== 'Escape') return;
    // Priority: rename > docs > login (topmost first)
    const renameOverlay = document.getElementById('renameOverlay');
    if (renameOverlay && renameOverlay.classList.contains('show')) { cancelRename(); return; }
    const docsModal = document.getElementById('docsModal');
    if (docsModal && docsModal.classList.contains('show')) { closeDocs(); return; }
    const loginModal = document.getElementById('loginModal');
    // 登录页作为首页，Escape键不关闭登录弹窗
    if (loginModal && loginModal.classList.contains('show') && currentUser) { closeLoginModal(); return; }
});

async function doLogin() {
    const username = document.getElementById('loginUser').value.trim();
    const password = document.getElementById('loginPass').value.trim();
    const msgEl = document.getElementById('loginMsg');
    if (!username || !password) { msgEl.className = 'msg-box error'; msgEl.textContent = '请输入用户名和密码'; return; }
    try {
        const resp = await fetch('/api/v1/auth/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username, password }) });
        const data = await resp.json();
        if (data.success) {
            currentUser = username;
            userRole = data.role || 'user';
            if (data.token) { authToken = data.token; localStorage.setItem('authToken', data.token); }
            localStorage.setItem('userRole', userRole);
            msgEl.className = 'msg-box success'; msgEl.textContent = '登录成功！';
            setTimeout(async () => {
                document.getElementById('loginModal').classList.remove('show');
                document.getElementById('chatPage').style.display = 'flex';
                document.body.classList.add('body-chat-mode');
                // [BUG FIX] Push history state so browser back button returns to login
                history.pushState({page: 'chat'}, '');
                document.getElementById('headerUserName').textContent = username;
                document.getElementById('headerUserAvatar').textContent = username[0].toUpperCase();
                // 显示管理员标识
                if (userRole === 'admin') {
                    document.getElementById('headerUserName').textContent = username + ' (管理员)';
                }
                // 清空旧的对话缓存（避免 JLAGENT 旧数据残留）
                allChats = [];
                currentChatId = null;
                modeChatId = { agent: null, chat: null };
                Object.keys(agentActiveChatId).forEach(k => delete agentActiveChatId[k]);
                // 清空侧边栏对话列表（避免闪现旧数据）
                const chatListEl = document.getElementById('chatList');
                if (chatListEl) chatListEl.innerHTML = '';
                loadModels();
                await syncAgentsFromServer(true);
                renderMyAgents();
                updateKbUploadVisibility();
                updateHeaderKbVisibility();
                if (!currentAgentId && myAgents.length > 0) {
                    currentAgentId = myAgents[0].id;
                    currentMode = 'agent';
                    renderMyAgents();
                    updateGenButtonsVisibility();
                    updateHeaderKbVisibility();
                }
                // 先加载对话列表（await 确保完成后再判断）
                await loadChatList();
                // 再检查是否有历史对话
                const modeChats = getModeChats();
                if (modeChats && modeChats.length > 0) {
                    currentChatId = modeChats[0].chat_id;
                    modeChatId['agent'] = currentChatId;
                    renderChatList();
                    await loadChatHistory(currentChatId);
                } else {
                    // 没有历史对话 → 显示填表页
                    setTimeout(() => showSurveyForm(), 100);
                }
            }, 500);
        } else { msgEl.className = 'msg-box error'; msgEl.textContent = data.message || '登录失败'; }
    } catch (e) { msgEl.className = 'msg-box error'; msgEl.textContent = '网络错误'; }
}

async function doRegister() {
    // 注册功能已禁用，新用户只能由管理员在后端创建
    alert('注册功能已禁用，请联系管理员创建账号');
}

function doLogout() {
    currentUser = null; userRole = null; authToken = null; selectedFile = null; currentChatId = null; allChats = []; currentAgentId = null; agentKbUploadMode = false;
    localStorage.removeItem('authToken');
    localStorage.removeItem('userRole');
    // Hide KB page if open
    const kbPage = document.getElementById('kbPage');
    if (kbPage) kbPage.style.display = 'none';
    document.getElementById('chatPage').style.display = 'none';
    // 登出后直接显示登录页
    document.getElementById('loginModal').classList.add('show');
    document.body.classList.remove('body-chat-mode');
    document.getElementById('chatMessages').innerHTML = '';
    document.getElementById('loginUser').value = '';
    document.getElementById('loginPass').value = '';
    // 清除header用户信息
    const headerUserName = document.getElementById('headerUserName');
    const headerUserAvatar = document.getElementById('headerUserAvatar');
    if (headerUserName) headerUserName.textContent = '';
    if (headerUserAvatar) headerUserAvatar.textContent = '';
    // [BUG FIX] 清除登录消息，避免登出后仍显示"登录成功"
    const logoutMsg = document.getElementById('loginMsg');
    if (logoutMsg) { logoutMsg.textContent = ''; logoutMsg.className = 'msg-box'; }
    updateHeaderKbVisibility();
    // [BUG FIX] Update history state so back button is consistent
    if (history.state && (history.state.page === 'chat' || history.state.page === 'kb')) {
        history.replaceState({page: 'login'}, '');
    }
}

// [BUG FIX] Handle browser back/forward navigation
// When user presses back from chat, return to login page (with logout).
// When user presses forward from login while authenticated, return to chat.
window.addEventListener('popstate', function(e) {
    const loginModal = document.getElementById('loginModal');
    const chatPage = document.getElementById('chatPage');
    const kbPage = document.getElementById('kbPage');
    const helpPage = document.getElementById('helpPage');
    const externalKbPage = document.getElementById('externalKbPage');
    const chatContent = document.getElementById('chatContent');
    const sidebar = document.getElementById('sidebar');

    // 处理调研表单/帮助/外部知识库页面的后退
    if (e.state && (e.state.page === 'survey' || e.state.page === 'help' || e.state.page === 'external_kb')) {
        // 前进到帮助/外部知识库页面（用户按了前进按钮）
        if (currentUser && authToken) {
            loginModal.classList.remove('show');
            chatPage.style.display = 'flex';
            document.body.classList.add('body-chat-mode');
            if (chatContent) chatContent.style.display = 'none';
            if (kbPage) kbPage.style.display = 'none';
            if (e.state.page === 'help') {
                if (helpPage) helpPage.style.display = '';
                if (externalKbPage) externalKbPage.style.display = 'none';
                const surveyPage = document.getElementById('surveyPage');
                if (surveyPage) surveyPage.style.display = 'none';
            } else if (e.state.page === 'external_kb') {
                if (externalKbPage) externalKbPage.style.display = '';
                if (helpPage) helpPage.style.display = 'none';
                const surveyPage = document.getElementById('surveyPage');
                if (surveyPage) surveyPage.style.display = 'none';
            } else if (e.state.page === 'survey') {
                const surveyPage = document.getElementById('surveyPage');
                if (surveyPage) surveyPage.style.display = 'block';
                if (helpPage) helpPage.style.display = 'none';
                if (externalKbPage) externalKbPage.style.display = 'none';
                if (kbPage) kbPage.style.display = 'none';
                if (chatContent) chatContent.style.display = 'none';
                if (sidebar) sidebar.style.display = '';
            }
        } else {
            history.replaceState({page: 'login'}, '');
        }
        return;
    }

    if (e.state && e.state.page === 'chat') {
        // 回到聊天页 - 从知识库页返回 或 从登录页前进
        if (currentUser && authToken) {
            // [BUG FIX] 检测是否是 hideKbPage 主动触发的后退（从 kb 返回 chat）
            // 这种情况下绝不连续后退，直接显示 chat 页即可
            const fromKbNavigation = window._navigatingFromKb === true;
            if (fromKbNavigation) {
                // 清除标志位
                window._navigatingFromKb = false;
            } else {
                // 不是 hideKbPage 触发的，是用户主动点浏览器后退按钮
                // 此时 kbPage 本来就是关的，又后退到 chat 条目
                // 说明 history 栈有修复前的存量堆积（[login, chat, chat, ...]）
                // 但为了安全，不自动连续后退（可能误伤其他场景）
                // 只做正常的 UI 切换，让用户多按几次后退到达 login
                // 这样保证不会错误地退出登录
            }
            loginModal.classList.remove('show');
            chatPage.style.display = 'flex';
            document.body.classList.add('body-chat-mode');
            // [BUG FIX] 如果从知识库/帮助/外部知识库返回，关闭对应页面，恢复聊天页
            if (kbPage) kbPage.style.display = 'none';
            if (helpPage) helpPage.style.display = 'none';
            if (externalKbPage) externalKbPage.style.display = 'none';
            const surveyPage = document.getElementById('surveyPage');
            if (surveyPage) surveyPage.style.display = 'none';
            if (chatContent) chatContent.style.display = 'flex';
            if (sidebar) sidebar.style.display = '';
        } else {
            // Not authenticated anymore, go back to login
            history.replaceState({page: 'login'}, '');
        }
    } else if (e.state && e.state.page === 'kb') {
        // 前进到知识库页（用户按了前进按钮）
        if (currentUser && authToken && currentAgentId) {
            loginModal.classList.remove('show');
            chatPage.style.display = 'flex';
            document.body.classList.add('body-chat-mode');
            if (chatContent) chatContent.style.display = 'none';
            if (kbPage) kbPage.style.display = 'flex';
            const surveyPageEl = document.getElementById('surveyPage');
            if (surveyPageEl) surveyPageEl.style.display = 'none';
            if (helpPage) helpPage.style.display = 'none';
            if (externalKbPage) externalKbPage.style.display = 'none';
            // [BUG FIX] 知识库页隐藏侧边栏
            if (sidebar) sidebar.style.display = 'none';
            const sidebarOverlay = document.getElementById('sidebarOverlay');
            if (sidebarOverlay) sidebarOverlay.style.display = 'none';
        } else {
            history.replaceState({page: 'login'}, '');
        }
    } else {
        // Back to login - perform logout to ensure clean state
        if (currentUser) {
            // Clear session but don't push another history entry
            currentUser = null; userRole = null; authToken = null; selectedFile = null; currentChatId = null; allChats = []; currentAgentId = null; agentKbUploadMode = false;
            localStorage.removeItem('authToken');
            localStorage.removeItem('userRole');
            if (kbPage) kbPage.style.display = 'none';
            chatPage.style.display = 'none';
            loginModal.classList.add('show');
            document.body.classList.remove('body-chat-mode');
            document.getElementById('chatMessages').innerHTML = '';
            document.getElementById('loginUser').value = '';
            document.getElementById('loginPass').value = '';
            // [BUG FIX] 清除登录消息，避免回退到登录页后仍显示"登录成功"
            const loginMsg = document.getElementById('loginMsg');
            if (loginMsg) { loginMsg.textContent = ''; loginMsg.className = 'msg-box'; }
            updateHeaderKbVisibility();
        }
    }
});

// ===== Auto-login with JWT token =====
async function tryAutoLogin() {
    const token = localStorage.getItem('authToken');
    if (!token) return false;
    try {
        const resp = await fetch('/api/v1/auth/me', { headers: { 'Authorization': 'Bearer ' + token } });
        const data = await resp.json();
        if (data.valid && data.username) {
            currentUser = data.username;
            authToken = token;
            // 自动登录成功：隐藏登录页，显示聊天页
            document.getElementById('loginModal').classList.remove('show');
            document.getElementById('chatPage').style.display = 'flex';
            document.body.classList.add('body-chat-mode');
            // [BUG FIX] Push history state so browser back button returns to login
            history.pushState({page: 'chat'}, '');
            document.getElementById('headerUserName').textContent = data.username;
            document.getElementById('headerUserAvatar').textContent = data.username[0].toUpperCase();
            loadChatList();
            loadModels();
            await syncAgentsFromServer(true);  // [#12] 自动登录时强制同步
            renderMyAgents();
            updateKbUploadVisibility();
            updateHeaderKbVisibility();
            // [#14] 默认选中第一个智能体，避免进入空白的agent模式
            if (!currentAgentId && myAgents.length > 0) {
                await switchToAgent(myAgents[0].id);
            }
            return true;
        }
    } catch (e) { console.warn('自动登录失败', e); }
    localStorage.removeItem('authToken');
    // 自动登录失败：确保登录页可见
    document.getElementById('loginModal').classList.add('show');
    return false;
}

// ===== Centered Mode =====
function updateCenteredMode() {
    const content = document.getElementById('chatContent');
    const messages = document.getElementById('chatMessages');
    const hasMessages = messages.children.length > 0;
    content.classList.toggle('centered', !hasMessages);
    // 更新欢迎页内容（根据当前智能体动态显示）
    updateWelcomeContent();
}

// 根据当前智能体更新欢迎页内容
function updateWelcomeContent() {
    const welcomeEl = document.getElementById('welcomeCenter');
    if (!welcomeEl) return;

    const config = currentAgentId ? getAgentWelcomeConfig(currentAgentId) : null;

    if (config) {
        // 将20个按钮分配到5行，每行至少2个，贪心平衡行宽
        const questions = config.questions;
        const NUM_ROWS = 5;
        let rowsHtml = '';

        if (questions.length === 20) {
            // 估算按钮宽度（与CSS font-size=13px对应）
            const charWidth = (ch) => {
                const code = ch.charCodeAt(0);
                if (code >= 0x4e00 && code <= 0x9fff) return 13; // 中文
                if ((code >= 65 && code <= 90) || (code >= 97 && code <= 122) || (code >= 48 && code <= 57)) return 7.8; // 英文/数字
                return 6.5; // 其他
            };
            const estimateWidth = (label) => {
                let w = 0;
                for (const ch of label) w += charWidth(ch);
                return w + 30; // padding 14*2 + border 1*2
            };

            const CONTAINER = 660;
            const GAP = 8;

            // 按宽度降序排列
            const indexed = questions.map((q, i) => ({
                idx: i,
                w: estimateWidth(typeof q === 'object' && q.label ? q.label : String(q))
            }));
            indexed.sort((a, b) => b.w - a.w);

            // 动态选择行模式：检查中间5个按钮能否放进一行
            const middleFiveWidth = indexed.slice(10, 15).reduce((s, x) => s + x.w, 0) + 4 * GAP;

            // 不同智能体使用不同行模式，避免千篇一律
            // 按钮窄的智能体可用双5行模式，按钮宽的用单5行模式
            const agentIndex = ALLOWED_AGENT_IDS.indexOf(currentAgentId);
            const PATTERNS_TWO5 = [
                [3,5,4,5,3], [4,5,3,5,3], [3,5,3,5,4], [5,3,4,5,3], [3,5,5,4,3],
                [5,3,5,4,3], [3,5,4,3,5], [5,4,3,5,3], [3,4,5,3,5], [4,3,5,3,5],
            ];
            const PATTERNS_ONE5 = [
                [3,4,5,4,4], [4,3,5,4,4], [4,4,5,4,3], [4,4,5,3,4], [3,4,4,5,4],
                [4,3,4,5,4], [4,4,3,5,4], [5,4,4,4,3], [3,5,4,4,4], [4,4,4,3,5],
            ];

            const pattern = middleFiveWidth <= CONTAINER
                ? PATTERNS_TWO5[agentIndex >= 0 ? agentIndex % PATTERNS_TWO5.length : 0]
                : PATTERNS_ONE5[agentIndex >= 0 ? agentIndex % PATTERNS_ONE5.length : 0];

            // 按行目标数量分组：宽按钮→少行，窄按钮→多行
            const countToRows = {};
            for (let r = 0; r < NUM_ROWS; r++) {
                const c = pattern[r];
                if (!countToRows[c]) countToRows[c] = [];
                countToRows[c].push(r);
            }

            const rowItems = Array.from({length: NUM_ROWS}, () => []);
            let btnPtr = 0;
            for (const count of Object.keys(countToRows).map(Number).sort((a, b) => a - b)) {
                for (const rowIdx of countToRows[count]) {
                    for (let i = 0; i < count; i++) {
                        rowItems[rowIdx].push(indexed[btnPtr].idx);
                        btnPtr++;
                    }
                }
            }

            // 每行内部交替排列（最宽-最窄-次宽-次窄），增加视觉变化
            for (let r = 0; r < NUM_ROWS; r++) {
                const items = rowItems[r].map(idx => ({
                    idx,
                    w: indexed.find(x => x.idx === idx).w
                }));
                items.sort((a, b) => b.w - a.w);
                const reordered = [];
                let left = 0, right = items.length - 1;
                while (left <= right) {
                    reordered.push(items[left]);
                    if (left !== right) reordered.push(items[right]);
                    left++; right--;
                }
                rowItems[r] = reordered.map(x => x.idx);
            }

            // 生成5行HTML
            for (let r = 0; r < NUM_ROWS; r++) {
                const rowBtns = rowItems[r].map(idx => {
                    const q = questions[idx];
                    if (typeof q === 'object' && q.label) {
                        return `<span class="quick-action" onclick="fillQuick(this)" data-question="${escapeHtml(q.question)}" role="button" tabindex="0">${escapeHtml(q.label)}</span>`;
                    }
                    return `<span class="quick-action" onclick="fillQuick(this)" data-question="${escapeHtml(q)}" role="button" tabindex="0">${escapeHtml(q)}</span>`;
                }).join('');
                rowsHtml += `<div class="kw-row">${rowBtns}</div>`;
            }
        } else {
            // 非20个按钮时，使用原来的flex-wrap布局
            rowsHtml = `<div class="quick-actions${questions.length >= 8 ? ' many-questions' : ''}">` +
                questions.map(q => {
                    if (typeof q === 'object' && q.label) {
                        return `<span class="quick-action" onclick="fillQuick(this)" data-question="${escapeHtml(q.question)}" role="button" tabindex="0">${escapeHtml(q.label)}</span>`;
                    }
                    return `<span class="quick-action" onclick="fillQuick(this)" data-question="${escapeHtml(q)}" role="button" tabindex="0">${escapeHtml(q)}</span>`;
                }).join('') +
                `</div>`;
        }

        // 智能体专属欢迎页
        const questionsHtml = (config.questions && config.questions.length > 0)
            ? `<div class="quick-actions many-questions kw-five-rows">${rowsHtml}</div>`
            : '';
        welcomeEl.innerHTML = `
            <h2 class="welcome-agent-name">${escapeHtml(config.name)}</h2>
            <p class="welcome-agent-desc">${escapeHtml(config.desc)}</p>
            ${questionsHtml}
        `;
    } else {
        // 默认欢迎页
        welcomeEl.innerHTML = `
            <h2>质量改进工程师助手</h2>
            <p>专业模具AI智能体，独立赋能研发与质量管理</p>
            <div class="quick-actions">
                <span class="quick-action" onclick="fillQuick(this)" data-question="模具设计评审有哪些关键节点？" role="button" tabindex="0">设计评审</span>
                <span class="quick-action" onclick="fillQuick(this)" data-question="VDA6.4过程审核要点是什么？" role="button" tabindex="0">过程审核</span>
                <span class="quick-action" onclick="fillQuick(this)" data-question="帮我分析DFMEA风险" role="button" tabindex="0">DFMEA分析</span>
                <span class="quick-action" onclick="fillQuick(this)" data-question="不合格品纠正措施怎么制定？" role="button" tabindex="0">CAPA建议</span>
            </div>
        `;
    }
}

// 点击快捷问题：填入输入框（不自动发送），用户可编辑后发送
function fillQuick(el) {
    const text = el.getAttribute('data-question') || el.textContent;
    const input = document.getElementById('msgInput');
    if (input) {
        input.value = text;
        autoResize(input);
        input.focus();
    }
}

// ===== Chat List =====
async function loadChatList() {
    if (!currentUser) return;
    try {
        const resp = await fetch(`/api/v1/chats?username=${encodeURIComponent(currentUser)}`, { headers: apiHeaders() });
        const data = await resp.json();
        if (data.success) {
            allChats = data.chats;
            renderChatList();
            // 按当前模式恢复会话
            const modeChats = getModeChats();
            // 如果当前聊天仍然存在于全部聊天列表中，不要强制跳走
            // （避免智能体对话回复完成后，因过滤不同步导致跳转到空页面）
            const currentChatStillExists = currentChatId && allChats.some(c => c.chat_id === currentChatId);
            if (modeChats.length === 0 && !currentChatStillExists) {
                // 没有会话，不自动创建，根据调研状态显示
                const hasSurvey = localStorage.getItem('surveyData');
                if (!hasSurvey) {
                    showSurveyForm();
                } else {
                    clearChatUI();
                    const welcomeEl = document.getElementById('welcomeCenter');
                    if (welcomeEl) welcomeEl.style.display = '';
                }
            } else if (!currentChatId || (!currentChatStillExists && !modeChats.some(c => c.chat_id === currentChatId))) {
                currentChatId = modeChats[0].chat_id;
                modeChatId[currentMode] = currentChatId;
                renderChatList();
                await loadChatHistory(currentChatId);
            }
        }
    } catch (e) { console.error('加载会话列表失败', e); }
}

function renderChatList() {
    const list = document.getElementById('chatList');
    list.innerHTML = '';
    // 只显示当前模式的会话
    const modeChats = getModeChats();
    // 控制底部提示文字的显示
    const footerHint = document.getElementById('sidebarFooterHint');
    if (footerHint) {
        if (currentAgentId && modeChats.length === 0) {
            footerHint.textContent = '暂无历史对话';
            footerHint.style.display = '';
        } else if (!currentAgentId) {
            footerHint.textContent = '选择智能体查看历史对话';
            footerHint.style.display = '';
        } else {
            footerHint.style.display = 'none';
        }
    }
    modeChats.forEach(chat => {
        const item = document.createElement('div');
        item.className = `chat-item${chat.chat_id === currentChatId ? ' active' : ''}`;
        item.onclick = (e) => {
            if (e.target.closest('.chat-action-btn')) return;
            switchChat(chat.chat_id);
            closeSidebarOnMobile();
        };
        const safeTitle = escapeHtml(chat.title || '新对话');
        const safeTitleJs = (chat.title || '新对话').replace(/\\/g, '\\\\').replace(/'/g, "\\'");
        const timeStr = formatTime(chat.updated_at || chat.created_at);
        item.innerHTML = `
            <span class="chat-icon">💬</span>
            <span class="chat-title" title="${safeTitle}">${safeTitle}</span>
            <span class="chat-time">${timeStr}</span>
            <div class="chat-actions">
                <button class="chat-action-btn" onclick="openRename('${chat.chat_id}', '${safeTitleJs}')" title="重命名" aria-label="重命名对话">✏️</button>
                <button class="chat-action-btn delete" onclick="deleteChatItem('${chat.chat_id}')" title="删除" aria-label="删除对话">🗑️</button>
            </div>
        `;
        list.appendChild(item);
    });
}

async function createNewChat() {
    if (!currentUser) return;
    try {
        const chatTitle = currentAgentId ? (myAgents.find(a => a.id === currentAgentId)?.name || '新对话') : '新对话';
        const resp = await fetch(`/api/v1/chats?username=${encodeURIComponent(currentUser)}&title=${encodeURIComponent(chatTitle)}&mode=${currentMode}&agent_id=${currentAgentId || ''}`, { method: 'POST', headers: apiHeaders() });
        const data = await resp.json();
        if (data.success) {
            currentChatId = data.chat.chat_id;
            modeChatId[currentMode] = currentChatId;
            // Associate chat with current agent
            if (currentAgentId) {
                const agent = myAgents.find(a => a.id === currentAgentId);
                if (agent) {
                    if (!agent.chat_ids) agent.chat_ids = [];
                    if (!agent.chat_ids.includes(data.chat.chat_id)) agent.chat_ids.push(data.chat.chat_id);
                    agentActiveChatId[currentAgentId] = data.chat.chat_id;
                    saveAgentActiveChatIds();
                    saveAgents();
                }
            }
            await loadChatList();
            clearChatUI();
            closeSidebarOnMobile();
        }
    } catch (e) { console.error('创建会话失败', e); }
}

async function switchChat(chatId) {
    if (chatId === currentChatId) return;
    // [BUG FIX #2] 切换聊天时中断正在进行的流式响应
    // 防止旧SSE流在后台继续运行导致 isLoading 锁死、新聊天无法发送消息
    stopGeneration();
    currentChatId = chatId;
    modeChatId[currentMode] = chatId;

    // Determine which agent owns this chat (check both local chat_ids and server agent_id)
    let belongsToAgent = null;
    const chatData = allChats.find(c => c.chat_id === chatId);
    myAgents.forEach(agent => {
        if (chatBelongsToAgent(chatData || { chat_id: chatId }, agent.id)) {
            belongsToAgent = agent.id;
        }
    });
    if (belongsToAgent) {
        currentAgentId = belongsToAgent;
        agentActiveChatId[currentAgentId] = chatId;
        saveAgentActiveChatIds();
    }

    renderChatList();
    updateHeaderKbVisibility();
    await loadChatHistory(chatId);
}

async function loadChatHistory(chatId) {
    const container = document.getElementById('chatMessages');
    container.innerHTML = '';
    try {
        const resp = await fetch(`/api/v1/history/${chatId}`, { headers: apiHeaders() });
        const data = await resp.json();
        const messages = data.messages || [];
        if (messages.length > 0) {
            // [性能修复] 限制加载的消息数量，避免DOM过多导致页面卡顿
            const MAX_RENDER_MESSAGES = 50;
            let messagesToRender = messages;
            let hasOlderMessages = false;
            if (messages.length > MAX_RENDER_MESSAGES) {
                hasOlderMessages = true;
                messagesToRender = messages.slice(-MAX_RENDER_MESSAGES);
            }
            if (hasOlderMessages) {
                const hint = document.createElement('div');
                hint.className = 'message system';
                hint.innerHTML = '<div class="bubble" style="text-align:center;color:var(--text-secondary);font-size:13px;">已省略较早的 ' + (messages.length - MAX_RENDER_MESSAGES) + ' 条消息（完整记录已保存）</div>';
                container.appendChild(hint);
            }
            messagesToRender.forEach(m => addMessageToUI(m.role, m.content));
            scrollToBottom();
        }
        updateCenteredMode();
    } catch (e) { console.error('加载历史失败', e); }
}

async function deleteChatItem(chatId) {
    if (!confirm('确定删除这个对话？')) return;
    try {
        await fetch(`/api/v1/chats/${chatId}?username=${encodeURIComponent(currentUser)}`, { method: 'DELETE', headers: apiHeaders() });

        // Remove chat_id from all agents
        myAgents.forEach(agent => {
            if (agent.chat_ids) {
                agent.chat_ids = agent.chat_ids.filter(id => id !== chatId);
            }
            // Also clean agentActiveChatId
            if (agentActiveChatId[agent.id] === chatId) {
                agentActiveChatId[agent.id] = agent.chat_ids && agent.chat_ids.length > 0 ? agent.chat_ids[0] : null;
            }
        });
        saveAgentActiveChatIds();
        saveAgents();

        if (chatId === currentChatId) {
            currentChatId = null;
            modeChatId[currentMode] = null;
            clearChatUI();
        }
        await loadChatList();
        // 如果当前模式没有会话了，不自动创建新对话
        const modeChats = getModeChats();
        if (modeChats.length === 0) {
            // 没有会话了 → 直接显示体系调研填写页
            showSurveyForm();
        }
    } catch (e) { console.error('删除会话失败', e); }
}

function openRename(chatId, currentTitle) {
    renamingChatId = chatId;
    document.getElementById('renameInput').value = currentTitle;
    document.getElementById('renameOverlay').classList.add('show');
    setTimeout(() => document.getElementById('renameInput').focus(), 100);
}

function closeRename() {
    document.getElementById('renameOverlay').classList.remove('show');
    renamingChatId = null;
}

async function confirmRename() {
    const newTitle = document.getElementById('renameInput').value.trim();
    if (!newTitle || !renamingChatId) return;
    const username = currentUser || '';
    try {
        await fetch(`/api/v1/chats/${renamingChatId}/rename`, {
            method: 'PUT',
            headers: apiHeaders(),
            body: JSON.stringify({ username, chat_id: renamingChatId, new_title: newTitle })
        });
        document.getElementById('renameOverlay').classList.remove('show');
        await loadChatList();
    } catch (e) { showToast('重命名失败'); }
    renamingChatId = null;
}

function cancelRename() {
    document.getElementById('renameOverlay').classList.remove('show');
    renamingChatId = null;
}

function clearChatUI() {
    document.getElementById('chatMessages').innerHTML = '';
    updateCenteredMode();
}

async function clearCurrentChat() {
    if (!currentChatId) return;
    if (!confirm('确定清除当前对话的所有消息？')) return;
    try {
        await fetch(`/api/v1/history/${currentChatId}`, { method: 'DELETE', headers: apiHeaders() });
        clearChatUI();
    } catch (e) {}
}

// ===== Sidebar =====
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebarOverlay');
    if (window.innerWidth <= 768) {
        sidebar.classList.toggle('mobile-open');
        overlay.classList.toggle('active');
    } else {
        sidebar.classList.toggle('collapsed');
    }
}
function closeSidebarMobile() {
    document.getElementById('sidebar').classList.remove('mobile-open');
    document.getElementById('sidebarOverlay').classList.remove('active');
}
function closeSidebarOnMobile() {
    if (window.innerWidth <= 768) setTimeout(closeSidebarMobile, 200);
}

// ===== Scroll =====
function setupScrollDetection() {
    const el = document.getElementById('chatMessages');
    el.addEventListener('scroll', () => {
        const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
        userScrolledUp = distFromBottom > 100;
        const btn = document.getElementById('scrollBottomBtn');
        btn.classList.toggle('show', userScrolledUp);
    });
}

function scrollToBottom() {
    const el = document.getElementById('chatMessages');
    setTimeout(() => {
        el.scrollTop = el.scrollHeight;
        userScrolledUp = false;
        document.getElementById('scrollBottomBtn').classList.remove('show');
    }, 50);
}

function smartScrollToBottom() {
    if (!userScrolledUp) scrollToBottom();
}

// ===== Stop Generation =====
function stopGeneration() {
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }
    isLoading = false;
    document.getElementById('sendBtn').style.display = '';
    document.getElementById('stopBtn').style.display = 'none';
    document.getElementById('sendBtn').disabled = false;
}

// ===== Thinking Status Texts =====
const THINKING_TEXTS = [
    '正在思考...',
    '分析问题中...',
    '整理思路...',
    '查找信息中...',
    '生成回答中...',
];
let thinkingTextIndex = 0;
let thinkingInterval = null;

// ===== Streaming Chat =====
function createStreamingBubble() {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'message assistant';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    const actions = document.createElement('div');
    actions.className = 'message-actions';
    actions.innerHTML = `
        <button class="msg-action-btn" title="复制" onclick="copyMessage(this)" aria-label="复制消息">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
        </button>
        <button class="msg-action-btn" title="重新生成" onclick="regenerateMessage(this)" aria-label="重新生成">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 4v6h6"/><path d="M3.51 15a9 9 0 102.13-9.36L1 10"/></svg>
        </button>
    `;
    div.appendChild(bubble);
    div.appendChild(actions);
    container.appendChild(div);
    return bubble;
}

// 统一重置流式 UI 状态，防止按钮灰色/工具标签转圈等残留
function resetStreamingUI() {
    const sendBtn = document.getElementById('sendBtn');
    const stopBtn = document.getElementById('stopBtn');
    if (sendBtn) {
        sendBtn.disabled = false;
        sendBtn.style.display = '';
    }
    if (stopBtn) {
        stopBtn.style.display = 'none';
    }
    isLoading = false;
    currentAbortController = null;
    // [性能修复] 每次对话结束后清理过多的DOM节点，防止长时间运行后页面变慢
    cleanupExcessMessages();
}

function cleanupExcessMessages() {
    // 限制聊天区域DOM节点数量，超过100条消息时移除最早的
    const container = document.getElementById('chatMessages');
    if (!container) return;
    const MAX_DOM_MESSAGES = 100;
    const messages = container.querySelectorAll('.message');
    if (messages.length > MAX_DOM_MESSAGES) {
        const toRemove = messages.length - MAX_DOM_MESSAGES;
        for (let i = 0; i < toRemove; i++) {
            messages[i].remove();
        }
        // 如果没有省略提示，加一个
        const existingHint = container.querySelector('.system .bubble');
        if (!existingHint || !existingHint.textContent.includes('省略')) {
            const hint = document.createElement('div');
            hint.className = 'message system';
            hint.innerHTML = '<div class="bubble" style="text-align:center;color:var(--text-secondary);font-size:13px;">已省略较早的消息（完整记录已保存）</div>';
            container.insertBefore(hint, container.firstChild);
        }
    }
}

    // [性能修复] 前端内存清理：页面长时间打开后定期清理
function cleanupFrontendMemory() {
    // 1. 清理过多的DOM消息节点
    cleanupExcessMessages();
    
    // 2. 清理已完成的 AbortController 引用
    if (currentAbortController && currentAbortController.signal.aborted) {
        currentAbortController = null;
    }
    
    // 3. 清理 thinkingInterval（如果残留）
    if (thinkingInterval && !isLoading) {
        clearInterval(thinkingInterval);
        thinkingInterval = null;
    }
    
    // 4. 清理 Blob URL（浏览器不会自动释放）
    try {
        document.querySelectorAll('a[href^="blob:"]').forEach(a => {
            // 只清理已下载过的（有download属性的）
            if (a.download) {
                try { URL.revokeObjectURL(a.href); } catch(e) {}
            }
        });
    } catch(e) {}
}

// [性能修复] 每5分钟自动执行一次前端内存清理，防止长时间打开页面变慢
setInterval(cleanupFrontendMemory, 5 * 60 * 1000);

async function streamChat(url, options, bubble) {
    let fullText = '';
    let cursorEl = null;
    let thinkingEl = null;

    currentAbortController = new AbortController();
    if (options && !options.signal) {
        options.signal = currentAbortController.signal;
    }

    // Show stop button
    document.getElementById('sendBtn').style.display = 'none';
    document.getElementById('stopBtn').style.display = '';

    function addThinking() {
        if (thinkingEl) return;
        thinkingEl = document.createElement('div');
        thinkingEl.className = 'thinking-indicator';
        thinkingTextIndex = 0;
        thinkingEl.innerHTML = `<div class="spinner"></div><span class="think-status">${THINKING_TEXTS[0]}</span>`;
        bubble.appendChild(thinkingEl);
        smartScrollToBottom();
        // Rotate thinking text
        thinkingInterval = setInterval(() => {
            thinkingTextIndex = (thinkingTextIndex + 1) % THINKING_TEXTS.length;
            const statusEl = thinkingEl?.querySelector('.think-status');
            if (statusEl) statusEl.textContent = THINKING_TEXTS[thinkingTextIndex];
        }, 2000);
    }

    function removeThinking() {
        if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }
        if (thinkingInterval) { clearInterval(thinkingInterval); thinkingInterval = null; }
    }

    function addToolTag(display, isDone) {
        removeThinking();
        // [BUG FIX] 当 isDone=true 时，找到已有的 running 标签并更新状态，
        // 而不是创建新标签。原代码总是创建新标签，导致工具完成时出现重复：
        // "搜索文档(spinner) ✓ 搜索文档" 而不是 "✓ 搜索文档"
        if (isDone) {
            // 查找已有的 running 状态的同名工具标签
            const runningTags = bubble.querySelectorAll('.tool-tag.running');
            for (const existingTag of runningTags) {
                // 提取标签中的工具名称文本（去除 spinner/icon 部分）
                const tagText = existingTag.textContent.trim();
                if (tagText === display || tagText.includes(display)) {
                    // 找到匹配的 running 标签，更新为 done 状态
                    existingTag.className = 'tool-tag done';
                    existingTag.innerHTML = `<span class="tool-icon">✓</span> ${escapeHtml(display)}`;
                    smartScrollToBottom();
                    return;  // 更新完成，不创建新标签
                }
            }
            // 如果没找到匹配的 running 标签（异常情况），仍创建新标签
            const tag = document.createElement('span');
            tag.className = 'tool-tag done';
            tag.innerHTML = `<span class="tool-icon">✓</span> ${escapeHtml(display)}`;
            bubble.appendChild(tag);
            bubble.appendChild(document.createTextNode(' '));
        } else {
            // isDone=false：创建新的 running 标签
            const tag = document.createElement('span');
            tag.className = 'tool-tag running';
            tag.innerHTML = `<span class="tool-spinner"></span> ${escapeHtml(display)}`;
            bubble.appendChild(tag);
            bubble.appendChild(document.createTextNode(' '));
        }
        smartScrollToBottom();
    }

    function addCursor() {
        // cursor 现在由 renderStreamMarkdown 负责追加，这里只触发首次渲染
        removeThinking();
        if (!streamRenderTimer) {
            renderStreamMarkdown();
        }
    }

    // [流式 Markdown 渲染] 节流：80ms 内最多渲染一次，避免高频 re-parse 卡顿
    // 长回复（几百字以上）时纯文本追加是 O(1)，marked.parse 是 O(n)，
    // 每个 token 都 re-parse 会掉帧；节流到 ~12fps 人眼无感但流畅
    let streamRenderTimer = null;
    const STREAM_RENDER_INTERVAL = 80;  // ms

    function renderStreamMarkdown() {
        if (streamRenderTimer) return;
        streamRenderTimer = setTimeout(() => {
            streamRenderTimer = null;
            doStreamRender();
        }, STREAM_RENDER_INTERVAL);
    }

    function doStreamRender() {
        // 保存 tool-tag（renderBubbleMarkdown 会覆盖 innerHTML）
        const toolTags = Array.from(bubble.querySelectorAll('.tool-tag'));
        if (fullText) {
            try {
                if (typeof marked !== 'undefined') {
                    bubble.innerHTML = marked.parse(fullText);
                    injectDownloadButtons(bubble);
                } else {
                    bubble.innerHTML = escapeHtml(fullText).replace(/\n/g, '<br>');
                }
            } catch (e) {
                bubble.innerHTML = escapeHtml(fullText).replace(/\n/g, '<br>');
            }
        }
        // 重新插入 tool-tag 到开头
        if (toolTags.length > 0) {
            const fragment = document.createDocumentFragment();
            toolTags.forEach(tag => fragment.appendChild(tag));
            fragment.appendChild(document.createTextNode(' '));
            bubble.insertBefore(fragment, bubble.firstChild);
        }
        // 追加流式光标
        if (cursorEl) {
            cursorEl.remove();
        }
        cursorEl = document.createElement('span');
        cursorEl.className = 'stream-cursor';
        cursorEl.textContent = '▊';
        bubble.appendChild(cursorEl);
        smartScrollToBottom();
    }

    function appendToken(text) {
        removeThinking();
        fullText += text;
        // 触发节流渲染（首次也会通过 addCursor 触发，这里做兜底）
        renderStreamMarkdown();
    }

    function finalize() {
        // 清除节流定时器，立即做最终渲染（不带 cursor）
        if (streamRenderTimer) {
            clearTimeout(streamRenderTimer);
            streamRenderTimer = null;
        }
        if (cursorEl) { cursorEl.remove(); cursorEl = null; }
    }

    try {
        const resp = await fetch(url, options);

        if (!resp.ok) {
            removeThinking();
            const errData = await resp.json().catch(() => ({}));
            if (resp.status === 401) {
                showToast('登录已过期，请重新登录');
                doLogout();
                return;
            }
            bubble.innerHTML = escapeHtml(errData.detail || `请求失败 (${resp.status})`);
            return;
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const jsonStr = line.slice(6).trim();
                if (!jsonStr) continue;

                try {
                    const data = JSON.parse(jsonStr);
                    switch (data.type) {
                        case 'thinking': addThinking(); break;
                        case 'tool': addToolTag(data.display || data.name, false); break;
                        case 'tool_done': addToolTag(data.display || data.name, true); break;
                        case 'token': addCursor(); appendToken(data.content); break;
                        case 'done': finalize(); break;
                        case 'error': removeThinking(); finalize(); { const errSpan = document.createElement('span'); errSpan.style.color = 'var(--error)'; errSpan.textContent = data.content; bubble.appendChild(document.createElement('br')); bubble.appendChild(errSpan); } break;
                    }
                } catch (e) { console.warn('SSE parse error:', e, jsonStr); }
            }
        }

        finalize();
        removeThinking();

        if (!fullText) {
            if (bubble.textContent.trim() === '') {
                bubble.innerHTML = '（未获取到回复）';
            }
        } else {
            // 保存已有的 tool 标签，renderBubbleMarkdown 会覆盖 innerHTML
            const toolTags = Array.from(bubble.querySelectorAll('.tool-tag'));
            renderBubbleMarkdown(bubble, fullText);
            // 将 tool 标签重新插入到 bubble 开头
            if (toolTags.length > 0) {
                const fragment = document.createDocumentFragment();
                toolTags.forEach(tag => fragment.appendChild(tag));
                fragment.appendChild(document.createTextNode(' '));
                bubble.insertBefore(fragment, bubble.firstChild);
            }
        }

    } catch (e) {
        removeThinking();
        finalize();
        if (e.name === 'AbortError') {
            if (fullText) {
                renderBubbleMarkdown(bubble, fullText);
                const stopSpan = document.createElement('span');
                stopSpan.style.cssText = 'color:var(--text-secondary);font-size:13px;';
                stopSpan.textContent = '（已停止生成）';
                bubble.appendChild(document.createElement('br'));
                bubble.appendChild(stopSpan);
            } else {
                bubble.innerHTML = '<span style="color:var(--text-secondary)">已停止生成</span>';
            }
        } else {
            bubble.innerHTML = `<span style="color:var(--error)">网络错误，请重试</span>`;
        }
    } finally {
        resetStreamingUI();
    }
}

// ===== Markdown Rendering =====
function renderBubbleMarkdown(bubble, text) {
    if (typeof marked !== 'undefined' && text) {
        try {
            // 先用 marked 渲染 Markdown
            bubble.innerHTML = marked.parse(text);
            // 渲染后再替换下载链接为可点击按钮（避免 marked 过滤 HTML 标签）
            injectDownloadButtons(bubble);
            return;
        } catch (e) { console.warn('Markdown渲染失败', e); }
    }
    bubble.innerHTML = escapeHtml(text).replace(/\n/g, '<br>');
}

function injectDownloadButtons(container) {
    // [修复 v2] 更健壮的导出链接匹配：
    // 1. 容忍 URL 中的空格（流式渲染时 marked breaks:true 可能插入空格）
    // 2. 容忍 URL 被拆到多个文本节点（marked 可能把 URL 拆成 <em> 等子元素）
    // 3. 第三步兜底用 DOM 操作替代 innerHTML 字符串拼接（避免 onclick 引号转义问题）
    // 4. 处理 marked 自动把纯 URL 转成 <a href> 的情况（autolink 或 GFM）
    const EXPORT_URL_PATTERN = /\/api\/v1\/documents\/export[-\s]*download\/[^ \n\)<"\u0060]+\.(docx|xlsx|pdf|txt)/;
    const EXPORT_URL_GLOBAL = /(?:\/api\/v1\/documents\/export[-\s]*download\/[^ \n\)<"\u0060]+\.(docx|xlsx|pdf|txt))/g;
    const btnLabels = { docx: '点击下载Word文档', xlsx: '点击下载Excel表格', pdf: '点击下载PDF文档', txt: '点击下载文本文件' };

    // 工具函数：清理 URL（去除空格、修正格式）
    function cleanUrl(url) {
        return url
            .replace(/\s+/g, '')                    // 去除所有空格（流式渲染可能插入）
            .replace('/export/download/', '/export-download/')  // 修正斜杠格式
            .replace(/\/export-\s+download\//, '/export-download/');  // 修正 export- download 格式
    }

    // 1. 处理所有 <a> 标签中的导出链接
    //    覆盖：marked 渲染的 [文字](URL)、autolink 自动转的 <a href="URL">
    const existingLinks = container.querySelectorAll('a[href*="/api/v1/documents/export"], a[href*="api/v1/documents/export"]');
    existingLinks.forEach(a => {
        const href = a.getAttribute('href') || '';
        // 清理 href 后再匹配
        const cleanedHref = cleanUrl(href);
        if (!EXPORT_URL_PATTERN.test(cleanedHref)) return;
        const ext = cleanedHref.split('.').pop().toLowerCase();
        if (!['docx', 'xlsx', 'pdf', 'txt'].includes(ext)) return;
        const correctUrl = cleanUrl(cleanedHref.match(EXPORT_URL_PATTERN)[0]);
        a.className = 'doc-download-btn' + (ext === 'xlsx' ? ' xlsx-btn' : '');
        a.href = 'javascript:void(0)';
        a.removeAttribute('target');  // 防止新标签页打开
        a.textContent = btnLabels[ext] || '点击下载文档';
        // 用 addEventListener 而非 onclick 属性（避免 innerHTML 重写时丢失）
        a.onclick = function(e) { e.preventDefault(); e.stopPropagation(); downloadExportFile(correctUrl); };
    });

    // 2. 处理文本节点中的导出链接（LLM 直接输出 URL 文本，未被 marked 转成 <a>）
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null, false);
    const nodesToReplace = [];
    while (walker.nextNode()) {
        const node = walker.currentNode;
        if (node.nodeValue && EXPORT_URL_PATTERN.test(node.nodeValue)) {
            nodesToReplace.push(node);
        }
    }
    nodesToReplace.forEach(node => {
        const text = node.nodeValue;
        const urlMatch = text.match(EXPORT_URL_PATTERN);
        if (urlMatch) {
            const url = urlMatch[0];
            // 修正URL格式：清理空格 + 修正斜杠
            const correctUrl = cleanUrl(url);
            const ext = correctUrl.split('.').pop().toLowerCase();
            const btn = document.createElement('a');
            btn.className = 'doc-download-btn' + (ext === 'xlsx' ? ' xlsx-btn' : '');
            btn.href = 'javascript:void(0)';
            btn.textContent = btnLabels[ext] || '点击下载文档';
            btn.onclick = function(e) { e.preventDefault(); e.stopPropagation(); downloadExportFile(correctUrl); };
            const parent = node.parentNode;
            const beforeText = text.substring(0, text.indexOf(url)).replace(/下载链接[：:]*\s*$/, '');
            if (beforeText.trim()) {
                parent.insertBefore(document.createTextNode(beforeText), node);
            }
            parent.insertBefore(btn, node);
            const afterText = text.substring(text.indexOf(url) + url.length);
            if (afterText.trim()) {
                parent.insertBefore(document.createTextNode(afterText), node);
            }
            parent.removeChild(node);
        }
    });

    // [修复 v2] 3. 兜底检查：用 DOM 操作替代 innerHTML 字符串拼接
    // 旧版用 innerHTML.replace 把 URL 替换成 <a onclick="downloadExportFile('URL')">
    // 当 URL 含中文/特殊字符时，onclick 字符串里的引号会破坏 HTML 解析
    // 新版：再次扫描文本节点（覆盖 marked 把 URL 包在 <code>/<strong> 等元素里的情况）
    // 用 DOM API 创建按钮，避免 innerHTML 字符串拼接
    const walker2 = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null, false);
    const nodesToReplace2 = [];
    while (walker2.nextNode()) {
        const node = walker2.currentNode;
        // 跳过已经在按钮内的文本节点
        if (node.parentNode && node.parentNode.classList && node.parentNode.classList.contains('doc-download-btn')) continue;
        if (node.nodeValue && EXPORT_URL_PATTERN.test(node.nodeValue)) {
            nodesToReplace2.push(node);
        }
    }
    nodesToReplace2.forEach(node => {
        const text = node.nodeValue;
        const urlMatch = text.match(EXPORT_URL_PATTERN);
        if (urlMatch) {
            const url = urlMatch[0];
            const correctUrl = cleanUrl(url);
            const ext = correctUrl.split('.').pop().toLowerCase();
            const btn = document.createElement('a');
            btn.className = 'doc-download-btn' + (ext === 'xlsx' ? ' xlsx-btn' : '');
            btn.href = 'javascript:void(0)';
            btn.textContent = btnLabels[ext] || '点击下载文档';
            btn.onclick = function(e) { e.preventDefault(); e.stopPropagation(); downloadExportFile(correctUrl); };
            const parent = node.parentNode;
            const beforeText = text.substring(0, text.indexOf(url));
            if (beforeText.trim()) {
                parent.insertBefore(document.createTextNode(beforeText), node);
            }
            parent.insertBefore(btn, node);
            const afterText = text.substring(text.indexOf(url) + url.length);
            if (afterText.trim()) {
                parent.insertBefore(document.createTextNode(afterText), node);
            }
            parent.removeChild(node);
        }
    });
}

// ===== 导出文件下载（支持中文文件名） =====
async function downloadExportFile(url) {
    try {
        // [修复 v2] URL 完整性校验：防止流式渲染时点击到残缺 URL
        // 合法的导出 URL 必须以 /api/v1/documents/export-download/ 开头，且以文件扩展名结尾
        const validUrlPattern = /^\/api\/v1\/documents\/export-download\/[^]+\.(docx|xlsx|pdf|txt)$/i;
        if (!validUrlPattern.test(url)) {
            console.warn('下载URL不完整或格式错误:', url);
            showToast('文件链接尚未生成完毕，请稍候 1-2 秒后再试', 3000);
            return;
        }
        const headers = {};
        if (authToken) headers['Authorization'] = 'Bearer ' + authToken;
        const response = await fetch(url, { headers });
        if (!response.ok) {
            showToast('下载失败：' + response.status + ' ' + response.statusText, 3000);
            return;
        }
        // 从Content-Disposition提取文件名
        const disposition = response.headers.get('Content-Disposition');
        // 根据URL中的扩展名决定默认文件名
        const urlExt = url.split('.').pop().toLowerCase();
        const defaultNames = { docx: '导出文档.docx', xlsx: '导出表格.xlsx', pdf: '导出文档.pdf', txt: '导出文本.txt' };
        let filename = defaultNames[urlExt] || '导出文档.docx';
        if (disposition) {
            const utf8Match = disposition.match(/filename\*=UTF-8''(.+)/i);
            if (utf8Match) {
                try { filename = decodeURIComponent(utf8Match[1]); } catch(e) { filename = utf8Match[1]; }
            } else {
                const plainMatch = disposition.match(/filename="?([^"]+)"?/);
                if (plainMatch) filename = plainMatch[1];
            }
        }
        // 从URL提取文件名（兜底：默认文件名未被服务端覆盖时才使用URL中的文件名）
        if (filename === defaultNames[urlExt] || filename === '导出文档.docx') {
            const urlParts = url.split('/');
            const lastPart = urlParts[urlParts.length - 1];
            if (lastPart) { try { filename = decodeURIComponent(lastPart); } catch(e) { filename = lastPart; } }
        }
        const blob = await response.blob();
        const blobUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = blobUrl;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(blobUrl);
    } catch (e) {
        console.error('下载导出文件失败:', e);
        // 降级：直接在新标签页打开
        window.open(url, '_blank');
    }
}

// ===== Send Message =====
async function sendMessage() {
    if (isLoading) return;
    // [BUG FIX #1] 竞态条件修复：在 createNewChat() 之前就设置 isLoading
    // 防止快速双击/连按回车时，第二次调用在 await createNewChat() 期间
    // 仍通过 isLoading 检查（此时仍为 false），导致创建重复聊天会话
    isLoading = true;
    if (!currentChatId) {
        // 没有当前对话时自动创建新对话（点击智能体后直接发消息的场景）
        await createNewChat();
        if (!currentChatId) { isLoading = false; return; }  // 创建失败才退出，同时释放锁
    }
    const input = document.getElementById('msgInput');
    const message = input.value.trim();
    if (!message && !selectedFile) { isLoading = false; return; }
    const sendBtn = document.getElementById('sendBtn');
    sendBtn.disabled = true;

    try {
    document.getElementById('chatContent').classList.remove('centered');

    if (selectedFile && message) {
        const isImage = selectedFile.type.startsWith('image/');
        const icon = isImage ? '🖼️' : '📎';
        if (isImage && selectedFileBase64) {
            addMessageToUI('user', `${icon} ${selectedFile.name}\n${message}`, selectedFileBase64);
        } else {
            addMessageToUI('user', `${icon} ${selectedFile.name}\n${message}`);
        }
        input.value = ''; autoResize(input);
        const bubble = createStreamingBubble();
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('message', message);
        formData.append('session_id', currentChatId);
        formData.append('web_search', webSearchEnabled);
        formData.append('mode', currentMode);
        formData.append('deep_think', deepThinkEnabled);
        formData.append('skill', selectedSkill || '');
        // 智能体ID和任务描述
        if (currentAgentId) {
            formData.append('agent_id', currentAgentId);
            const curAgent = myAgents.find(a => a.id === currentAgentId);
            if (curAgent) formData.append('agent_task', curAgent.task);
        } else {
            formData.append('agent_id', '');
        }
        // 聊天框上传文件仅用于临时分析，不存入知识库
        formData.append('store_to_kb', 'false');
        await streamChat('/api/v1/chat-with-file/stream', { method: 'POST', body: formData, headers: authToken ? { 'Authorization': 'Bearer ' + authToken } : {} }, bubble);
        removeFile();
        await loadChatList();
    } else if (selectedFile && !message) {
        // 文件无消息时，自动添加分析提示，走聊天流式分析（不存知识库）
        addMessageToUI('user', `[上传文档] ${selectedFile.name}`);
        const bubble = createStreamingBubble();
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('message', '请分析这个文件的内容');
        formData.append('session_id', currentChatId);
        formData.append('web_search', webSearchEnabled);
        formData.append('mode', currentMode);
formData.append('skill', selectedSkill || '');
        formData.append('deep_think', deepThinkEnabled);
        if (currentAgentId) {
            formData.append('agent_id', currentAgentId);
            const curAgent = myAgents.find(a => a.id === currentAgentId);
            if (curAgent) formData.append('agent_task', curAgent.task);
        }
        formData.append('store_to_kb', 'false');
        await streamChat('/api/v1/chat-with-file/stream', { method: 'POST', body: formData, headers: authToken ? { 'Authorization': 'Bearer ' + authToken } : {} }, bubble);
        removeFile();
        await loadChatList();
    } else {
        lastMessageText = message;
        addMessageToUI('user', message);
        input.value = ''; autoResize(input);
        const bubble = createStreamingBubble();
        await streamChat('/api/v1/chat/stream', {
            method: 'POST',
            headers: apiHeaders(),
            body: JSON.stringify({ message, session_id: currentChatId, web_search: webSearchEnabled, mode: currentMode, deep_think: deepThinkEnabled, skill: selectedSkill || '', agent_id: currentAgentId || '', agent_task: (currentAgentId && myAgents.find(a => a.id === currentAgentId)) ? myAgents.find(a => a.id === currentAgentId).task : '' })
        }, bubble);
        await loadChatList();
    }
    scrollToBottom();
    } finally {
        resetStreamingUI();
    }
}

function sendQuick(text) {
    // 填入输入框但不自动发送，用户可编辑后发送
    const input = document.getElementById('msgInput');
    if (input) {
        input.value = text;
        autoResize(input);
        input.focus();
    }
}

function addMessageToUI(role, content, imageBase64) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = `message ${role}`;
    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    if (role === 'assistant') {
        renderBubbleMarkdown(bubble, content);
    } else {
        let htmlContent = escapeHtml(content).replace(/\n/g, '<br>');
        if (imageBase64) htmlContent += `<img class="chat-img" src="${imageBase64}" alt="上传的图片">`;
        bubble.innerHTML = htmlContent;
        bubble.style.whiteSpace = 'pre-wrap';
    }

    const actions = document.createElement('div');
    actions.className = 'message-actions';
    if (role === 'assistant') {
        actions.innerHTML = `
            <button class="msg-action-btn" title="复制" onclick="copyMessage(this)" aria-label="复制消息">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
            </button>
            <button class="msg-action-btn" title="重新生成" onclick="regenerateMessage(this)" aria-label="重新生成">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 4v6h6"/><path d="M3.51 15a9 9 0 102.13-9.36L1 10"/></svg>
            </button>
        `;
    } else {
        actions.innerHTML = `
            <button class="msg-action-btn" title="复制" onclick="copyMessage(this)" aria-label="复制消息">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
            </button>
        `;
    }

    div.appendChild(bubble);
    div.appendChild(actions);
    container.appendChild(div);

    document.getElementById('chatContent').classList.remove('centered');
    scrollToBottom();
}

// ===== Message Actions =====
function copyMessage(btn) {
    const messageDiv = btn.closest('.message');
    const bubble = messageDiv ? messageDiv.querySelector('.bubble') : null;
    if (!bubble) { showToast('复制失败：未找到消息内容'); return; }
    // 获取纯文本，排除代码块复制按钮的文字
    let text = bubble.innerText || bubble.textContent || '';
    // 去除代码块中的"复制"/"已复制"文字
    text = text.replace(/\n?复制\n?/g, '\n').replace(/\n?已复制\n?/g, '\n').trim();
    if (!text) { showToast('复制失败：内容为空'); return; }
    copyToClipboard(text, () => { showToast('已复制到剪贴板'); }, () => { showToast('复制失败，请手动复制'); });
}

async function regenerateMessage(btn) {
    if (isLoading) return;
    const messageDiv = btn.closest('.message');
    const prev = messageDiv.previousElementSibling;
    if (!prev || !prev.classList.contains('user')) { showToast('无法找到对应的用户消息'); return; }
    const userBubble = prev.querySelector('.bubble');
    const userText = userBubble.textContent || userBubble.innerText;
    messageDiv.remove();
    if (!currentChatId) return;
    isLoading = true;
    const sendBtn = document.getElementById('sendBtn');
    sendBtn.disabled = true;

    try {
    const bubble = createStreamingBubble();
    await streamChat('/api/v1/chat/stream', {
        method: 'POST',
        headers: apiHeaders(),
        body: JSON.stringify({ message: userText, session_id: currentChatId, web_search: webSearchEnabled, mode: currentMode, deep_think: deepThinkEnabled,
        skill: selectedSkill || '', agent_id: currentAgentId || '', agent_task: (currentAgentId && myAgents.find(a => a.id === currentAgentId)) ? myAgents.find(a => a.id === currentAgentId).task : '' })
    }, bubble);
    } finally {
        resetStreamingUI();
    }
}

function showTyping(show) { document.getElementById('typingIndicator').style.display = show ? 'block' : 'none'; if (show) scrollToBottom(); }

// ===== File Handling =====
function onFileSelected(event) {
    const file = event.target.files[0];
    if (file) {
        if (file.size > MAX_FILE_SIZE) { showToast('文件大小不能超过 50MB'); event.target.value = ''; return; }
        setFilePreview(file);
    }
}

function setFilePreview(file) {
    selectedFile = file;
    selectedFileBase64 = null;
    const isImage = file.type.startsWith('image/');
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileIcon').textContent = isImage ? '🖼️' : '📎';
    document.getElementById('fileBar').style.display = 'flex';
    document.getElementById('msgInput').placeholder = '针对此文件输入问题，或修改要求...';
    if (isImage) {
        const reader = new FileReader();
        reader.onload = function(e) { selectedFileBase64 = e.target.result; };
        reader.readAsDataURL(file);
    }
}

function removeFile() {
    selectedFile = null;
    selectedFileBase64 = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('fileBar').style.display = 'none';
    document.getElementById('fileIcon').textContent = '📎';
    document.getElementById('msgInput').placeholder = '输入问题，或粘贴/拖拽文件...';
}

// ===== Paste & Drag =====
document.addEventListener('DOMContentLoaded', function() {
    const msgInput = document.getElementById('msgInput');
    const inputContainer = document.querySelector('.input-container');

    msgInput.addEventListener('paste', function(e) {
        const items = e.clipboardData && e.clipboardData.items;
        if (!items) return;
        for (let i = 0; i < items.length; i++) {
            const item = items[i];
            if (item.type.startsWith('image/')) {
                e.preventDefault();
                const file = item.getAsFile();
                if (file) { if (file.size > MAX_FILE_SIZE) { showToast('图片大小不能超过 50MB'); return; } setFilePreview(file); showToast('已粘贴图片，输入问题后发送'); }
                return;
            }
            if (item.kind === 'file' && !item.type.startsWith('image/')) {
                e.preventDefault();
                const file = item.getAsFile();
                if (file) { setFilePreview(file); showToast('已粘贴文件，输入问题后发送'); }
                return;
            }
        }
    });

    inputContainer.addEventListener('dragover', function(e) { e.preventDefault(); e.stopPropagation(); inputContainer.style.borderColor = 'var(--accent)'; inputContainer.style.background = 'rgba(26,26,26,0.03)'; });
    inputContainer.addEventListener('dragleave', function(e) { e.preventDefault(); e.stopPropagation(); inputContainer.style.borderColor = ''; inputContainer.style.background = ''; });
    inputContainer.addEventListener('drop', function(e) { e.preventDefault(); e.stopPropagation(); inputContainer.style.borderColor = ''; inputContainer.style.background = ''; const files = e.dataTransfer.files; if (files.length > 0) { setFilePreview(files[0]); showToast('已添加文件，输入问题后发送'); } });
});

// ===== Knowledge Base Modal =====
async function showDocs() {
    document.getElementById('docsModal').classList.add('show');
    await loadDocList();
}
function closeDocs() { document.getElementById('docsModal').classList.remove('show'); document.getElementById('uploadProgress').style.display = 'none'; }

async function loadDocList() {
    const list = document.getElementById('docList');
    list.innerHTML = '<div class="doc-empty">加载中...</div>';
    try {
        // 按 agent_id 获取对应知识库的文档列表
        const agentParam = currentAgentId ? `?agent_id=${encodeURIComponent(currentAgentId)}` : '';
        const resp = await fetch(`/api/v1/documents${agentParam}`, { headers: apiHeaders() });
        const data = await resp.json();
        list.innerHTML = '';
        if (data.documents && data.documents.length > 0) {
            data.documents.forEach(doc => {
                const item = document.createElement('div');
                item.className = 'doc-item';
                let icon = '📄';
                if (doc.endsWith('.pdf')) icon = '📕';
                else if (doc.endsWith('.docx')) icon = '📘';
                else if (doc.endsWith('.xlsx') || doc.endsWith('.xls')) icon = '📊';
                else if (doc.endsWith('.txt')) icon = '📝';
                const safeName = escapeHtml(doc);
                const safeNameForAttr = doc.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '&quot;');
                item.innerHTML = `<span class="doc-icon">${icon}</span><span class="doc-name">${safeName}</span><button class="doc-download-btn" onclick="downloadDocument('${safeNameForAttr}')" title="下载" aria-label="下载文档">📥</button><button class="doc-delete-btn" onclick="deleteDocument('${safeNameForAttr}', this)">删除</button>`;
                list.appendChild(item);
            });
        } else { list.innerHTML = '<div class="doc-empty">暂无文档，请上传</div>'; }
    } catch (e) { list.innerHTML = '<div class="doc-empty">加载失败</div>'; }
}

async function onKbFileSelected(event) {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    for (let i = 0; i < files.length; i++) { await uploadToKnowledgeBase(files[i]); }
    document.getElementById('kbFileInput').value = '';
    await loadDocList();
}

async function deleteDocument(filename, btnEl) {
    if (!confirm(`确定要删除文档 "${filename}" 吗？此操作不可恢复！`)) return;
    const docItem = btnEl.closest('.doc-item');
    btnEl.disabled = true; btnEl.textContent = '删除中...';
    try {
        const agentParam = currentAgentId ? `?agent_id=${encodeURIComponent(currentAgentId)}` : '';
        const resp = await fetch(`/api/v1/documents/${encodeURIComponent(filename)}${agentParam}`, { method: 'DELETE', headers: apiHeaders() });
        const data = await resp.json();
        if (resp.ok && data.status === 'success') {
            docItem.style.transition = 'all 0.3s'; docItem.style.opacity = '0'; docItem.style.transform = 'translateX(20px)';
            setTimeout(() => { docItem.remove(); const list = document.getElementById('docList'); if (list.children.length === 0) list.innerHTML = '<div class="doc-empty">暂无文档，请上传</div>'; }, 300);
            // 同步刷新右侧KB面板
            if (currentAgentId) loadKbDocs();
        } else { alert('删除失败：' + (data.detail || '未知错误')); btnEl.disabled = false; btnEl.textContent = '删除'; }
    } catch (e) { alert('删除失败：网络错误'); btnEl.disabled = false; btnEl.textContent = '删除'; }
}

async function uploadToKnowledgeBase(file) {
    const progressEl = document.getElementById('uploadProgress');
    const fileNameEl = document.getElementById('progressFileName');
    const barFill = document.getElementById('progressBarFill');
    const statusEl = document.getElementById('progressStatus');
    progressEl.style.display = 'block';
    const isImage = file.type && file.type.startsWith('image/');
    const kbLabel = currentAgentId ? `智能体「${myAgents.find(a => a.id === currentAgentId)?.name || ''}」知识库` : '知识库';
    fileNameEl.textContent = `${isImage ? '🖼️' : '📎'} ${file.name} → ${kbLabel}${isImage ? '（VLM解析中）' : ''}`;
    barFill.style.width = '10%';
    statusEl.textContent = '上传中...';
    statusEl.className = 'progress-status';
    const formData = new FormData();
    formData.append('file', file);
    if (currentAgentId) formData.append('agent_id', currentAgentId);
    formData.append('category', currentKbCategory || '');
    try {
        barFill.style.width = '30%';
        const resp = await fetch('/api/v1/upload', { method: 'POST', body: formData, headers: authToken ? { 'Authorization': 'Bearer ' + authToken } : {} });
        barFill.style.width = '80%';
        const data = await resp.json();
        if (resp.ok) { barFill.style.width = '100%'; statusEl.textContent = `✅ 上传成功！文档已索引到${kbLabel}`; statusEl.className = 'progress-status success'; }
        else { barFill.style.width = '100%'; barFill.style.background = 'var(--error)'; statusEl.textContent = '❌ 上传失败：' + (data.detail || '未知错误'); statusEl.className = 'progress-status error'; }
    } catch (e) { barFill.style.width = '100%'; barFill.style.background = 'var(--error)'; statusEl.textContent = '❌ 网络错误，请重试'; statusEl.className = 'progress-status error'; }
    setTimeout(() => { progressEl.style.display = 'none'; barFill.style.background = 'var(--accent)'; }, 3000);
}

function downloadDocument(filename) {
    // 在新标签页打开下载链接
    const agentParam = currentAgentId ? `?agent_id=${encodeURIComponent(currentAgentId)}` : '';
    const url = `/api/v1/documents/${encodeURIComponent(filename)}/download${agentParam}`;
    window.open(url, '_blank');
}

// ===== Utility Functions =====
function formatTime(timestamp) {
    if (!timestamp) return '';
    const now = Date.now() / 1000;
    const diff = now - timestamp;
    if (diff < 60) return '刚刚';
    if (diff < 3600) return Math.floor(diff / 60) + '分钟前';
    if (diff < 86400) return Math.floor(diff / 3600) + '小时前';
    if (diff < 604800) return Math.floor(diff / 86400) + '天前';
    const d = new Date(timestamp * 1000);
    return `${d.getMonth() + 1}/${d.getDate()}`;
}

function escapeHtml(str) {
    if (str == null) return '';
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function handleKey(event) { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendMessage(); } }
function autoResize(el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 120) + 'px'; }

// ===== Chat Search =====

// ===== Export Chat =====
function toggleExportDropdown() {
    const dropdown = document.getElementById('exportDropdown');
    dropdown.classList.toggle('show');
    // Close when clicking outside
    if (dropdown.classList.contains('show')) {
        setTimeout(() => {
            document.addEventListener('click', closeExportDropdown, { once: true });
        }, 0);
    }
}

function closeExportDropdown(e) {
    const dropdown = document.getElementById('exportDropdown');
    if (dropdown && !dropdown.contains(e.target)) {
        dropdown.classList.remove('show');
    }
}

// ===== Skills Dropdown =====
function toggleSkillsDropdown(event) {
    event.stopPropagation();
    const dropdown = document.getElementById('skillsDropdown');
    dropdown.classList.toggle('show');
    if (dropdown.classList.contains('show')) {
        setTimeout(() => {
            document.addEventListener('click', closeSkillsDropdown, { once: true });
        }, 0);
    }
}

function closeSkillsDropdown(e) {
    const dropdown = document.getElementById('skillsDropdown');
    const btn = document.getElementById('headerSkillsBtn');
    if (dropdown && !dropdown.contains(e.target) && btn && !btn.contains(e.target)) {
        dropdown.classList.remove('show');
    }
}

function selectSkill(skillId) {
    const dropdown = document.getElementById('skillsDropdown');
    if (dropdown) dropdown.classList.remove('show');
    selectedSkill = skillId;
    // 显示技能模式提示栏
    const bar = document.getElementById('skillModeBar');
    const text = document.getElementById('skillModeText');
    const hint = document.getElementById('skillModeHint');
    if (bar && text) {
        if (skillId === '8d-skill') {
            text.textContent = '8D SKILL模式';
            if (hint) hint.textContent = '当前启用了8D技能，AI将按8D流程生成报告';
        } else if (skillId === 'pfmea-dfmea-skill') {
            text.textContent = 'FMEA SKILL模式';
            if (hint) hint.textContent = '当前启用了FMEA技能，AI将按FMEA七步法生成报告';
        }
        bar.style.display = '';
    }
    // 同步隐藏知识库上传模式（互斥）
    if (agentKbUploadMode) toggleAgentKbUpload();
    const skillDisplay = skillId === 'pfmea-dfmea-skill' ? 'PFMEA/DFMEA' : '8D';
    showToast('已启用 ' + skillDisplay + ' SKILL 模式');
}

function clearSkill() {
    selectedSkill = null;
    const bar = document.getElementById('skillModeBar');
    if (bar) bar.style.display = 'none';
    showToast('已退出技能模式');
}

// 开发中 skill 点击：仅展示提示，不切换 selectedSkill
function showDevSkillToast(skillName) {
    const dropdown = document.getElementById('skillsDropdown');
    if (dropdown) dropdown.classList.remove('show');
    showToast('「' + skillName + '」 Skill需要公司资深专家参与，敬请期待', 3500);
}

async function exportChat(format) {
    if (!currentChatId) return;
    const dropdown = document.getElementById('exportDropdown');
    if (dropdown) dropdown.classList.remove('show');

    // 获取当前智能体名称，用于文件名
    let agentName = '';
    if (currentAgentId) {
        const agent = myAgents.find(a => a.id === currentAgentId);
        if (agent) agentName = agent.name;
    }

    try {
        const params = new URLSearchParams({ format });
        if (agentName) params.set('agent_name', agentName);
        const resp = await fetch(`/api/v1/export/${currentChatId}?${params.toString()}`, { headers: apiHeaders() });
        if (!resp.ok) { showToast('导出失败'); return; }

        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const extMap = { docx: 'docx', pdf: 'pdf', md: 'md' };
        const nameMap = { docx: 'Word', pdf: 'PDF', md: 'Markdown' };
        const ext = extMap[format] || 'md';
        // 文件名包含智能体名称
        const safeName = agentName ? agentName.replace(/[\\/:*?"<>|]/g, '_') : 'chat';
        a.download = `${safeName}_对话记录.${ext}`;
        a.click();
        URL.revokeObjectURL(url);
        showToast(`已导出为 ${nameMap[format] || format.toUpperCase()}`);
    } catch (e) {
        showToast('导出失败');
    }
}

// ===== Knowledge Base Panel =====
function toggleKbPanel() {
    const panel = document.getElementById('kbPanel');
    if (!panel) return;
    const wasShown = panel.classList.contains('show');
    panel.classList.toggle('show');
    
    if (!wasShown) {
        // Update agent name display
        const agentNameEl = document.getElementById('kbAgentName');
        if (currentAgentId) {
            const agent = myAgents.find(a => a.id === currentAgentId);
            if (agentNameEl) agentNameEl.textContent = agent ? agent.name : '';
        } else {
            if (agentNameEl) agentNameEl.textContent = '（未选择智能体）';
        }
        const uploadBtn = document.querySelector('.kb-panel-upload');
        if (uploadBtn) uploadBtn.style.display = currentAgentId ? '' : 'none';
        loadKbDocs();
        setTimeout(() => { document.addEventListener('click', closeKbPanel, { once: true }); }, 0);
    }
}

function closeKbPanel(e) {
    const panel = document.getElementById('kbPanel');
    if (panel && !panel.contains(e.target) && !e.target.closest('.kb-btn')) {
        panel.classList.remove('show');
    }
}

async function loadKbDocs() {
    const listEl = document.getElementById('kbDocList');
    if (!currentAgentId) {
        listEl.innerHTML = '<div class="kb-empty">请先选择一个智能体</div>';
        return;
    }
    listEl.innerHTML = '<div class="kb-empty">加载中...</div>';
    try {
        const resp = await fetch(`/api/v1/documents?agent_id=${encodeURIComponent(currentAgentId)}`, { headers: apiHeaders() });
        const data = await resp.json();
        console.log('[KB] loadKbDocs response:', JSON.stringify(data));
        // Handle multiple response formats - docs can be strings or objects
        let docs = data.documents || data.files || [];
        if (!Array.isArray(docs)) docs = [];
        // Extract filenames from objects if needed
        docs = docs.map(d => typeof d === 'string' ? d : (d.filename || d.name || d.title || String(d)));
        
        if (docs.length === 0) {
            listEl.innerHTML = '<div class="kb-empty">暂无文档，点击上方按钮上传</div>';
            return;
        }
        let html = '<div class="kb-doc-count">共 ' + docs.length + ' 个文档</div>';
        docs.forEach(docName => {
            const ext = docName.split('.').pop().toLowerCase();
            const icon = ext === 'pdf' ? '📕' : ext === 'docx' ? '📘' : '📄';
            html += '<div class="kb-doc-item">' +
                '<div class="kb-doc-info">' +
                '<span class="kb-doc-icon">' + icon + '</span>' +
                '<span class="kb-doc-name" title="' + escapeHtml(docName) + '">' + escapeHtml(docName) + '</span>' +
                '</div>' +
                (userRole === 'admin' ? '<button class="kb-doc-delete" onclick="deleteKbDoc(\'' + docName.replace(/'/g, "\\'") + '\')" title="删除文档">🗑️</button>' : '') +
                '</div>';
        });
        listEl.innerHTML = html;
    } catch (e) {
        console.error('加载知识库文档列表失败', e);
        listEl.innerHTML = '<div class="kb-empty">加载失败，请重试</div>';
    }
}

async function uploadKbDoc(input) {
    if (!input.files || !input.files[0]) return;
    const file = input.files[0];
    if (!currentAgentId) {
        showToast('请先选择一个智能体');
        input.value = '';
        return;
    }
    showToast('正在上传并索引...');
    const formData = new FormData();
    formData.append('file', file);
    formData.append('agent_id', currentAgentId);
    try {
        const resp = await fetch('/api/v1/upload', { method: 'POST', body: formData, headers: authToken ? { 'Authorization': 'Bearer ' + authToken } : {} });
        const data = await resp.json();
        if (data.status === 'success') {
            const chunks = data.detail?.chunks || 0;
            showToast(`文档已上传，共 ${chunks} 个分块`);
            loadKbDocs();
        } else {
            showToast(data.detail || '上传失败');
        }
    } catch (e) {
        showToast('上传失败，请重试');
    }
    input.value = '';
}

async function deleteKbDoc(filename) {
    if (userRole !== 'admin') { showToast('仅管理员可删除文档'); return; }
    if (!confirm(`确定删除文档「${filename}」？`)) return;
    try {
        const agentParam = currentAgentId ? `?agent_id=${encodeURIComponent(currentAgentId)}` : '';
        const resp = await fetch(`/api/v1/documents/${encodeURIComponent(filename)}${agentParam}`, { method: 'DELETE', headers: apiHeaders() });
        const data = await resp.json();
        if (data.status === 'success') {
            showToast('文档已删除');
            loadKbDocs();
        } else {
            showToast(data.detail?.message || data.message || '删除失败');
        }
    } catch (e) {
        showToast('删除失败，请重试');
    }
}

// ===== File Drag to Chat Area =====
(function() {
    const chatContent = document.getElementById('chatContent');
    if (!chatContent) return;
    chatContent.addEventListener('dragover', (e) => { e.preventDefault(); e.stopPropagation(); chatContent.classList.add('drag-over'); });
    chatContent.addEventListener('dragleave', (e) => { e.preventDefault(); e.stopPropagation(); chatContent.classList.remove('drag-over'); });
    chatContent.addEventListener('drop', (e) => { e.preventDefault(); e.stopPropagation(); chatContent.classList.remove('drag-over'); const files = e.dataTransfer.files; if (files.length > 0) handleDroppedFile(files[0]); });
})();

function handleDroppedFile(file) {
    const validExts = ['.pdf','.txt','.docx','.png','.jpg','.jpeg','.gif','.bmp','.webp','.csv','.xlsx','.xls','.doc','.ppt','.pptx','.md','.json','.py','.js','.html','.css'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!validExts.includes(ext)) { showToast('不支持的文件格式'); return; }
    if (file.size > 50 * 1024 * 1024) { showToast('文件大小超过50MB限制'); return; }
    selectedFile = file;
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileBar').style.display = 'flex';
    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => { selectedFileBase64 = e.target.result; };
        reader.readAsDataURL(file);
    } else { selectedFileBase64 = null; }
    showToast('文件已添加：' + file.name);
}

// ===== Mobile Keyboard =====
if (/Mobi|Android/i.test(navigator.userAgent)) {
    window.visualViewport && window.visualViewport.addEventListener('resize', () => {
        const chatContent = document.getElementById('chatContent');
        if (chatContent && document.activeElement && document.activeElement.tagName === 'TEXTAREA') {
            // Adjust layout for virtual keyboard
            const viewportHeight = window.visualViewport.height;
            chatContent.style.height = viewportHeight + 'px';
            setTimeout(() => scrollToBottom(), 100);
        } else {
            chatContent.style.height = '';
        }
    });
    window.visualViewport && window.visualViewport.addEventListener('scroll', () => {
        const chatContent = document.getElementById('chatContent');
        if (chatContent && document.activeElement && document.activeElement.tagName === 'TEXTAREA') {
            // Scroll input into view
            const inputArea = document.querySelector('.chat-input-area');
            if (inputArea) {
                inputArea.scrollIntoView({ block: 'end' });
            }
        }
    });
}

// ===== Init =====
document.addEventListener('DOMContentLoaded', async function() {
    // Drag upload zone
    const uploadZone = document.getElementById('uploadZone');
    uploadZone.addEventListener('dragover', function(e) { e.preventDefault(); e.stopPropagation(); uploadZone.classList.add('dragover'); });
    uploadZone.addEventListener('dragleave', function(e) { e.preventDefault(); e.stopPropagation(); uploadZone.classList.remove('dragover'); });
    uploadZone.addEventListener('drop', function(e) {
        e.preventDefault(); e.stopPropagation(); uploadZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            (async () => { for (let i = 0; i < files.length; i++) { await uploadToKnowledgeBase(files[i]); } await loadDocList(); })();
        }
    });

    // Scroll detection
    setupScrollDetection();

    // Centered mode init
    updateCenteredMode();

    // [禁用自动登录] 每次访问必须手动输入用户名密码
    localStorage.removeItem('authToken');

    // [BUG FIX] Set initial history state for login page
    // This ensures the browser back button has a proper state to return to
    history.replaceState({page: 'login'}, '');

    // Landing page: nav scroll & smooth scroll (宣传页已删除，跳过)

    // Sync agents when tab becomes visible (cross-browser prompt sync)
    // [#12] 不传force=true，受5秒防抖限制，避免频繁Alt-Tab触发大量请求
    document.addEventListener('visibilitychange', async function() {
        if (!document.hidden && currentUser && authToken) {
            await syncAgentsFromServer();
        }
        // [性能修复] 页面隐藏时清理内存，防止长时间打开页面变慢
        if (document.hidden) {
            cleanupFrontendMemory();
        }
    });

    // Landing page: scroll-reveal animation with IntersectionObserver
    const revealElements = document.querySelectorAll('.reveal');
    if (revealElements.length > 0 && 'IntersectionObserver' in window) {
        // Add .reveal-init to enable animation (content visible by default without it)
        revealElements.forEach(function(el) { el.classList.add('reveal-init'); });
        const revealObserver = new IntersectionObserver(function(entries) {
            entries.forEach(function(entry) {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    revealObserver.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });
        revealElements.forEach(function(el) { revealObserver.observe(el); });
    }
});

// ===== Knowledge Base Full Page =====
// ===== 体系调研上传处理 =====
async function onSurveyFileSelected(event) {
    const file = event.target.files[0];
    if (!file) return;

    const statusEl = document.getElementById('surveyUploadStatus');
    const uploadZone = document.getElementById('surveyUploadZone');

    // 显示处理中
    statusEl.className = 'survey-upload-status processing';
    statusEl.innerHTML = '⏳ 正在分析文档并提取信息，请稍候...';
    statusEl.style.display = 'block';

    try {
        // 1. 上传文件到服务器
        const formData = new FormData();
        formData.append('file', file);

        const uploadResp = await fetch('/api/v1/survey/upload', {
            method: 'POST',
            headers: { 'Authorization': 'Bearer ' + authToken },
            body: formData
        });
        const uploadData = await uploadResp.json();

        if (!uploadData.success) {
            throw new Error(uploadData.message || '上传失败');
        }

        // 2. 调用 AI 提取信息
        statusEl.innerHTML = '🤖 AI 正在识别文档中的企业信息...';

        const extractResp = await fetch('/api/v1/survey/extract', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + authToken
            },
            body: JSON.stringify({
                file_path: uploadData.file_path,
                filename: file.name
            })
        });
        const extractData = await extractResp.json();

        if (extractData.success && extractData.fields) {
            // 3. 自动填充表单
            const filledFields = fillSurveyForm(extractData.fields);

            // 4. 显示成功提示
            statusEl.className = 'survey-upload-status success';
            const fieldTags = filledFields.map(f => '<span class="field-updated">' + f + '</span>').join('');
            statusEl.innerHTML = '✓ 已从文档中识别并填充 ' + filledFields.length + ' 个字段：' + fieldTags + '<br><span style="font-size:12px;opacity:0.8;">请检查并补充未识别的字段</span>';
        } else {
            throw new Error(extractData.message || 'AI 提取失败');
        }
    } catch (error) {
        console.error('[调研上传] 错误:', error);
        statusEl.className = 'survey-upload-status error';
        statusEl.innerHTML = '❌ ' + error.message + '<br><span style="font-size:12px;">请手动填写表单</span>';
    }

    // 清空 input，允许重复上传
    event.target.value = '';
}

function fillSurveyForm(fields) {
    const filled = [];
    const fieldMap = {
        // 基本信息映射
        'company_name': 'sv_company_name',
        'cert_other': 'sv_cert_other',
        'chairman': 'sv_chairman',
        'legal_rep': 'sv_legal_rep',
        'gm': 'sv_gm',
        'deputy_gm': 'sv_deputy_gm',
        'mgmt_rep': 'sv_mgmt_rep',
        'leader_group_leader': 'sv_leader_group_leader',
        'leader_group_members': 'sv_leader_group_members',
        'iso_office_head': 'sv_iso_office_head',
        'iso_office_members': 'sv_iso_office_members',
        'auditors': 'sv_auditors',
        'products': 'sv_products',
        'process_flow': 'sv_process_flow',
        'location': 'sv_location',
        'area': 'sv_area',
        'building_area': 'sv_building_area',
        'staff_total': 'sv_staff_total',
        'staff_mgmt': 'sv_staff_mgmt',
        'staff_edu': 'sv_staff_edu',
        'equipment': 'sv_equipment',
        'customers': 'sv_customers',
        'address': 'sv_address',
        'contact': 'sv_contact',
        'phone': 'sv_phone',
        'fax': 'sv_fax',
        'mobile': 'sv_mobile',
        'purpose': 'sv_purpose',
        'quality_policy': 'sv_quality_policy',
        'quality_goal': 'sv_quality_goal',
        'cert_date': 'sv_cert_date',
        'audit_date': 'sv_audit_date',
        'rest_day': 'sv_rest_day',
        'design_dev': 'sv_design_dev',
        'filler_name': 'sv_filler_name',
        'filler_phone': 'sv_filler_phone',
    };

    const fieldLabels = {
        'company_name': '公司名称',
        'chairman': '董事长',
        'legal_rep': '法人代表',
        'gm': '总经理',
        'deputy_gm': '副总经理',
        'mgmt_rep': '管理者代表',
        'products': '体系覆盖产品',
        'process_flow': '生产流程',
        'location': '地理位置',
        'address': '公司地址',
        'contact': '联系人',
        'mobile': '手机',
        'purpose': '公司宗旨',
        'quality_policy': '质量方针',
        'quality_goal': '质量目标',
        'filler_name': '填写人',
    };

    // 填充文本字段
    Object.keys(fields).forEach(key => {
        const inputId = fieldMap[key];
        if (inputId && fields[key]) {
            const el = document.getElementById(inputId);
            if (el && !el.value) {
                // 只填充空字段，不覆盖已有内容
                el.value = fields[key];
                const label = fieldLabels[key] || key;
                if (!filled.includes(label)) filled.push(label);
            } else if (el && el.value) {
                // 已有值也更新（AI 数据可能更准确）
                el.value = fields[key];
                const label = fieldLabels[key] || key;
                if (!filled.includes(label)) filled.push(label);
            }
        }
    });

    // 填充证书复选框
    if (fields.certs && Array.isArray(fields.certs)) {
        document.querySelectorAll('.sv_cert').forEach(cb => {
            if (fields.certs.includes(cb.value)) {
                cb.checked = true;
                if (!filled.includes('认证证书')) filled.push('认证证书');
            }
        });
    }

    // 填充机构设置
    if (fields.org && typeof fields.org === 'object') {
        Object.keys(fields.org).forEach(funcKey => {
            const deptInput = document.querySelector('#sv_org_table input[data-field="' + funcKey + '_dept"]');
            const headInput = document.querySelector('#sv_org_table input[data-field="' + funcKey + '_head"]');
            if (deptInput && fields.org[funcKey].dept) deptInput.value = fields.org[funcKey].dept;
            if (headInput && fields.org[funcKey].head) headInput.value = fields.org[funcKey].head;
        });
        if (!filled.includes('机构设置')) filled.push('机构设置');
    }

    return filled;
}


// ===== 外部知识库分类管理 =====
let currentExtKbCategory = '体系文件-手册';

function toggleExtCatGroup(btnEl) {
    const group = btnEl.closest('.kb-cat-group');
    if (group) group.classList.toggle('collapsed');
}

function selectExtKbCategory(cat, btnEl) {
    currentExtKbCategory = cat;
    document.querySelectorAll('#extKbCatList .kb-cat-item').forEach(b => b.classList.remove('active'));
    if (btnEl) btnEl.classList.add('active');
    const titleEl = document.getElementById('extKbFileTitle');
    if (titleEl) titleEl.textContent = cat.split('-').pop();
    loadExtKbDocs();
}

function addExtKbSubCategory(parentGroup) {
    const name = prompt('请输入新分类名称：');
    if (!name || !name.trim()) return;
    const subListId = parentGroup === '按产品分类' ? 'extKbProductList' :
                      parentGroup === '按工艺分类' ? 'extKbProcessList' : null;
    if (subListId) {
        const subList = document.getElementById(subListId);
        if (subList) {
            const btn = document.createElement('button');
            btn.className = 'kb-cat-item';
            btn.textContent = name.trim();
            btn.onclick = function() { selectExtKbCategory(parentGroup + '-' + name.trim(), this); };
            subList.appendChild(btn);
        }
    } else {
        // 其他分类：在最后一个 group 的 sub-list 里添加
        const groups = document.querySelectorAll('#extKbCatList .kb-cat-group');
        if (groups.length >= 4) {
            const subList = groups[3].querySelector('.kb-cat-sub-list');
            if (subList) {
                const btn = document.createElement('button');
                btn.className = 'kb-cat-item';
                btn.textContent = name.trim();
                btn.onclick = function() { selectExtKbCategory('其他-' + name.trim(), this); };
                subList.appendChild(btn);
            }
        }
    }
    showToast('已添加：' + name.trim(), 2000);
}

async function loadExtKbDocs() {
    const docList = document.getElementById('extKbDocList');
    if (!docList) return;
    docList.innerHTML = '<div class="kb-doc-empty">加载中...</div>';
    
    try {
        const url = '/api/v1/external-kb/documents?category=' + encodeURIComponent(currentExtKbCategory || '');
        const resp = await fetch(url, {
            headers: { 'Authorization': 'Bearer ' + authToken }
        });
        const data = await resp.json();
        
        if (data.success && data.documents && data.documents.length > 0) {
            let html = '';
            data.documents.forEach(doc => {
                const name = doc.filename || doc.name || doc;
                html += '<div class="kb-doc-item"><span class="kb-doc-name">' + escapeHtml(name) + '</span><button class="kb-doc-del-btn" onclick="deleteExtKbDoc(\'' + name + '\')">删除</button></div>';
            });
            docList.innerHTML = html;
        } else {
            docList.innerHTML = '<div class="kb-doc-empty">暂无文件，点击右上角上传</div>';
        }
        
        // 更新统计（文档数 + 切片数）
        const statUrl = '/api/v1/external-kb/stats?category=' + encodeURIComponent(currentExtKbCategory || '');
        const statResp = await fetch(statUrl, {
            headers: { 'Authorization': 'Bearer ' + authToken }
        });
        const statData = await statResp.json();
        if (statData.success) {
            const docEl = document.getElementById('extKbStatDocCount');
            const chunkEl = document.getElementById('extKbStatChunkCount');
            if (docEl) docEl.textContent = statData.doc_count || 0;
            if (chunkEl) chunkEl.textContent = statData.chunk_count || 0;
        }
    } catch (e) {
        docList.innerHTML = '<div class="kb-doc-empty">暂无文件，点击右上角上传</div>';
    }
}

async function deleteExtKbDoc(filename) {
    if (!confirm('确认删除「' + filename + '」？')) return;
    try {
        const resp = await fetch('/api/v1/external-kb/documents/' + encodeURIComponent(filename), {
            method: 'DELETE',
            headers: { 'Authorization': 'Bearer ' + authToken }
        });
        const data = await resp.json();
        if (data.success) {
            showToast('已删除: ' + filename, 2000);
            loadExtKbDocs();
        } else {
            showToast('删除失败', 2000);
        }
    } catch (e) {
        showToast('删除失败: ' + e.message, 2000);
    }
}

async function onExtKbFileSelected(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    showToast('正在上传: ' + file.name, 2000);
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('category', currentExtKbCategory || '');
        
        const resp = await fetch('/api/v1/external-kb/upload', {
            method: 'POST',
            headers: { 'Authorization': 'Bearer ' + authToken },
            body: formData
        });
        const data = await resp.json();
        
        if (data.success) {
            const chunks = data.chunks || 0;
            showToast('✓ 上传成功: ' + file.name + (chunks ? '（共 ' + chunks + ' 个分块）' : ''), 3000);
            loadExtKbDocs();
        } else {
            showToast('上传失败: ' + (data.detail || data.message || '未知错误'), 3000);
        }
    } catch (e) {
        showToast('上传失败: ' + e.message, 3000);
    }
    event.target.value = '';
}

// ===== 体系调研表单 =====
function showSurveyForm() {
    const surveyPage = document.getElementById('surveyPage');
    const chatContent = document.getElementById('chatContent');
    const kbPage = document.getElementById('kbPage');
    const helpPage = document.getElementById('helpPage');
    const externalKbPage = document.getElementById('externalKbPage');
    if (kbPage) kbPage.style.display = 'none';
    if (helpPage) helpPage.style.display = 'none';
    if (externalKbPage) externalKbPage.style.display = 'none';
    if (chatContent) chatContent.style.display = 'none';
    if (chatContent) chatContent.style.display = 'none';
    if (surveyPage) {
        surveyPage.style.display = 'block';
        loadSurveyData();
    }
    // 选中智能体
    currentAgentId = 'dfmea-risk-agent';
    currentMode = 'agent';
    localStorage.setItem('chatMode', 'agent');
    updateGenButtonsVisibility();
    updateHeaderKbVisibility();
    renderMyAgents();
    // push history
    history.pushState({page: 'survey'}, '');
}

function hideSurveyForm() {
    const surveyPage = document.getElementById('surveyPage');
    const chatContent = document.getElementById('chatContent');
    if (surveyPage) surveyPage.style.display = 'none';
    if (chatContent) chatContent.style.display = 'flex';
    // 显示欢迎页
    const welcomeEl = document.getElementById('welcomeCenter');
    if (welcomeEl) welcomeEl.style.display = '';
    // 更新 history state（不调 history.back，避免 popstate 干扰）
    history.replaceState({page: 'chat'}, '');
    // 渲染智能体列表和会话列表
    renderMyAgents();
    loadChatList();
    updateGenButtonsVisibility();
    updateHeaderKbVisibility();
}

function collectSurveyData() {
    const data = {};
    // 文本字段
    const textFields = [
        'sv_company_name', 'sv_cert_other', 'sv_chairman', 'sv_legal_rep',
        'sv_gm', 'sv_deputy_gm', 'sv_mgmt_rep',
        'sv_leader_group_leader', 'sv_leader_group_members',
        'sv_iso_office_head', 'sv_iso_office_members', 'sv_auditors',
        'sv_products', 'sv_process_flow',
        'sv_location', 'sv_area', 'sv_building_area',
        'sv_staff_total', 'sv_staff_mgmt', 'sv_staff_edu',
        'sv_equipment', 'sv_customers',
        'sv_address', 'sv_contact', 'sv_phone', 'sv_fax', 'sv_mobile',
        'sv_purpose', 'sv_quality_policy', 'sv_quality_goal',
        'sv_cert_date', 'sv_audit_date', 'sv_rest_day', 'sv_design_dev',
        'sv_filler_name', 'sv_filler_phone'
    ];
    textFields.forEach(id => {
        const el = document.getElementById(id);
        if (el) data[id] = el.value;
    });
    // 证书复选框
    const certs = [];
    document.querySelectorAll('.sv_cert:checked').forEach(cb => certs.push(cb.value));
    if (data.sv_cert_other) certs.push(data.sv_cert_other);
    data.sv_certs = certs;
    // 机构设置表格
    data.sv_org = {};
    document.querySelectorAll('#sv_org_table input[type="text"]').forEach(inp => {
        data.sv_org[inp.getAttribute('data-field')] = inp.value;
    });
    return data;
}

function saveSurveyData() {
    const data = collectSurveyData();
    // 验证必填字段
    if (!data.sv_company_name) { showToast('请填写公司名称', 3000); return; }
    if (!data.sv_products) { showToast('请填写体系覆盖的产品', 3000); return; }
    if (!data.sv_filler_name) { showToast('请填写填写人姓名', 3000); return; }
    if (!data.sv_filler_phone) { showToast('请填写填写人手机', 3000); return; }
    // 保存到 localStorage
    localStorage.setItem('surveyData', JSON.stringify(data));
    // 隐藏表单，显示聊天界面
    const surveyPage = document.getElementById('surveyPage');
    const chatContent = document.getElementById('chatContent');
    if (surveyPage) surveyPage.style.display = 'none';
    if (chatContent) chatContent.style.display = 'flex';
    // 显示欢迎页
    const welcomeEl = document.getElementById('welcomeCenter');
    if (welcomeEl) welcomeEl.style.display = '';
    // 更新 history
    history.replaceState({page: 'chat'}, '');
    // 刷新界面
    renderMyAgents();
    updateGenButtonsVisibility();
    updateHeaderKbVisibility();
    // 如果没有当前对话，创建一个新对话
    if (!currentChatId) {
        createNewChat().then(() => {
            loadChatList();
            showToast('✓ 体系调研信息已保存，点击左侧按钮可一键生成文档', 3000);
        });
    } else {
        loadChatList();
        showToast('✓ 体系调研信息已保存，点击左侧按钮可一键生成文档', 3000);
    }
}

function saveSurveyDraft() {
    const data = collectSurveyData();
    localStorage.setItem('surveyData', JSON.stringify(data));
    showToast('💾 草稿已暂存', 2000);
}

function clearSurveyData() {
    if (!confirm('确认清空所有填写内容？此操作不可恢复。')) return;
    // 清空所有文本输入
    const inputs = document.querySelectorAll('#surveyPage input[type="text"], #surveyPage input[type="date"], #surveyPage textarea, #surveyPage select');
    inputs.forEach(inp => inp.value = '');
    // 清空复选框
    document.querySelectorAll('#surveyPage input[type="checkbox"]').forEach(cb => cb.checked = false);
    // 清空机构设置表格
    document.querySelectorAll('#sv_org_table input[type="text"]').forEach(inp => inp.value = '');
    // 删除 localStorage
    localStorage.removeItem('surveyData');
    showToast('已清空所有填写内容', 2000);
}

function loadSurveyData() {
    const saved = localStorage.getItem('surveyData');
    if (!saved) return;
    try {
        const data = JSON.parse(saved);
        // 文本字段
        Object.keys(data).forEach(key => {
            if (key === 'sv_certs' || key === 'sv_org') return;
            const el = document.getElementById(key);
            if (el) el.value = data[key];
        });
        // 证书复选框
        if (data.sv_certs) {
            document.querySelectorAll('.sv_cert').forEach(cb => {
                cb.checked = data.sv_certs.includes(cb.value);
            });
        }
        // 机构设置
        if (data.sv_org) {
            Object.keys(data.sv_org).forEach(field => {
                const inp = document.querySelector('#sv_org_table input[data-field="' + field + '"]');
                if (inp) inp.value = data.sv_org[field];
            });
        }
    } catch(e) {
        console.warn('加载调研数据失败:', e);
    }
}

function getSurveyData() {
    const saved = localStorage.getItem('surveyData');
    if (!saved) return null;
    try {
        return JSON.parse(saved);
    } catch(e) {
        return null;
    }
}

function formatSurveyDataForAI() {
    const data = getSurveyData();
    if (!data) return '';
    let text = '【体系调研信息】\n';
    text += '公司名称: ' + (data.sv_company_name || '未填写') + '\n';
    text += '认证证书: ' + ((data.sv_certs || []).join(', ')) + '\n';
    text += '董事长: ' + (data.sv_chairman || '') + '  法人代表: ' + (data.sv_legal_rep || '') + '\n';
    text += '总经理: ' + (data.sv_gm || '') + '  副总经理: ' + (data.sv_deputy_gm || '') + '\n';
    text += '管理者代表: ' + (data.sv_mgmt_rep || '') + '\n';
    text += '体系覆盖产品: ' + (data.sv_products || '') + '\n';
    text += '生产流程: ' + (data.sv_process_flow || '') + '\n';
    text += '公司地址: ' + (data.sv_address || '') + '\n';
    text += '联系人: ' + (data.sv_contact || '') + '  手机: ' + (data.sv_mobile || '') + '\n';
    text += '占地面积: ' + (data.sv_area || '') + '㎡  建筑面积: ' + (data.sv_building_area || '') + '㎡\n';
    text += '正式员工: ' + (data.sv_staff_total || '') + '人  管理技术人员: ' + (data.sv_staff_mgmt || '') + '人\n';
    text += '公司宗旨: ' + (data.sv_purpose || '') + '\n';
    text += '质量方针: ' + (data.sv_quality_policy || '') + '\n';
    text += '质量目标: ' + (data.sv_quality_goal || '') + '\n';
    text += '有无设计开发: ' + (data.sv_design_dev || '') + '\n';
    text += '填写人: ' + (data.sv_filler_name || '') + '  手机: ' + (data.sv_filler_phone || '') + '\n';
    // 机构设置
    if (data.sv_org) {
        text += '\n机构设置:\n';
        Object.keys(data.sv_org).forEach(k => {
            if (data.sv_org[k] && k.endsWith('_dept')) {
                const funcName = k.replace('_dept', '').replace('org_', '');
                text += '  ' + funcName + ': ' + data.sv_org[k] + ' (负责人: ' + (data.sv_org[k.replace('_dept','_head')] || '') + ')\n';
            }
        });
    }
    return text;
}

// ===== 一键生成文档 =====
function generateDocument(type) {
    const typeMap = {
        'manual': '一键生成手册',
        'procedure': '一键生成程序文件',
        'third-level': '一键生成三层次文件',
        'record': '一键生成记录表格',
        'rectification': '不合格项整改'
    };
    const typeName = typeMap[type] || type;
    
    // 一键生成手册：调用 SCskill API（SSE 流式接收进度）
    if (type === 'manual') {
        const surveyData = getSurveyData();
        if (!surveyData) {
            showToast('请先点击"填写体系调研"填写企业信息', 3000);
            showSurveyForm();
            return;
        }
        const surveyPage = document.getElementById('surveyPage');
        if (surveyPage && surveyPage.style.display !== 'none') {
            hideSurveyForm();
        }
        document.getElementById('chatContent').classList.remove('centered');
        const bubble = createStreamingBubble();
        const bubbleContent = bubble.querySelector('.bubble') || bubble;

        (async () => {
            if (isLoading) return;
            isLoading = true;
            if (!currentChatId) {
                await createNewChat();
                if (!currentChatId) { isLoading = false; return; }
            }
            try {
                // 渲染初始容器
                bubbleContent.innerHTML = `
                    <div class="gen-manual-progress">
                        <div class="gen-progress-header">
                            <span class="gen-spinner"></span>
                            <span class="gen-step-text">正在启动...</span>
                        </div>
                        <div class="gen-progress-bar-wrap">
                            <div class="gen-progress-bar" style="width:0%"></div>
                        </div>
                        <div class="gen-message">准备中...</div>
                        <div class="gen-modifications"></div>
                    </div>
                `;
                scrollToBottom();

                const stepTextEl = bubbleContent.querySelector('.gen-step-text');
                const progressBarEl = bubbleContent.querySelector('.gen-progress-bar');
                const messageEl = bubbleContent.querySelector('.gen-message');
                const modsEl = bubbleContent.querySelector('.gen-modifications');

                // 调用 SSE 接口
                const genResp = await fetch('/api/v1/generate/manual', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer ' + authToken,
                        'Accept': 'text/event-stream'
                    },
                    body: JSON.stringify({ survey_data: surveyData })
                });

                if (!genResp.ok) {
                    const errText = await genResp.text();
                    throw new Error('HTTP ' + genResp.status + ': ' + errText);
                }

                // 用 ReadableStream 读取 SSE
                const reader = genResp.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buffer = '';
                let receivedCount = 0;
                let totalMods = 0;
                let modificationLog = []; // 收集所有修改记录，用于最终展示

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });

                    // 按 SSE 协议解析（事件以 \n\n 分隔）
                    const events = buffer.split('\n\n');
                    buffer = events.pop(); // 最后一段可能不完整，保留

                    for (const evt of events) {
                        const lines = evt.split('\n');
                        let dataStr = '';
                        for (const line of lines) {
                            if (line.startsWith('data:')) {
                                dataStr += line.substring(5).trim();
                            }
                        }
                        if (!dataStr) continue;
                        let data;
                        try { data = JSON.parse(dataStr); }
                        catch (e) { console.warn('[SSE] 解析失败:', dataStr); continue; }

                        if (data.type === 'progress') {
                            stepTextEl.textContent = data.step || '处理中';
                            if (typeof data.progress === 'number') {
                                progressBarEl.style.width = data.progress + '%';
                            }
                            messageEl.textContent = data.message || '';
                            // 应用修改阶段时显示头部（兼容旧版）
                            if (data.step === '应用修改方案' && !modsEl.querySelector('.gen-mods-header')) {
                                modsEl.innerHTML = '<div class="gen-mods-header">正在应用的修改：</div><div class="gen-mods-list"></div>';
                            }
                        } else if (data.type === 'modifications_start') {
                            // 新增：开始接收修改方案时立即显示头部
                            if (!modsEl.querySelector('.gen-mods-header')) {
                                modsEl.innerHTML = '<div class="gen-mods-header">' + (data.message || '正在应用的修改：') + '</div><div class="gen-mods-list"></div>';
                            } else if (data.message) {
                                modsEl.querySelector('.gen-mods-header').textContent = data.message;
                            }
                        } else if (data.type === 'modification') {
                            receivedCount++;
                            const list = modsEl.querySelector('.gen-mods-list') || modsEl;
                            const modItem = document.createElement('div');
                            modItem.className = 'gen-mod-item gen-mod-' + (data.mod_type || 'other');
                            const reasonText = data.reason ? ` <span class="gen-mod-reason">${data.reason}</span>` : '';
                            modItem.innerHTML = `
                                <span class="gen-mod-badge">${data.location || ''}</span>
                                <span class="gen-mod-preview">${data.preview || ''}</span>${reasonText}
                            `;
                            list.appendChild(modItem);
                            modificationLog.push({
                                location: data.location,
                                preview: data.preview,
                                reason: data.reason,
                                mod_type: data.mod_type
                            });
                            if (typeof data.progress === 'number') {
                                progressBarEl.style.width = data.progress + '%';
                            }
                            scrollToBottom();
                        } else if (data.type === 'success') {
                            totalMods = data.modifications_count || receivedCount;
                            const stats = data.stats || {};
                            const statLines = [];
                            if (stats.paragraph) statLines.push(`段落 ${stats.paragraph} 处`);
                            if (stats.table_cell) statLines.push(`表格 ${stats.table_cell} 处`);
                            if (stats.global_replace) statLines.push(`全文替换 ${stats.global_replace} 处`);
                            if (stats.header_replace) statLines.push(`页眉页脚 ${stats.header_replace} 处`);
                            const statText = statLines.join('，') || '无修改';

                            progressBarEl.style.width = '100%';
                            stepTextEl.textContent = '完成';
                            stepTextEl.previousElementSibling.style.display = 'none'; // 隐藏 spinner

                            // 显示完整结果
                            bubbleContent.innerHTML = `
                                <div class="gen-manual-success">
                                    <p class="gen-success-title">✓ 质量手册已生成完成</p>
                                    <p class="gen-success-info">使用模型：${data.model_used || '未知'} ｜ 共 ${totalMods} 个修改方案</p>
                                    <p class="gen-success-stats">修改统计：${statText}</p>
                                    <details class="gen-mods-details">
                                        <summary>查看详细修改记录（共 ${modificationLog.length} 条）</summary>
                                        <div class="gen-mods-detail-list">
                                            ${modificationLog.map(m => `
                                                <div class="gen-mod-detail-item">
                                                    <span class="gen-mod-badge">${m.location || ''}</span>
                                                    <span class="gen-mod-preview">${(m.preview || '').replace(/</g, '&lt;')}</span>
                                                    ${m.reason ? `<span class="gen-mod-reason">${m.reason}</span>` : ''}
                                                </div>
                                            `).join('')}
                                        </div>
                                    </details>
                                    <br>
                                    <a href="${data.download_url}" class="doc-download-btn xlsx-btn" style="display:inline-block;padding:10px 20px;background:#15589B;color:#fff;text-decoration:none;border-radius:8px;font-weight:bold;">点击下载质量手册</a>
                                </div>
                            `;
                            scrollToBottom();

                            // 保存到对话记录
                            if (currentChatId) {
                                await fetch('/api/v1/history/' + currentChatId, { method: 'GET', headers: apiHeaders() });
                            }
                            await loadChatList();
                        } else if (data.type === 'error') {
                            bubbleContent.innerHTML = `<p style="color:#e63946;">生成失败：${data.message || '未知错误'}</p>`;
                        }
                    }
                }
            } catch (e) {
                console.error('[生成手册] 失败:', e);
                bubbleContent.innerHTML = '<p style="color:#e63946;">生成失败：' + e.message + '</p>';
            } finally {
                resetStreamingUI();
            }
        })();
        return;
    }
    
    // 付费功能：三层次文件、记录表格、不合格项整改
    if (type === 'third-level' || type === 'record' || type === 'rectification') {
        const surveyData = getSurveyData();
        if (!surveyData) {
            showToast('请先点击"填写体系调研"填写企业信息', 3000);
            showSurveyForm();
            return;
        }
        const surveyPage = document.getElementById('surveyPage');
        if (surveyPage && surveyPage.style.display !== 'none') {
            hideSurveyForm();
        }
        document.getElementById('chatContent').classList.remove('centered');
        const bubble = createStreamingBubble();
        const payMsg = '本版本为试用，如需使用：\n请汇款至：\n账户名称：北京全质科技股份有限公司\n账户号码：11050163810000000267\n开户银行：中国建设银行股份有限公司北京北洼路支行\n（提供6%的增值税专用发票）\n或联系售前服务电话（微信同号）：18601256219';
        (async () => {
            if (isLoading) return;
            isLoading = true;
            if (!currentChatId) {
                await createNewChat();
                if (!currentChatId) { isLoading = false; return; }
            }
            try {
                await streamChat('/api/v1/chat/stream', {
                    method: 'POST',
                    headers: apiHeaders(),
                    body: JSON.stringify({
                        message: '请回复以下内容，原样输出：' + payMsg,
                        session_id: currentChatId,
                        web_search: false,
                        mode: currentMode,
                        deep_think: false,
                        skill: '',
                        agent_id: currentAgentId || '',
                        agent_task: (currentAgentId && myAgents.find(a => a.id === currentAgentId)) ? myAgents.find(a => a.id === currentAgentId).task : ''
                    })
                }, bubble);
                await loadChatList();
                scrollToBottom();
            } catch (e) {
                console.error('[付费提示] 失败:', e);
            } finally {
                resetStreamingUI();
            }
        })();
        return;
    }
    const surveyData = getSurveyData();
    if (!surveyData) {
        showToast('请先点击"填写体系调研"填写企业信息', 3000);
        showSurveyForm();
        return;
    }
    const surveyPage = document.getElementById('surveyPage');
    if (surveyPage && surveyPage.style.display !== 'none') {
        hideSurveyForm();
    }
    const surveyText = formatSurveyDataForAI();
    // 不显示任何用户消息，直接创建 AI 回复气泡
    document.getElementById('chatContent').classList.remove('centered');
    const bubble = createStreamingBubble();
    
    (async () => {
        if (isLoading) { showToast('请等待当前回复完成', 2000); return; }
        isLoading = true;
        // 确保有当前对话
        if (!currentChatId) {
            await createNewChat();
            if (!currentChatId) {
                isLoading = false;
                showToast('创建对话失败，请重试', 3000);
                return;
            }
        }
        const sendBtn = document.getElementById('sendBtn');
        if (sendBtn) sendBtn.disabled = true;
        try {
            await streamChat('/api/v1/chat/stream', {
                method: 'POST',
                headers: apiHeaders(),
                body: JSON.stringify({
                    message: '请' + typeName + '。\n\n' + surveyText + '\n\n基于以上体系调研信息，生成完整的' + typeName + '。',
                    session_id: currentChatId,
                    web_search: webSearchEnabled,
                    mode: currentMode,
                    deep_think: deepThinkEnabled,
                    skill: type,
                    agent_id: currentAgentId || '',
                    agent_task: (currentAgentId && myAgents.find(a => a.id === currentAgentId)) ? myAgents.find(a => a.id === currentAgentId).task : ''
                })
            }, bubble);
            await loadChatList();
            scrollToBottom();
        } catch (e) {
            console.error('[生成文档] 失败:', e);
        } finally {
            resetStreamingUI();
        }
    })();
}

// ===== 生成按钮显示/隐藏逻辑 =====
function updateGenButtonsVisibility() {
    const genSection = document.getElementById('sidebarGenSection');
    if (genSection) {
        // 选中智能体后显示生成按钮
        genSection.style.display = currentAgentId ? 'block' : 'none';
    }
}

// ===== 帮助页面 =====
function showHelpPage() {
    const helpPage = document.getElementById('helpPage');
    const chatContent = document.getElementById('chatContent');
    const kbPage = document.getElementById('kbPage');
    const externalKbPage = document.getElementById('externalKbPage');
    const surveyPage = document.getElementById('surveyPage');
    if (kbPage) kbPage.style.display = 'none';
    if (externalKbPage) externalKbPage.style.display = 'none';
    if (surveyPage) surveyPage.style.display = 'none';
    if (chatContent) chatContent.style.display = 'none';
    if (helpPage) helpPage.style.display = '';
    // push history state，让浏览器后退按钮能返回聊天页
    history.pushState({page: 'help'}, '');
}

function hideHelpPage() {
    const helpPage = document.getElementById('helpPage');
    const chatContent = document.getElementById('chatContent');
    if (helpPage) helpPage.style.display = 'none';
    if (chatContent) chatContent.style.display = '';
    // 触发浏览器后退
    window._navigatingFromHelp = true;
    history.back();
}

// ===== 外部知识库页面 =====
function showExternalKbPage() {
    const externalKbPage = document.getElementById('externalKbPage');
    const chatContent = document.getElementById('chatContent');
    const kbPage = document.getElementById('kbPage');
    const helpPage = document.getElementById('helpPage');
    const surveyPage = document.getElementById('surveyPage');
    if (kbPage) kbPage.style.display = 'none';
    if (helpPage) helpPage.style.display = 'none';
    if (surveyPage) surveyPage.style.display = 'none';
    if (chatContent) chatContent.style.display = 'none';
    if (externalKbPage) externalKbPage.style.display = '';
    // push history state
    history.pushState({page: 'external_kb'}, '');
}

function hideExternalKbPage() {
    const externalKbPage = document.getElementById('externalKbPage');
    const chatContent = document.getElementById('chatContent');
    if (externalKbPage) externalKbPage.style.display = 'none';
    if (chatContent) chatContent.style.display = '';
    // 触发浏览器后退
    window._navigatingFromExternalKb = true;
    history.back();
}

// ===== 知识库分类管理 =====
let currentKbCategory = '手册';
let kbCategories = ['手册', '程序文件', '三层次文件', '记录表格', '其他'];

function selectKbCategory(cat, btnEl) {
    currentKbCategory = cat;
    // 更新选中样式
    document.querySelectorAll('.kb-cat-item').forEach(b => b.classList.remove('active'));
    if (btnEl) btnEl.classList.add('active');
    // 更新右列标题
    const titleEl = document.getElementById('kbFileTitle');
    if (titleEl) titleEl.textContent = cat;
    // 重新加载该分类的文件
    loadKbPageDocs();
}

function addKbCategory() {
    const name = prompt('请输入新分类名称：');
    if (!name || !name.trim()) return;
    name = name.trim();
    if (kbCategories.includes(name)) {
        showToast('该分类已存在', 2000);
        return;
    }
    kbCategories.push(name);
    // 在 DOM 中添加新按钮
    const catList = document.getElementById('kbCatList');
    if (catList) {
        const btn = document.createElement('button');
        btn.className = 'kb-cat-item';
        btn.innerHTML = '  ' + escapeHtml(name) + ' <span class="cat-del" onclick="delKbCategory(\'' + name + '\', event)">×</span>';
        btn.onclick = function() { selectKbCategory(name, this); };
        catList.appendChild(btn);
    }
    showToast('已添加分类：' + name, 2000);
}

function delKbCategory(name, event) {
    event.stopPropagation();
    if (!confirm('确认删除分类「' + name + '」？该分类下的文件不会被删除。')) return;
    kbCategories = kbCategories.filter(c => c !== name);
    // 如果删除的是当前选中的，切换到第一个
    if (currentKbCategory === name) {
        currentKbCategory = kbCategories[0] || '手册';
        const firstBtn = document.querySelector('.kb-cat-item');
        if (firstBtn) selectKbCategory(currentKbCategory, firstBtn);
    }
    // 从 DOM 移除
    document.querySelectorAll('.kb-cat-item').forEach(btn => {
        if (btn.textContent.includes(name)) btn.remove();
    });
    showToast('已删除分类：' + name, 2000);
}

function showKbPage() {
    if (!currentAgentId) {
        showToast('请先选择一个智能体');
        return;
    }
    const chatContent = document.getElementById('chatContent');
    const kbPage = document.getElementById('kbPage');
    const surveyPage = document.getElementById('surveyPage');
    const helpPage = document.getElementById('helpPage');
    const externalKbPage = document.getElementById('externalKbPage');
    if (surveyPage) surveyPage.style.display = 'none';
    if (helpPage) helpPage.style.display = 'none';
    if (externalKbPage) externalKbPage.style.display = 'none';
    const sidebar = document.getElementById('sidebar');
    const sidebarOverlay = document.getElementById('sidebarOverlay');
    chatContent.style.display = 'none';
    kbPage.style.display = 'flex';
    // 隐藏侧边栏
    if (sidebar) sidebar.style.display = 'none';
    if (sidebarOverlay) sidebarOverlay.style.display = 'none';
    // Update title
    const agent = myAgents.find(a => a.id === currentAgentId);
    const agentName = agent ? agent.name : '智能体';
    document.getElementById('kbPageTitle').textContent = agentName + ' - 知识库管理';
    document.getElementById('kbPageDesc').textContent = '上传和管理' + agentName + '相关文档，系统将自动进行向量化处理';
    // [BUG FIX] 推入历史状态，让浏览器←按钮能回到聊天页
    history.pushState({page: 'kb'}, '');
    // Load docs
    loadKbPageDocs();
    // Setup drag and drop
    setupKbPageDragDrop();
}

function hideKbPage() {
    const chatContent = document.getElementById('chatContent');
    const kbPage = document.getElementById('kbPage');
    const sidebar = document.getElementById('sidebar');
    kbPage.style.display = 'none';
    chatContent.style.display = 'flex';
    // 恢复侧边栏
    if (sidebar) sidebar.style.display = '';
    updateCenteredMode();
    // [BUG FIX] 使用 history.back() 弹出 kb 条目，而不是 replaceState 堆积 chat 条目
    // 旧代码 replaceState({page:'chat'}) 会把 kb 条目替换成 chat，导致 history 栈
    // 堆积大量 chat 条目，用户点后退时在 chat→chat 间跳转，UI 不变，看起来"没反应"
    // 改用 history.back() 让浏览器自动 pop kb 条目，回到前一个 chat 条目
    // popstate 监听器会接管 UI 切换（幂等，重复执行无副作用）
    if (history.state && history.state.page === 'kb') {
        // 设置标志位，告诉 popstate 监听器这是 hideKbPage 主动触发的后退
        // 不要误判为"chat→chat 堆积"而连续后退（那会错误地退到 login 页）
        window._navigatingFromKb = true;
        history.back();
    }
}

async function loadKbPageDocs() {
    const listEl = document.getElementById('kbPageDocList');
    if (!currentAgentId) {
        listEl.innerHTML = '<div class="kb-doc-empty">请先选择一个智能体</div>';
        return;
    }
    listEl.innerHTML = '<div class="kb-doc-empty">加载中...</div>';
    try {
        const url = '/api/v1/documents?agent_id=' + encodeURIComponent(currentAgentId) + '&category=' + encodeURIComponent(currentKbCategory || '');
        const resp = await fetch(url, { headers: apiHeaders() });
        const data = await resp.json();
        let docs = data.documents || data.files || [];
        if (!Array.isArray(docs)) docs = [];
        docs = docs.map(d => typeof d === 'string' ? d : (d.filename || d.name || d.title || String(d)));
        
        // Update stats
        document.getElementById('kbStatDocCount').textContent = docs.length;
        // Get chunk count from stats API
        let totalChunks = 0;
        try {
            const chunkResp = await fetch('/api/v1/documents/stats?agent_id=' + encodeURIComponent(currentAgentId), { headers: apiHeaders() });
            if (chunkResp.ok) {
                const chunkData = await chunkResp.json();
                totalChunks = chunkData.total_chunks || 0;
            }
        } catch(e) { console.warn('获取知识库统计失败', e); }
        document.getElementById('kbStatChunkCount').textContent = totalChunks;
        
        if (docs.length === 0) {
            listEl.innerHTML = '<div class="kb-doc-empty">暂无文档，请点击上方区域上传</div>';
            return;
        }
        let html = '';
        docs.forEach(docName => {
            const ext = docName.split('.').pop().toLowerCase();
            let iconHtml = '';
            if (ext === 'pdf') {
                iconHtml = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1051BF" stroke-width="1.5" stroke-linecap="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>';
            } else if (ext === 'docx' || ext === 'doc') {
                iconHtml = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="1.5" stroke-linecap="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>';
            } else if (ext === 'xlsx' || ext === 'xls') {
                iconHtml = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="1.5" stroke-linecap="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><rect x="8" y="12" width="8" height="6" rx="1"/></svg>';
            } else {
                iconHtml = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="1.5" stroke-linecap="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>';
            }
            const safeName = escapeHtml(docName);
            const safeNameForJs = docName.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '&quot;');
            html += '<div class="kb-doc-item">' +
                '<div class="kb-doc-icon">' + iconHtml + '</div>' +
                '<div class="kb-doc-info">' +
                '<div class="kb-doc-name" title="' + safeName + '">' + safeName + '</div>' +
                '<div class="kb-doc-meta">' + ext.toUpperCase() + '</div>' +
                '</div>' +
                (userRole === 'admin' ? '<button class="kb-doc-delete-btn" onclick="deleteKbPageDoc(\'' + safeNameForJs + '\', this)" title="删除文档" aria-label="删除">' +
                '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>' +
                ' 删除</button>' : '') +
                '</div>';
        });
        listEl.innerHTML = html;
    } catch (e) {
        console.error('加载知识库文档失败', e);
        listEl.innerHTML = '<div class="kb-doc-empty">加载失败，请重试</div>';
    }
}

async function onKbPageFileSelected(event) {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    for (let i = 0; i < files.length; i++) {
        await uploadToKbPage(files[i]);
    }
    event.target.value = '';
    await loadKbPageDocs();
}

async function uploadToKbPage(file) {
    const progressEl = document.getElementById('kbPageProgress');
    const fileNameEl = document.getElementById('kbProgressFileName');
    const barFill = document.getElementById('kbProgressBarFill');
    const statusEl = document.getElementById('kbProgressStatus');
    progressEl.style.display = 'block';
    const isImage = file.type && file.type.startsWith('image/');
    const agent = myAgents.find(a => a.id === currentAgentId);
    const kbLabel = agent ? agent.name + ' 知识库' : '知识库';
    fileNameEl.textContent = (isImage ? '🖼️ ' : '') + file.name + ' → ' + kbLabel + (isImage ? '（VLM解析中）' : '');
    barFill.style.width = '10%';
    statusEl.textContent = '上传中...';
    statusEl.className = 'kb-progress-status';
    const formData = new FormData();
    formData.append('file', file);
    if (currentAgentId) formData.append('agent_id', currentAgentId);
    formData.append('category', currentKbCategory || '');
    try {
        barFill.style.width = '30%';
        const resp = await fetch('/api/v1/upload', { method: 'POST', body: formData, headers: authToken ? { 'Authorization': 'Bearer ' + authToken } : {} });
        barFill.style.width = '80%';
        const data = await resp.json();
        if (resp.ok && (data.status === 'success' || data.filename)) {
            barFill.style.width = '100%';
            const chunks = data.detail?.chunks || data.chunks || 0;
            statusEl.textContent = '上传成功！' + (chunks ? '共 ' + chunks + ' 个分块' : '');
            statusEl.className = 'kb-progress-status success';
        } else {
            barFill.style.width = '100%';
            barFill.style.background = '#1051BF';
            statusEl.textContent = '上传失败：' + (data.detail || '未知错误');
            statusEl.className = 'kb-progress-status error';
        }
    } catch (e) {
        barFill.style.width = '100%';
        barFill.style.background = '#1051BF';
        statusEl.textContent = '网络错误，请重试';
        statusEl.className = 'kb-progress-status error';
    }
    setTimeout(() => { progressEl.style.display = 'none'; barFill.style.background = ''; }, 3000);
}

async function deleteKbPageDoc(filename, btnEl) {
    if (userRole !== 'admin') { showToast('仅管理员可删除文档'); return; }
    if (!confirm('确定删除文档「' + filename + '」？此操作不可恢复！')) return;
    const docItem = btnEl.closest('.kb-doc-item');
    btnEl.disabled = true;
    btnEl.textContent = '删除中...';
    try {
        const agentParam = currentAgentId ? '?agent_id=' + encodeURIComponent(currentAgentId) : '';
        const resp = await fetch('/api/v1/documents/' + encodeURIComponent(filename) + agentParam, { method: 'DELETE', headers: apiHeaders() });
        const data = await resp.json();
        if (data.status === 'success') {
            docItem.style.transition = 'all 0.3s';
            docItem.style.opacity = '0';
            docItem.style.transform = 'translateX(20px)';
            setTimeout(() => {
                docItem.remove();
                const list = document.getElementById('kbPageDocList');
                if (list.children.length === 0) list.innerHTML = '<div class="kb-doc-empty">暂无文档，请点击上方区域上传</div>';
                // Update stats
                const countEl = document.getElementById('kbStatDocCount');
                const current = parseInt(countEl.textContent) || 0;
                countEl.textContent = Math.max(0, current - 1);
            }, 300);
        } else {
            showToast('删除失败：' + (data.detail?.message || data.message || '未知错误'));
            btnEl.disabled = false;
            btnEl.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg> 删除';
        }
    } catch (e) {
        showToast('删除失败：网络错误');
        btnEl.disabled = false;
        btnEl.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg> 删除';
    }
}

// [BUG FIX #3] 防重入守卫：标记拖拽事件是否已绑定，避免重复绑定
let _kbPageDragDropBound = false;

function setupKbPageDragDrop() {
    const zone = document.getElementById('kbPageUploadZone');
    if (!zone) return;
    // [BUG FIX #3] 如果已经绑定过事件监听器，直接返回，防止重复绑定
    // 每次打开知识库页面 showKbPage() 都会调用此函数，但事件监听器不会自动移除
    // 第N次打开后拖放文件会触发N次上传，导致同一文件被重复上传N次
    if (_kbPageDragDropBound) return;
    _kbPageDragDropBound = true;
    zone.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        zone.classList.add('drag-over');
    });
    zone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        zone.classList.remove('drag-over');
    });
    zone.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        zone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            for (let i = 0; i < files.length; i++) {
                uploadToKbPage(files[i]);
            }
            setTimeout(() => loadKbPageDocs(), 1500);
        }
    });
}

