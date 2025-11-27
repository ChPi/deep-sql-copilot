class SQLCopilotApp {
    constructor() {
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendMessage');
        this.chatMessages = document.getElementById('chatMessages');
        this.clearChatButton = document.getElementById('clearChat');
        this.initDbButton = document.getElementById('initDb');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.toast = document.getElementById('toast');

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadChatHistory();
        this.setWelcomeTime();
        this.autoResizeTextarea();
    }

    setupEventListeners() {
        // Send message on button click
        this.sendButton.addEventListener('click', () => this.sendMessage());

        // Send message on Enter key (but allow Shift+Enter for new line)
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => this.autoResizeTextarea());

        // Clear chat history
        this.clearChatButton.addEventListener('click', () => this.clearChat());

        // Initialize database
        this.initDbButton.addEventListener('click', () => this.initializeDatabase());

        // Focus on input when page loads
        this.messageInput.focus();
    }

    autoResizeTextarea() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }

    setWelcomeTime() {
        const welcomeTime = document.getElementById('welcomeTime');
        if (welcomeTime) {
            welcomeTime.textContent = this.formatTime(new Date());
        }
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();

        if (!message) {
            this.showToast('请输入您的问题', 'warning');
            return;
        }

        // Disable input and show loading
        this.setLoadingState(true);

        try {
            // Add user message to chat
            this.addMessage('user', message);

            // Clear input
            this.messageInput.value = '';
            this.autoResizeTextarea();

            // Send to backend with streaming
            const response = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            
            // Variables for streaming response
            let botMessageContainer = null;
            let currentContentDiv = null;
            let currentNode = null;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.slice(6));

                        if (data.type === 'chunk') {
                            // Create bot message container only when we have actual content
                            if (!botMessageContainer) {
                                botMessageContainer = this.createBotMessageContainer();
                            }
                            
                            // Handle new node or existing node chunk
                            if (currentNode !== data.node) {
                                currentNode = data.node;
                                currentContentDiv = this.createNodeBlock(botMessageContainer, data.node);
                            }

                            // Append content
                            if (currentContentDiv) {
                                // Check if content is markdown code block
                                currentContentDiv.innerHTML = this.formatMessageContent(data.content);
                                this.scrollToBottom();
                            }
                        } else if (data.type === 'complete') {
                            // Final answer
                            this.addMessage('bot', data.content, data.sql);
                        } else if (data.type === 'error') {
                            this.addMessage('error', data.content);
                        } else if (data.type === 'interrupt') {
                            this.addMessage('bot', data.content);
                        }
                    }
                }
            }

        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage('error', '网络错误，请检查连接后重试');
        } finally {
            this.setLoadingState(false);
        }
    }

    createBotMessageContainer() {
        const messageElement = document.createElement('div');
        messageElement.className = 'message bot-message';

        messageElement.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="message-text" style="background: transparent; border: none; padding: 0; box-shadow: none;">
                    <!-- Nodes will be appended here -->
                </div>
            </div>
        `;

        this.chatMessages.appendChild(messageElement);
        return messageElement.querySelector('.message-text');
    }

    createNodeBlock(container, nodeName) {
        const block = document.createElement('div');
        block.className = `node-block node-${nodeName}`;

        const header = document.createElement('div');
        header.className = 'node-header';


        // Icon mapping
        let icon = 'fas fa-cog';
        if (nodeName === 'planner') icon = 'fas fa-map';
        if (nodeName === 'sql_coder') icon = 'fas fa-code';
        if (nodeName === 'executor') icon = 'fas fa-database';
        if (nodeName === 'refiner') icon = 'fas fa-check-double';

        header.innerHTML = `
        <i class="${icon}"></i> ${nodeName}
        <button class="collapse-btn">
            <i class="fas fa-chevron-down"></i>
        </button>
    `;

        const content = document.createElement('div');
        content.className = 'node-content';

        block.appendChild(header);
        block.appendChild(content);
        container.appendChild(block);
        const collapseBtn = header.querySelector('.collapse-btn');
        collapseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const isCollapsed = content.style.display === 'none';

            if (isCollapsed) {
                content.style.display = 'block';
                collapseBtn.innerHTML = '<i class="fas fa-chevron-down"></i>';
            } else {
                content.style.display = 'none';
                collapseBtn.innerHTML = '<i class="fas fa-chevron-right"></i>';
            }
        });
        return content;
    }

    addMessage(type, content, sql = null) {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${type}-message`;

        const avatarIcon = type === 'user' ? 'fas fa-user' : 'fas fa-robot';
        const avatarClass = type === 'user' ? 'user-message' : 'bot-message';

        let sqlHtml = '';
        if (sql && type === 'bot') {
            sqlHtml = `<div class="sql-code">${this.escapeHtml(sql)}</div>`;
        }

        messageElement.innerHTML = `
            <div class="message-avatar">
                <i class="${avatarIcon}"></i>
            </div>
            <div class="message-content">
                <div class="message-text">
                    ${this.formatMessageContent(content)}
                    ${sqlHtml}
                </div>
                <div class="message-time">${this.formatTime(new Date())}</div>
            </div>
        `;

        this.chatMessages.appendChild(messageElement);
        this.scrollToBottom();

        // Save to session storage for persistence
        this.saveChatHistory();
    }

    formatMessageContent(content) {
        // 配置marked选项
        marked.setOptions({
            highlight: function(code, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    try {
                        return hljs.highlight(code, { language: lang }).value;
                    } catch (err) {
                        console.warn(`Highlight.js error for language ${lang}:`, err);
                    }
                }
                return hljs.highlightAuto(code).value;
            },
            breaks: true, // 将换行符转换为<br>
            gfm: true,    // 启用GitHub风格的Markdown
            sanitize: false // 不清理HTML，因为我们信任后端内容
        });

        try {
            // 使用marked渲染Markdown
            return marked.parse(content);
        } catch (error) {
            console.error('Markdown parsing error:', error);
            // 如果Markdown解析失败，回退到基本HTML转义
            return this.escapeHtml(content).replace(/\n/g, '<br>');
        }
    }

    escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&")
            .replace(/</g, "<")
            .replace(/>/g, ">")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    formatTime(date) {
        return date.toLocaleTimeString('zh-CN', {
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    setLoadingState(loading) {
        if (loading) {
            this.sendButton.disabled = true;
            this.messageInput.disabled = true;
        } else {
            this.sendButton.disabled = false;
            this.messageInput.disabled = false;
            this.messageInput.focus();
        }
    }

    async clearChat() {
        if (!confirm('确定要清空所有聊天记录吗？')) {
            return;
        }

        try {
            const response = await fetch('/api/chat/clear', {
                method: 'POST'
            });

            const data = await response.json();

            if (data.success) {
                this.chatMessages.innerHTML = `
                    <div class="message bot-message">
                        <div class="message-avatar">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div class="message-content">
                            <div class="message-text">
                                <p>👋 欢迎使用 SQL Copilot！</p>
                                <p>我是一个智能SQL助手，可以将您的自然语言问题转换为SQL查询并返回结果。</p>
                                <p>请尝试询问关于数据库的问题，例如：</p>
                                <ul>
                                    <li>"查询所有用户的订单信息"</li>
                                    <li>"分析每个产品的销售情况"</li>
                                    <li>"找出销售额最高的品牌"</li>
                                </ul>
                            </div>
                            <div class="message-time">${this.formatTime(new Date())}</div>
                        </div>
                    </div>
                `;
                this.showToast('聊天记录已清空', 'success');
            } else {
                this.showToast('清空聊天记录失败', 'error');
            }
        } catch (error) {
            console.error('Error clearing chat:', error);
            this.showToast('网络错误，请重试', 'error');
        }
    }

    async initializeDatabase() {
        if (!confirm('确定要初始化数据库吗？这将加载数据库表结构信息。')) {
            return;
        }

        this.setLoadingState(true);

        try {
            const response = await fetch('/api/init', {
                method: 'POST'
            });

            const data = await response.json();

            if (data.success) {
                this.showToast('数据库初始化成功', 'success');
            } else {
                this.showToast(`初始化失败: ${data.error}`, 'error');
            }
        } catch (error) {
            console.error('Error initializing database:', error);
            this.showToast('网络错误，请重试', 'error');
        } finally {
            this.setLoadingState(false);
        }
    }

    async loadChatHistory() {
        try {
            const response = await fetch('/api/chat/history');
            const data = await response.json();

            if (data.success && data.history.length > 0) {
                // Clear welcome message
                this.chatMessages.innerHTML = '';

                // Load history messages
                data.history.forEach(msg => {
                    this.addMessage(msg.type, msg.content, msg.sql);
                });
            }
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }

    saveChatHistory() {
        // History is automatically saved on the server via session
        // This is just for client-side persistence if needed
    }

    showToast(message, type = 'info') {
        this.toast.textContent = message;
        this.toast.className = `toast ${type} show`;

        setTimeout(() => {
            this.toast.classList.remove('show');
        }, 3000);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SQLCopilotApp();
});

// Add some utility functions to global scope for easier debugging
window.SQLCopilot = {
    debug: {
        clearLocalStorage: () => {
            localStorage.clear();
            location.reload();
        },
        showSessionInfo: () => {
            fetch('/api/chat/history')
                .then(r => r.json())
                .then(console.log)
                .catch(console.error);
        }
    }
};