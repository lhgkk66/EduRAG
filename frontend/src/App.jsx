import { useState, useCallback, useEffect, useRef } from "react";
import ChatWindow from "./components/ChatWindow";
import Sidebar from "./components/Sidebar";

const STORAGE_KEY = "edurag_sessions";
const THEME_KEY = "edurag_theme";

function loadSessions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveSessions(sessions) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
}

let idCounter = Date.now();

export default function App() {
  const [sessions, setSessions] = useState(loadSessions);
  const [activeId, setActiveId] = useState(null);
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem(THEME_KEY);
    return saved || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  });

  // 同步 theme 到 DOM
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((t) => (t === "dark" ? "light" : "dark"));
  }, []);

  // 默认选中第一个会话
  useEffect(() => {
    if (!activeId && sessions.length > 0) {
      setActiveId(sessions[0].id);
    }
  }, [sessions, activeId]);

  const activeSession = sessions.find((s) => s.id === activeId) || null;

  const updateSession = useCallback((id, updater) => {
    setSessions((prev) => {
      const next = prev.map((s) => (s.id === id ? updater(s) : s));
      saveSessions(next);
      return next;
    });
  }, []);

  const handleNewChat = useCallback(() => {
    const id = String(++idCounter);
    const session = { id, title: "新对话", messages: [], createdAt: Date.now() };
    setSessions((prev) => {
      const next = [session, ...prev];
      saveSessions(next);
      return next;
    });
    setActiveId(id);
  }, []);

  const handleDelete = useCallback(
    (id) => {
      setSessions((prev) => {
        const next = prev.filter((s) => s.id !== id);
        saveSessions(next);
        if (activeId === id) {
          setTimeout(() => setActiveId(next[0]?.id || null), 0);
        }
        return next;
      });
    },
    [activeId]
  );

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-left">
          <h1>🎓 EduRAG</h1>
          <span className="app-subtitle">教育知识助手</span>
        </div>
        <button className="theme-toggle" onClick={toggleTheme} title="切换主题">
          {theme === "dark" ? "☀️" : "🌙"}
        </button>
      </header>
      <div className="app-body">
        <Sidebar
          sessions={sessions}
          activeId={activeId}
          onSelect={setActiveId}
          onNew={handleNewChat}
          onDelete={handleDelete}
        />
        <main className="app-main">
          {activeSession ? (
            <ChatWindow
              key={activeSession.id}
              session={activeSession}
              onUpdate={(updater) => updateSession(activeSession.id, updater)}
            />
          ) : (
            <div className="empty-state">
              <div className="logo">🎓</div>
              <p>欢迎使用 EduRAG 教育知识助手</p>
              <button className="btn-new-chat" onClick={handleNewChat}>
                + 开始新对话
              </button>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
