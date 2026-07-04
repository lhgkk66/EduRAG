export default function Sidebar({ sessions, activeId, onSelect, onNew, onDelete }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <button className="btn-new-chat" onClick={onNew}>
          + 新对话
        </button>
      </div>
      <div className="session-list">
        {sessions.length === 0 && (
          <p className="session-empty">暂无对话记录</p>
        )}
        {sessions.map((s) => (
          <div
            key={s.id}
            className={`session-item${s.id === activeId ? " active" : ""}`}
            onClick={() => onSelect(s.id)}
          >
            <span className="session-icon">💬</span>
            <span className="session-title">{s.title}</span>
            <button
              className="session-delete"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(s.id);
              }}
              title="删除对话"
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}
