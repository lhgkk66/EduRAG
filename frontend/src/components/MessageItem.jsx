export default function MessageItem({ message }) {
  const isUser = message.role === "user";

  return (
    <div className={`message ${isUser ? "message-user" : "message-assistant"}`}>
      <div className="message-avatar">{isUser ? "👤" : "🤖"}</div>
      <div className="message-body">
        <div className="message-content">{message.content}</div>
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="message-sources">
            <details>
              <summary>📚 参考来源 ({message.sources.length})</summary>
              {message.sources.map((s, i) => (
                <div key={i} className="source-item">
                  <span className="source-name">{s.source}</span>
                  <span className="source-score">{(s.score * 100).toFixed(0)}%</span>
                  <p className="source-text">{s.text}</p>
                </div>
              ))}
            </details>
          </div>
        )}
      </div>
    </div>
  );
}
