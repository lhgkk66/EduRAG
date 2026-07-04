import { useEffect, useRef } from "react";
import MessageItem from "./MessageItem";

export default function MessageList({ messages, loading }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  return (
    <div className="message-list">
      {messages.length === 0 && !loading && (
        <div className="empty-hint">
          <div className="hint-icon">💡</div>
          <p>在下方输入你的问题</p>
          <p>我会从知识库中检索并回答</p>
        </div>
      )}
      {messages.map((msg, i) => (
        <MessageItem key={i} message={msg} />
      ))}
      {loading && (
        <div className="message message-assistant">
          <div className="message-avatar">🤖</div>
          <div className="message-body">
            <div className="typing-indicator">
              <span></span><span></span><span></span>
            </div>
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
