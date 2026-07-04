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
          <p>👋 欢迎使用 EduRAG 教育知识助手</p>
          <p>在上方输入你的问题，我会从知识库中检索并回答</p>
        </div>
      )}
      {messages.map((msg, i) => (
        <MessageItem key={i} message={msg} />
      ))}
      {loading && (
        <div className="message message-assistant">
          <div className="message-role">🤖 EduRAG</div>
          <div className="message-content typing-indicator">
            <span></span><span></span><span></span>
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
