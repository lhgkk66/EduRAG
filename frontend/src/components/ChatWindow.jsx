import { useState, useCallback } from "react";
import { sendMessage } from "../api";
import MessageList from "./MessageList";
import ChatInput from "./ChatInput";

export default function ChatWindow() {
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSend = useCallback(
    async (text) => {
      const userMsg = { role: "user", content: text };
      setMessages((prev) => [...prev, userMsg]);
      setLoading(true);
      try {
        const data = await sendMessage(text, sessionId);
        setSessionId(data.session_id);
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: data.answer, sources: data.sources },
        ]);
      } catch {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: "抱歉，服务出错了，请稍后重试。", sources: [] },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [sessionId]
  );

  return (
    <div className="chat-window">
      <MessageList messages={messages} loading={loading} />
      <ChatInput onSend={handleSend} disabled={loading} />
    </div>
  );
}
