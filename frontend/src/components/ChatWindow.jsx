import { useState, useCallback, useEffect, useRef } from "react";
import { sendMessage } from "../api";
import MessageList from "./MessageList";
import ChatInput from "./ChatInput";

export default function ChatWindow({ session, onUpdate }) {
  const [loading, setLoading] = useState(false);
  const sessionIdRef = useRef(session.id);

  // 首次消息发送后，用问题内容更新会话标题
  const handleSend = useCallback(
    async (text) => {
      const userMsg = { role: "user", content: text };
      const isFirstMsg = session.messages.length === 0;
      onUpdate((s) => ({
        ...s,
        messages: [...s.messages, userMsg],
        ...(isFirstMsg ? { title: text.slice(0, 30) } : {}),
      }));
      setLoading(true);
      try {
        const data = await sendMessage(text, sessionIdRef.current);
        onUpdate((s) => ({
          ...s,
          messages: [
            ...s.messages,
            { role: "assistant", content: data.answer, sources: data.sources },
          ],
        }));
      } catch {
        onUpdate((s) => ({
          ...s,
          messages: [
            ...s.messages,
            { role: "assistant", content: "抱歉，服务出错了，请稍后重试。", sources: [] },
          ],
        }));
      } finally {
        setLoading(false);
      }
    },
    [session.messages.length, onUpdate]
  );

  return (
    <div className="chat-window">
      <MessageList messages={session.messages} loading={loading} />
      <ChatInput onSend={handleSend} disabled={loading} />
    </div>
  );
}
