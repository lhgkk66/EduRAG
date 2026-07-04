import ChatWindow from "./components/ChatWindow";

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>🎓 EduRAG</h1>
        <span className="app-subtitle">教育知识助手</span>
      </header>
      <main className="app-main">
        <ChatWindow />
      </main>
    </div>
  );
}
