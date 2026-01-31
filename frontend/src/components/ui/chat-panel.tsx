"use client";

import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Bot,
  User,
  Smile,
  Frown,
  Meh,
  AlertCircle,
  Sparkles,
  Loader2,
  MessageSquare
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { chatService } from "@/services/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  emotion?: string;
  timestamp: Date;
}

interface ChatPanelProps {
  currentEmotion?: string;
  className?: string;
}

const emotionIcons: Record<string, React.ReactNode> = {
  happy: <Smile className="w-4 h-4 text-green-400" />,
  sad: <Frown className="w-4 h-4 text-blue-400" />,
  neutral: <Meh className="w-4 h-4 text-slate-400" />,
  surprised: <AlertCircle className="w-4 h-4 text-yellow-400" />,
  angry: <AlertCircle className="w-4 h-4 text-red-400" />,
};

const SUGGESTIONS = [
  "How do I login?",
  "How does emotion detection work?",
  "What features are available?",
  "Is my data private?"
];

export function ChatPanel({ currentEmotion = "neutral", className }: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Send initial greeting
  useEffect(() => {
    if (messages.length === 0) {
      const greeting: Message = {
        id: "1",
        role: "assistant",
        content: `Hello! I'm Sentio, your AI assistant. 👋

I can help you with:
• **Login & Registration** - Account help
• **Camera & Detection** - Face and emotion detection
• **Dashboard Features** - Using Sentio
• **Privacy** - How we protect your data

What would you like to know?`,
        emotion: currentEmotion,
        timestamp: new Date(),
      };
      setMessages([greeting]);
    }
  }, []);

  const handleSend = async (messageText?: string) => {
    const text = messageText || inputValue.trim();
    if (!text) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: text,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsTyping(true);

    try {
      const { data, error } = await chatService.sendMessage(text, currentEmotion);

      if (data && data.success) {
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: data.response,
          emotion: currentEmotion,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } else {
        // Fallback response if API fails
        const fallbackMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: error || "I apologize, but I'm having trouble connecting. Please try again or check if the backend server is running.",
          emotion: currentEmotion,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, fallbackMessage]);
      }
    } catch (err) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "I'm having trouble connecting to the server. Please make sure the backend is running on port 5000.",
        emotion: currentEmotion,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    }

    setIsTyping(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    handleSend(suggestion);
  };

  return (
    <div className={cn("flex flex-col h-full bg-slate-900/50 rounded-xl border border-slate-800", className)}>
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-600 to-indigo-600 p-4 rounded-t-xl flex items-center gap-3">
        <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center">
          <Bot className="w-6 h-6 text-white" />
        </div>
        <div>
          <h3 className="font-semibold text-white">Sentio AI</h3>
          <div className="flex items-center gap-1 text-xs text-white/80">
            {emotionIcons[currentEmotion]}
            <span className="capitalize">{currentEmotion} mood</span>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
        {messages.map((message) => (
          <motion.div
            key={message.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={cn(
              "flex gap-3",
              message.role === "user" ? "flex-row-reverse" : ""
            )}
          >
            <div
              className={cn(
                "w-8 h-8 rounded-full flex items-center justify-center shrink-0",
                message.role === "user"
                  ? "bg-violet-500/20"
                  : "bg-gradient-to-br from-violet-500/20 to-cyan-500/20"
              )}
            >
              {message.role === "user" ? (
                <User className="w-4 h-4 text-violet-400" />
              ) : (
                <Sparkles className="w-4 h-4 text-violet-400" />
              )}
            </div>
            <div
              className={cn(
                "max-w-[85%] rounded-2xl px-4 py-3",
                message.role === "user"
                  ? "bg-violet-600 text-white rounded-tr-sm"
                  : "bg-slate-800 text-slate-200 rounded-tl-sm"
              )}
            >
              <div className="text-sm whitespace-pre-wrap prose prose-invert prose-sm max-w-none">
                {message.content.split('\n').map((line, i) => (
                  <React.Fragment key={i}>
                    {line.includes('**') ? (
                      <span dangerouslySetInnerHTML={{
                        __html: line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                      }} />
                    ) : line.startsWith('•') || line.startsWith('-') ? (
                      <span className="block ml-2">{line}</span>
                    ) : (
                      line
                    )}
                    {i < message.content.split('\n').length - 1 && <br />}
                  </React.Fragment>
                ))}
              </div>
              <span className="text-xs opacity-50 mt-2 block">
                {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
          </motion.div>
        ))}

        {isTyping && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex gap-3"
          >
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500/20 to-cyan-500/20 flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-violet-400" />
            </div>
            <div className="bg-slate-800 rounded-2xl rounded-tl-sm px-4 py-3">
              <div className="flex gap-1 items-center">
                <Loader2 className="w-4 h-4 text-violet-400 animate-spin" />
                <span className="text-sm text-slate-400 ml-2">Thinking...</span>
              </div>
            </div>
          </motion.div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Suggestions */}
      {messages.length <= 1 && (
        <div className="px-4 pb-2">
          <p className="text-xs text-slate-500 mb-2">Suggested questions:</p>
          <div className="flex flex-wrap gap-2">
            {SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                onClick={() => handleSuggestionClick(suggestion)}
                className="text-xs px-3 py-1.5 rounded-full bg-slate-800 text-slate-300 hover:bg-slate-700 transition-colors"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div className="p-4 border-t border-slate-800">
        <div className="flex gap-2">
          <Input
            ref={inputRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me anything..."
            className="flex-1 bg-slate-800 border-slate-700 text-white placeholder:text-slate-500 select-text"
          />
          <Button
            onClick={() => handleSend()}
            disabled={!inputValue.trim() || isTyping}
            size="icon"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}

export default ChatPanel;
