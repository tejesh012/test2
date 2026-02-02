"use client";

import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageCircle,
  Send,
  X,
  Bot,
  User,
  Smile,
  Frown,
  Meh,
  AlertCircle,
  Sparkles,
  Minimize2,
  Maximize2
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

interface EmotionalChatbotProps {
  currentEmotion?: string;
  onEmotionChange?: (emotion: string) => void;
  className?: string;
}

const emotionResponses: Record<string, string[]> = {
  happy: [
    "I can see you're in a great mood! That's wonderful! How can I make your day even better?",
    "Your happiness is contagious! What exciting things are you working on today?",
    "Love seeing that smile! What would you like to explore together?",
  ],
  sad: [
    "I notice you might be feeling a bit down. I'm here to listen if you want to talk.",
    "It's okay to have difficult moments. Would you like to chat about what's on your mind?",
    "I'm here for you. Sometimes talking helps - what's going on?",
  ],
  surprised: [
    "Something caught your attention! What's got you curious?",
    "I see that look of surprise! Discovered something interesting?",
    "Whoa, what's the exciting news?",
  ],
  angry: [
    "I sense some frustration. Take a deep breath - I'm here to help however I can.",
    "Things can be frustrating sometimes. Want to vent or would you like help solving something?",
    "I understand things might be tough right now. How can I assist you?",
  ],
  neutral: [
    "Hello! I'm Sentio, your emotional AI companion. How can I help you today?",
    "Hey there! What's on your mind?",
    "Welcome! I'm here to chat and assist. What would you like to talk about?",
  ],
};

const emotionIcons: Record<string, React.ReactNode> = {
  happy: <Smile className="w-4 h-4 text-green-400" />,
  sad: <Frown className="w-4 h-4 text-blue-400" />,
  neutral: <Meh className="w-4 h-4 text-slate-400" />,
  surprised: <AlertCircle className="w-4 h-4 text-yellow-400" />,
  angry: <AlertCircle className="w-4 h-4 text-red-400" />,
};


export function EmotionalChatbot({ currentEmotion = "neutral", className }: EmotionalChatbotProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
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

  useEffect(() => {
    if (isOpen && messages.length === 0) {
      // Send initial greeting based on emotion
      const greeting = emotionResponses[currentEmotion]?.[0] || emotionResponses.neutral[0];
      setMessages([
        {
          id: "1",
          role: "assistant",
          content: greeting,
          emotion: currentEmotion,
          timestamp: new Date(),
        },
      ]);
    }
  }, [isOpen, currentEmotion, messages.length]);

  const handleSend = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsTyping(true);

    try {
      // Format history for backend
      const history = messages.map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      const { data, error } = await chatService.sendMessage(inputValue, currentEmotion, history);

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
        // Fallback error message
        const errorMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: error || "I'm having trouble connecting to my brain right now. Please try again later.",
          emotion: currentEmotion,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errorMessage]);
      }
    } catch (err) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "I'm having trouble connecting to the server.",
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

  return (
    <>
      {/* Chat Toggle Button */}
      <AnimatePresence>
        {!isOpen && (
          <motion.button
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            onClick={() => setIsOpen(true)}
            className={cn(
              "fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full bg-gradient-to-r from-violet-600 to-indigo-600 text-white shadow-xl hover:shadow-2xl transition-shadow flex items-center justify-center",
              className
            )}
          >
            <MessageCircle className="w-6 h-6" />
            <span className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full animate-pulse" />
          </motion.button>
        )}
      </AnimatePresence>

      {/* Chat Window */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{
              opacity: 1,
              y: 0,
              scale: 1,
              height: isMinimized ? "auto" : "500px"
            }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            className={cn(
              "fixed bottom-6 right-6 z-50 w-96 bg-slate-900 rounded-2xl shadow-2xl border border-slate-800 overflow-hidden flex flex-col",
              className
            )}
          >
            {/* Header */}
            <div className="bg-gradient-to-r from-violet-600 to-indigo-600 p-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center">
                  <Bot className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="font-semibold text-white">Sentio AI</h3>
                  <div className="flex items-center gap-1 text-xs text-white/80">
                    {emotionIcons[currentEmotion]}
                    <span className="capitalize">{currentEmotion} mood detected</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setIsMinimized(!isMinimized)}
                  className="text-white/80 hover:text-white transition-colors"
                >
                  {isMinimized ? <Maximize2 className="w-5 h-5" /> : <Minimize2 className="w-5 h-5" />}
                </button>
                <button
                  onClick={() => setIsOpen(false)}
                  className="text-white/80 hover:text-white transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Messages */}
            {!isMinimized && (
              <>
                <div className="flex-1 overflow-y-auto p-4 space-y-4">
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
                          "max-w-[75%] rounded-2xl px-4 py-2",
                          message.role === "user"
                            ? "bg-violet-600 text-white rounded-tr-sm"
                            : "bg-slate-800 text-slate-200 rounded-tl-sm"
                        )}
                      >
                        <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                        <span className="text-xs opacity-50 mt-1 block">
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
                        <div className="flex gap-1">
                          <div className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                          <div className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                          <div className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                        </div>
                      </div>
                    </motion.div>
                  )}
                  <div ref={messagesEndRef} />
                </div>

                {/* Input */}
                <div className="p-4 border-t border-slate-800">
                  <div className="flex gap-2">
                    <Input
                      ref={inputRef}
                      value={inputValue}
                      onChange={(e) => setInputValue(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="Type your message..."
                      className="flex-1 bg-slate-800 border-slate-700 text-white placeholder:text-slate-500"
                    />
                    <Button
                      onClick={handleSend}
                      disabled={!inputValue.trim() || isTyping}
                      size="icon"
                    >
                      <Send className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

export default EmotionalChatbot;
