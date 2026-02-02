"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  Heart,
  Camera,
  BarChart3,
  User,
  LogOut,
  Home,
  Activity,
  History,
  Bot
} from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useAuth } from "@/contexts/AuthContext";
import { userService, type DashboardData } from "@/services/api";
import { ChatPanel } from "@/components/ui/chat-panel";
import CameraEmotionTFJS, {
  EMOTION_LABELS,
  EMOTION_COLORS,
  type EmotionLabel,
  type DetectionResultType
} from "@/components/CameraEmotionTFJS";

// Map new labels to icon paths
const emotionIconPaths: Record<string, string> = {
  happy: "/emotions/happy.svg",
  sad: "/emotions/sad.svg",
  neutral: "/emotions/neutral.svg",
  surprise: "/emotions/surprise.svg",
  angry: "/emotions/angry.svg",
  fear: "/emotions/fear.svg",
  disgust: "/emotions/disgust.svg",
};

// Gradient colors for emotion cards
const emotionGradients: Record<string, string> = {
  happy: "from-green-500/20 to-green-500/5 border-green-500/30",
  sad: "from-blue-500/20 to-blue-500/5 border-blue-500/30",
  neutral: "from-slate-500/20 to-slate-500/5 border-slate-500/30",
  surprise: "from-yellow-500/20 to-yellow-500/5 border-yellow-500/30",
  angry: "from-red-500/20 to-red-500/5 border-red-500/30",
  fear: "from-purple-500/20 to-purple-500/5 border-purple-500/30",
  disgust: "from-lime-500/20 to-lime-500/5 border-lime-500/30",
};

export default function DashboardPage() {
  const router = useRouter();
  const { user, isLoading: authLoading, isAuthenticated, logout } = useAuth();

  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [currentEmotion, setCurrentEmotion] = useState<EmotionLabel>("neutral");
  const [currentConfidence, setCurrentConfidence] = useState<number>(0);
  const [detectionResult, setDetectionResult] = useState<DetectionResultType | null>(null);

  // Fetch dashboard data
  useEffect(() => {
    const fetchData = async () => {
      if (isAuthenticated) {
        const { data } = await userService.getDashboard();
        if (data) {
          setDashboardData(data);
        }
      }
    };
    fetchData();
  }, [isAuthenticated]);

  // Redirect if not authenticated
  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push("/auth/login");
    }
  }, [authLoading, isAuthenticated, router]);

  // Handle emotion changes from camera detection
  const handleEmotionChange = (emotion: EmotionLabel, confidence: number) => {
    setCurrentEmotion(emotion);
    setCurrentConfidence(confidence);
  };

  // Handle detection results from camera detection
  const handleDetectionResult = (result: DetectionResultType | null) => {
    setDetectionResult(result);
  };

  // Get emotion-specific insight message
  const getEmotionInsight = (emotion: EmotionLabel): string => {
    const insights: Record<EmotionLabel, string> = {
      happy: "You're radiating positive energy! This is a great time for creative tasks and social interactions.",
      sad: "It's okay to feel down sometimes. Consider taking a short break or talking to someone you trust.",
      angry: "Take a deep breath. Physical activity or a brief walk can help release tension.",
      neutral: "You seem calm and focused. This is an ideal state for analytical tasks.",
      surprise: "Something caught your attention! Stay curious and explore new ideas.",
      fear: "Feeling anxious? Try grounding exercises - focus on 5 things you can see around you.",
      disgust: "Trust your instincts. If something doesn't feel right, it's okay to step back.",
    };
    return insights[emotion] || insights.neutral;
  };

  const handleLogout = async () => {
    await logout();
    router.push("/");
  };

  if (authLoading) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="w-12 h-12 border-4 border-violet-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-white select-none">
      {/* Sidebar */}
      <aside className="fixed left-0 top-0 h-full w-64 bg-slate-900/50 border-r border-slate-800 p-6 hidden lg:block">
        <Link href="/" className="flex items-center gap-2 mb-8">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center">
            <Heart className="w-6 h-6 text-white" />
          </div>
          <span className="text-xl font-bold gradient-text">Sentio</span>
        </Link>

        <nav className="space-y-2">
          <Link
            href="/dashboard"
            className="flex items-center gap-3 px-4 py-3 rounded-lg bg-violet-500/10 text-violet-400"
          >
            <BarChart3 className="w-5 h-5" />
            Dashboard
          </Link>
          <Link
            href="/"
            className="flex items-center gap-3 px-4 py-3 rounded-lg text-slate-400 hover:bg-slate-800/50 hover:text-white transition-colors"
          >
            <Home className="w-5 h-5" />
            Home
          </Link>
        </nav>

        <div className="absolute bottom-6 left-6 right-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-violet-500/20 to-cyan-500/20 flex items-center justify-center">
              <User className="w-5 h-5 text-violet-400" />
            </div>
            <div className="overflow-hidden">
              <p className="font-medium truncate">{user?.username || "User"}</p>
              <p className="text-xs text-slate-500 truncate">{user?.email}</p>
            </div>
          </div>
          <Button variant="ghost" onClick={handleLogout} className="w-full justify-start text-slate-400">
            <LogOut className="w-4 h-4 mr-2" />
            Sign Out
          </Button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="lg:pl-64 min-h-screen">
        {/* Header */}
        <header className="sticky top-0 z-40 glass border-b border-slate-800">
          <div className="px-6 py-4 flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">Dashboard</h1>
              <p className="text-slate-400 text-sm">Welcome back, {user?.first_name || user?.username}!</p>
            </div>
          </div>
        </header>

        <div className="p-6 space-y-6">
          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <Card className="bg-slate-900/50 border-slate-800">
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium text-slate-400">Total Detections</CardTitle>
                  <Activity className="w-4 h-4 text-violet-400" />
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-white">
                    {dashboardData?.stats.total_detections || 0}
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <Card className="bg-slate-900/50 border-slate-800">
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium text-slate-400">Face Detections</CardTitle>
                  <Camera className="w-4 h-4 text-cyan-400" />
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-white">
                    {dashboardData?.stats.face_detections || 0}
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              <Card className="bg-slate-900/50 border-slate-800">
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium text-slate-400">Emotion Analyses</CardTitle>
                  <img src="/emotions/happy.svg" alt="Emotion" className="w-4 h-4" />
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-white">
                    {dashboardData?.stats.emotion_detections || 0}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>

          {/* Main Panels: Video and Chat Side by Side */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Camera Feed */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <Card className="bg-slate-900/50 border-slate-800 h-[500px] flex flex-col">
                <CardHeader className="shrink-0">
                  <CardTitle className="text-white flex items-center gap-2">
                    <Camera className="w-5 h-5 text-violet-400" />
                    AI Emotion Detection
                  </CardTitle>
                  <CardDescription>
                    TensorFlow.js + MediaPipe FaceMesh
                  </CardDescription>
                </CardHeader>
                <CardContent className="flex-1 flex flex-col">
                  <CameraEmotionTFJS
                    onEmotionChange={handleEmotionChange}
                    onDetectionResult={handleDetectionResult}
                    className="flex-1 min-h-0"
                  />
                </CardContent>
              </Card>
            </motion.div>

            {/* AI Chatbot Panel */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              <ChatPanel
                currentEmotion={currentEmotion}
                className="h-[500px]"
                onEmotionChange={(emotion) => handleEmotionChange(emotion as EmotionLabel, 1.0)}
              />
            </motion.div>
          </div>

          {/* Detection Results Row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Emotion Results */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <Card className={`bg-gradient-to-br ${emotionGradients[currentEmotion] || emotionGradients.neutral} border-slate-800 h-full`}>
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <img
                      src={emotionIconPaths[currentEmotion] || emotionIconPaths.neutral}
                      alt={currentEmotion}
                      className="w-6 h-6"
                    />
                    Emotion Analysis
                  </CardTitle>
                  <CardDescription>
                    Real-time AI emotion detection results
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {detectionResult ? (
                    <>
                      <div className="flex items-center gap-4">
                        <div
                          className="w-16 h-16 rounded-full flex items-center justify-center"
                          style={{ backgroundColor: `${EMOTION_COLORS[currentEmotion]}20` }}
                        >
                          <img
                            src={emotionIconPaths[currentEmotion]}
                            alt={currentEmotion}
                            className="w-10 h-10"
                          />
                        </div>
                        <div>
                          <h3
                            className="text-2xl font-bold capitalize"
                            style={{ color: EMOTION_COLORS[currentEmotion] }}
                          >
                            {currentEmotion}
                          </h3>
                          <p className="text-slate-400">
                            Confidence: {(currentConfidence * 100).toFixed(0)}%
                          </p>
                        </div>
                      </div>
                      {/* Confidence bar */}
                      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${currentConfidence * 100}%`,
                            backgroundColor: EMOTION_COLORS[currentEmotion],
                          }}
                        />
                      </div>
                      <div className="bg-slate-800/50 rounded-xl p-4">
                        <h4 className="font-semibold mb-2 flex items-center gap-2">
                          <Bot className="w-4 h-4 text-violet-400" />
                          AI Insight
                        </h4>
                        <p className="text-slate-300 text-sm">
                          {getEmotionInsight(currentEmotion)}
                        </p>
                      </div>
                    </>
                  ) : (
                    <div className="text-center py-8">
                      <img
                        src="/emotions/neutral.svg"
                        alt="Waiting"
                        className="w-12 h-12 mx-auto mb-4 opacity-50"
                      />
                      <p className="text-slate-500">
                        Start the camera to see emotion detection results
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </motion.div>

          {/* Recent Detections */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 }}
            >
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <History className="w-5 h-5 text-violet-400" />
                  Recent Detections
                </CardTitle>
                <CardDescription>Your latest face detection history</CardDescription>
              </CardHeader>
              <CardContent>
                {dashboardData?.recent_detections && dashboardData.recent_detections.length > 0 ? (
                  <div className="space-y-3">
                    {dashboardData.recent_detections.slice(0, 5).map((detection) => (
                      <div
                        key={detection.id}
                        className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg"
                      >
                        <div className="flex items-center gap-3">
                          <img
                            src={emotionIconPaths[detection.result_data?.emotions?.[0] || "neutral"] || emotionIconPaths.neutral}
                            alt={detection.result_data?.emotions?.[0] || "neutral"}
                            className="w-6 h-6"
                          />
                          <div>
                            <p className="font-medium capitalize">
                              {detection.result_data?.emotions?.[0] || "Unknown"}
                            </p>
                            <p className="text-xs text-slate-500">
                              {new Date(detection.created_at).toLocaleString()}
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-sm text-slate-400">
                            {detection.result_data?.faces_count || 0} face(s)
                          </p>
                          <p className="text-xs text-slate-500">
                            {(detection.confidence * 100).toFixed(0)}% confidence
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-slate-500">
                    No detection history yet. Start the camera to begin!
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
          </div>
        </div>
      </main>
    </div>
  );
}
