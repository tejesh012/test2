"use client";

import React from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  Camera,
  ArrowRight,
  Eye,
  Heart,
  Bot,
  Shield,
  Zap,
  Brain,
  ChevronDown
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SparklesCore } from "@/components/ui/sparkles";
import { InteractiveRobotSpline } from "@/components/blocks/interactive-3d-robot";
import { EmotionalChatbot } from "@/components/ui/emotional-chatbot";
import { useAuth } from "@/contexts/AuthContext";

const ROBOT_SCENE_URL = "https://prod.spline.design/PyzDhpQ9E5f1E3MT/scene.splinecode";

const features = [
  {
    icon: Eye,
    title: "Real-time Face Detection",
    description: "Advanced ML-powered face detection that works instantly with your webcam.",
  },
  {
    icon: Brain,
    title: "Emotion Analysis",
    description: "Understand emotions through facial expressions using our trained models.",
  },
  {
    icon: Bot,
    title: "Emotional AI Chatbot",
    description: "Chat with Sentio AI that responds based on your detected mood and feelings.",
  },
  {
    icon: Shield,
    title: "Privacy First",
    description: "All processing happens securely. Your data stays protected.",
  },
  {
    icon: Zap,
    title: "Lightning Fast",
    description: "Optimized for performance with real-time processing capabilities.",
  },
  {
    icon: Heart,
    title: "Empathetic Support",
    description: "Get personalized emotional support and contextual assistance.",
  },
];

export default function HomePage() {
  const { isAuthenticated } = useAuth();

  const scrollToContent = () => {
    document.getElementById("content")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="bg-black text-white">
      {/* Full Screen Hero - Sentio Only */}
      <section className="h-screen relative w-full flex flex-col items-center justify-center overflow-hidden">
        {/* Full page sparkles background */}
        <div className="w-full absolute inset-0 h-screen">
          <SparklesCore
            id="tsparticlesfullpage"
            background="transparent"
            minSize={0.6}
            maxSize={1.4}
            particleDensity={100}
            className="w-full h-full"
            particleColor="#FFFFFF"
            speed={1}
          />
        </div>

        {/* Brand Name with Gradients - Behind Robot */}
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center pointer-events-none">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, ease: "easeOut" }}
            className="text-center"
          >
            <h1 className="md:text-8xl text-5xl lg:text-[12rem] font-bold text-center bg-clip-text text-transparent bg-gradient-to-b from-white via-white to-violet-400 tracking-tight">
              Sentio
            </h1>

            {/* Gradient lines under title */}
            <div className="w-full max-w-[40rem] h-20 relative mt-4 mx-auto">
              <div className="absolute inset-x-20 top-0 bg-gradient-to-r from-transparent via-violet-500 to-transparent h-[2px] w-3/4 blur-sm" />
              <div className="absolute inset-x-20 top-0 bg-gradient-to-r from-transparent via-violet-500 to-transparent h-px w-3/4" />
              <div className="absolute inset-x-60 top-0 bg-gradient-to-r from-transparent via-indigo-500 to-transparent h-[5px] w-1/4 blur-sm" />
              <div className="absolute inset-x-60 top-0 bg-gradient-to-r from-transparent via-indigo-500 to-transparent h-px w-1/4" />
            </div>
          </motion.div>
        </div>

        {/* 3D Robot - Main Attraction (In Front) */}
        <div className="absolute inset-0 z-20">
          <InteractiveRobotSpline
            scene={ROBOT_SCENE_URL}
            className="w-full h-full"
          />
        </div>

        {/* Buttons and Tagline - On Top */}
        <div className="relative z-30 flex flex-col items-center justify-center mt-48">
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.8 }}
            className="text-xl md:text-2xl text-neutral-400 text-center max-w-lg px-4"
          >
            Your Emotional AI Companion
          </motion.p>

          {/* Auth Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8, duration: 0.6 }}
            className="flex gap-4 mt-8"
          >
            {isAuthenticated ? (
              <Link href="/dashboard">
                <Button size="lg" className="bg-violet-600 hover:bg-violet-700 text-white px-8">
                  Dashboard
                  <ArrowRight className="w-5 h-5 ml-2" />
                </Button>
              </Link>
            ) : (
              <>
                <Link href="/auth/login">
                  <Button size="lg" variant="outline" className="border-white/20 hover:bg-white/10 text-white px-8">
                    Login
                  </Button>
                </Link>
                <Link href="/auth/register">
                  <Button size="lg" className="bg-violet-600 hover:bg-violet-700 text-white px-8">
                    Register
                    <ArrowRight className="w-5 h-5 ml-2" />
                  </Button>
                </Link>
              </>
            )}
          </motion.div>
        </div>

        {/* Scroll Indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5, duration: 0.8 }}
          className="absolute bottom-10 left-1/2 -translate-x-1/2 z-30 flex flex-col items-center cursor-pointer"
          onClick={scrollToContent}
        >
          <span className="text-sm text-neutral-400 mb-2">Scroll to explore</span>
          <motion.div
            animate={{ y: [0, 8, 0] }}
            transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
          >
            <ChevronDown className="w-6 h-6 text-violet-400" />
          </motion.div>
        </motion.div>
      </section>

      {/* Main Content - After Scroll */}
      <div id="content" className="bg-slate-950">
        {/* Navbar - Appears after scroll */}
        <nav className="sticky top-0 z-50 bg-slate-950/80 backdrop-blur-xl border-b border-slate-800">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <Link href="/" className="flex items-center gap-2">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center">
                  <Heart className="w-6 h-6 text-white" />
                </div>
                <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-violet-400 to-indigo-400">Sentio</span>
              </Link>

              <div className="hidden md:flex items-center gap-8">
                <Link href="#features" className="text-slate-300 hover:text-white transition-colors">
                  Features
                </Link>
                <Link href="#demo" className="text-slate-300 hover:text-white transition-colors">
                  Demo
                </Link>
                <Link href="#about" className="text-slate-300 hover:text-white transition-colors">
                  About
                </Link>
              </div>

              <div className="flex items-center gap-4">
                {isAuthenticated ? (
                  <Link href="/dashboard">
                    <Button>
                      Dashboard
                      <ArrowRight className="w-4 h-4 ml-2" />
                    </Button>
                  </Link>
                ) : (
                  <>
                    <Link href="/auth/login">
                      <Button variant="ghost">Login</Button>
                    </Link>
                    <Link href="/auth/register">
                      <Button>
                        Register
                        <ArrowRight className="w-4 h-4 ml-2" />
                      </Button>
                    </Link>
                  </>
                )}
              </div>
            </div>
          </div>
        </nav>

        {/* Features Section */}
        <section id="features" className="py-24 relative">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="text-center mb-16"
            >
              <h2 className="text-4xl md:text-5xl font-bold mb-4">
                Why Choose <span className="bg-clip-text text-transparent bg-gradient-to-r from-violet-400 to-indigo-400">Sentio</span>?
              </h2>
              <p className="text-xl text-slate-400 max-w-2xl mx-auto">
                The most advanced emotional AI platform for understanding and supporting your wellbeing
              </p>
            </motion.div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {features.map((feature, index) => (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <Card className="h-full bg-slate-900/50 border-slate-800 hover:border-violet-500/50 transition-all duration-300 hover:shadow-xl hover:shadow-violet-500/10">
                    <CardHeader>
                      <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500/20 to-indigo-500/20 flex items-center justify-center mb-4">
                        <feature.icon className="w-6 h-6 text-violet-400" />
                      </div>
                      <CardTitle className="text-white">{feature.title}</CardTitle>
                      <CardDescription className="text-slate-400">
                        {feature.description}
                      </CardDescription>
                    </CardHeader>
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Demo Section */}
        <section id="demo" className="py-24 relative bg-slate-900/50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="text-center mb-16"
            >
              <h2 className="text-4xl md:text-5xl font-bold mb-4">
                See Sentio In <span className="bg-clip-text text-transparent bg-gradient-to-r from-violet-400 to-indigo-400">Action</span>
              </h2>
              <p className="text-xl text-slate-400 max-w-2xl mx-auto">
                Experience emotional AI that adapts to how you feel
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              viewport={{ once: true }}
            >
              <Card className="bg-slate-900/50 border-slate-800 overflow-hidden">
                <CardContent className="p-8">
                  <div className="grid md:grid-cols-2 gap-8 items-center">
                    <div className="aspect-video bg-slate-800 rounded-xl flex items-center justify-center relative overflow-hidden">
                      <div className="absolute inset-0 bg-gradient-to-br from-violet-500/10 to-indigo-500/10" />
                      <div className="text-center relative z-10">
                        <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-gradient-to-br from-violet-500/30 to-indigo-500/30 flex items-center justify-center animate-pulse">
                          <Bot className="w-10 h-10 text-violet-400" />
                        </div>
                        <p className="text-slate-400 mb-4">Chat with Sentio AI</p>
                        <p className="text-sm text-slate-500">Click the chat button to start</p>
                      </div>
                    </div>
                    <div className="space-y-6">
                      <h3 className="text-2xl font-bold">How Sentio Works</h3>
                      <ul className="space-y-4">
                        <li className="flex gap-4">
                          <div className="w-8 h-8 rounded-full bg-violet-500/20 flex items-center justify-center shrink-0">
                            <span className="text-violet-400 font-bold">1</span>
                          </div>
                          <div>
                            <h4 className="font-semibold">Enable Camera</h4>
                            <p className="text-slate-400 text-sm">Grant camera access for emotion detection</p>
                          </div>
                        </li>
                        <li className="flex gap-4">
                          <div className="w-8 h-8 rounded-full bg-violet-500/20 flex items-center justify-center shrink-0">
                            <span className="text-violet-400 font-bold">2</span>
                          </div>
                          <div>
                            <h4 className="font-semibold">Emotion Detection</h4>
                            <p className="text-slate-400 text-sm">Sentio analyzes your facial expressions</p>
                          </div>
                        </li>
                        <li className="flex gap-4">
                          <div className="w-8 h-8 rounded-full bg-violet-500/20 flex items-center justify-center shrink-0">
                            <span className="text-violet-400 font-bold">3</span>
                          </div>
                          <div>
                            <h4 className="font-semibold">Empathetic Chat</h4>
                            <p className="text-slate-400 text-sm">Have meaningful conversations that understand your mood</p>
                          </div>
                        </li>
                        <li className="flex gap-4">
                          <div className="w-8 h-8 rounded-full bg-violet-500/20 flex items-center justify-center shrink-0">
                            <span className="text-violet-400 font-bold">4</span>
                          </div>
                          <div>
                            <h4 className="font-semibold">Personalized Support</h4>
                            <p className="text-slate-400 text-sm">Receive tailored responses based on your emotional state</p>
                          </div>
                        </li>
                      </ul>
                      <Link href={isAuthenticated ? "/dashboard" : "/auth/register"}>
                        <Button className="w-full">
                          Try Full Experience
                          <ArrowRight className="w-4 h-4 ml-2" />
                        </Button>
                      </Link>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        </section>

        {/* CTA Section */}
        <section id="about" className="py-24 relative">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
            >
              <h2 className="text-4xl md:text-5xl font-bold mb-6">
                Ready to Meet <span className="bg-clip-text text-transparent bg-gradient-to-r from-violet-400 to-indigo-400">Sentio</span>?
              </h2>
              <p className="text-xl text-slate-400 mb-10">
                Join thousands of users who are already experiencing emotional AI that truly understands.
                Start your journey to better emotional awareness today.
              </p>
              <Link href={isAuthenticated ? "/dashboard" : "/auth/register"}>
                <Button size="lg" className="bg-violet-600 hover:bg-violet-700">
                  {isAuthenticated ? "Go to Dashboard" : "Create Account"}
                  <ArrowRight className="w-5 h-5 ml-2" />
                </Button>
              </Link>
            </motion.div>
          </div>
        </section>

        {/* Footer */}
        <footer className="py-12 border-t border-slate-800">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex flex-col md:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center">
                  <Heart className="w-5 h-5 text-white" />
                </div>
                <span className="font-bold">Sentio</span>
              </div>
              <p className="text-slate-500 text-sm">
                Emotional AI that understands you. Built with Next.js, Flask, and OpenCV.
              </p>
            </div>
          </div>
        </footer>
      </div>

      {/* Emotional Chatbot */}
      <EmotionalChatbot currentEmotion="neutral" />
    </div>
  );
}
