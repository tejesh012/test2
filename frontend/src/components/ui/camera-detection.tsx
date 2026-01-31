"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Camera,
  CameraOff,
  AlertCircle,
  RefreshCw,
  Smile,
  Frown,
  Meh,
  Video,
  VideoOff
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { detectionService, type DetectionResult } from "@/services/api";

interface CameraDetectionProps {
  onEmotionChange?: (emotion: string) => void;
  onDetectionResult?: (result: DetectionResult | null) => void;
  className?: string;
}

const emotionIcons: Record<string, React.ReactNode> = {
  happy: <Smile className="w-5 h-5 text-green-400" />,
  sad: <Frown className="w-5 h-5 text-blue-400" />,
  neutral: <Meh className="w-5 h-5 text-slate-400" />,
  surprised: <AlertCircle className="w-5 h-5 text-yellow-400" />,
  angry: <AlertCircle className="w-5 h-5 text-red-400" />,
};

export function CameraDetection({
  onEmotionChange,
  onDetectionResult,
  className
}: CameraDetectionProps) {
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const analysisIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // State
  const [cameraState, setCameraState] = useState<"idle" | "requesting" | "active" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentDetection, setCurrentDetection] = useState<DetectionResult | null>(null);
  const [videoReady, setVideoReady] = useState(false);

  // Get current emotion
  const currentEmotion = currentDetection?.analysis?.[0]?.emotion || "neutral";

  // Cleanup function - force stop all tracks
  const cleanup = useCallback(() => {
    console.log("[Camera] Cleaning up...");

    // Stop analysis interval
    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current);
      analysisIntervalRef.current = null;
    }

    // Stop all tracks from ref
    if (streamRef.current) {
      const tracks = streamRef.current.getTracks();
      tracks.forEach(track => {
        track.stop();
        track.enabled = false;
        console.log("[Camera] Stopped track:", track.kind, track.readyState);
      });
      streamRef.current = null;
    }

    // Also stop tracks from video element directly
    if (videoRef.current && videoRef.current.srcObject) {
      const mediaStream = videoRef.current.srcObject as MediaStream;
      if (mediaStream && mediaStream.getTracks) {
        mediaStream.getTracks().forEach(track => {
          track.stop();
          track.enabled = false;
          console.log("[Camera] Stopped video element track:", track.kind);
        });
      }
      videoRef.current.srcObject = null;
      videoRef.current.load(); // Force reload to release camera
    }

    setVideoReady(false);
    setCurrentDetection(null);
    setIsAnalyzing(false);
  }, []);

  // Start camera
  const startCamera = async () => {
    console.log("[Camera] ===== START CAMERA =====");

    // Cleanup any existing stream
    cleanup();

    setCameraState("requesting");
    setErrorMessage(null);

    // Check if mediaDevices is available
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      console.error("[Camera] mediaDevices not available");
      setErrorMessage("Camera API not available. Make sure you're on HTTPS or localhost.");
      setCameraState("error");
      return;
    }

    console.log("[Camera] Requesting camera permission...");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 }
        },
        audio: false
      });

      console.log("[Camera] Permission GRANTED");
      console.log("[Camera] Stream tracks:", stream.getTracks().map(t => t.kind));

      streamRef.current = stream;

      // Set video source
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        console.log("[Camera] srcObject set on video element");

        // Force play
        try {
          await videoRef.current.play();
          console.log("[Camera] Video playing");
        } catch (playError) {
          console.error("[Camera] Play error:", playError);
        }
      } else {
        console.error("[Camera] videoRef.current is null!");
      }

      setCameraState("active");

    } catch (error: unknown) {
      console.error("[Camera] Error:", error);

      let message = "Failed to access camera.";
      if (error instanceof Error) {
        if (error.name === "NotAllowedError") {
          message = "Camera permission denied. Please allow camera access in your browser settings.";
        } else if (error.name === "NotFoundError") {
          message = "No camera found. Please connect a camera.";
        } else if (error.name === "NotReadableError") {
          message = "Camera is in use by another application.";
        } else {
          message = error.message;
        }
      }

      setErrorMessage(message);
      setCameraState("error");
    }
  };

  // Stop camera
  const stopCamera = () => {
    console.log("[Camera] ===== STOP CAMERA =====");
    cleanup();
    setCameraState("idle");
  };

  // Handle video metadata loaded
  const handleVideoLoadedMetadata = () => {
    console.log("[Camera] Video metadata loaded");
    if (videoRef.current) {
      console.log("[Camera] Video dimensions:", videoRef.current.videoWidth, "x", videoRef.current.videoHeight);
    }
  };

  // Handle video can play
  const handleVideoCanPlay = () => {
    console.log("[Camera] Video can play");
    setVideoReady(true);
  };

  // Handle video playing
  const handleVideoPlaying = () => {
    console.log("[Camera] Video is now playing");
    setVideoReady(true);
  };

  // Capture and analyze frame
  const captureAndAnalyze = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) {
      console.log("[Analysis] Missing refs");
      return;
    }

    const video = videoRef.current;

    // Check video is ready
    if (video.readyState < 2) {
      console.log("[Analysis] Video not ready, readyState:", video.readyState);
      return;
    }

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      console.log("[Analysis] Video dimensions not ready");
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      console.log("[Analysis] No canvas context");
      return;
    }

    // Set canvas size
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw frame
    ctx.drawImage(video, 0, 0);

    // Convert to base64
    const imageData = canvas.toDataURL("image/jpeg", 0.8);

    console.log("[Analysis] Sending frame for analysis...");
    setIsAnalyzing(true);

    try {
      const { data, error } = await detectionService.analyzeFrame(imageData, false);

      if (data) {
        console.log("[Analysis] Detection result:", {
          faces: data.faces_detected,
          emotion: data.analysis?.[0]?.emotion
        });
        setCurrentDetection(data);
        onDetectionResult?.(data);

        const emotion = data.analysis?.[0]?.emotion || "neutral";
        onEmotionChange?.(emotion);
      } else {
        console.error("[Analysis] Error:", error);
      }
    } catch (err) {
      console.error("[Analysis] Exception:", err);
    }

    setIsAnalyzing(false);
  }, [onEmotionChange, onDetectionResult]);

  // Start analysis interval when video is ready
  useEffect(() => {
    if (cameraState === "active" && videoReady) {
      console.log("[Camera] Starting analysis interval");

      // Run first analysis immediately
      captureAndAnalyze();

      // Then every 2 seconds
      analysisIntervalRef.current = setInterval(captureAndAnalyze, 2000);
    }

    return () => {
      if (analysisIntervalRef.current) {
        clearInterval(analysisIntervalRef.current);
        analysisIntervalRef.current = null;
      }
    };
  }, [cameraState, videoReady, captureAndAnalyze]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  // Also cleanup when page is closed/refreshed
  useEffect(() => {
    const handleBeforeUnload = () => {
      cleanup();
    };

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, [cleanup]);

  return (
    <div className={`relative bg-slate-800 rounded-xl overflow-hidden select-none ${className || ""}`}>
      {/* Video element - ALWAYS rendered */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        onLoadedMetadata={handleVideoLoadedMetadata}
        onCanPlay={handleVideoCanPlay}
        onPlaying={handleVideoPlaying}
        className={`w-full h-full object-cover ${cameraState === "active" ? "block" : "hidden"}`}
        style={{ transform: "scaleX(-1)" }}
      />

      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Idle state */}
      {cameraState === "idle" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-800">
          <Camera className="w-16 h-16 text-slate-600 mb-4" />
          <p className="text-slate-500 mb-4">Camera is off</p>
          <Button onClick={startCamera}>
            <Video className="w-4 h-4 mr-2" />
            Start Camera
          </Button>
        </div>
      )}

      {/* Requesting state */}
      {cameraState === "requesting" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-800">
          <RefreshCw className="w-12 h-12 text-violet-400 mb-4 animate-spin" />
          <p className="text-slate-300">Requesting camera access...</p>
          <p className="text-slate-500 text-sm mt-2">Please allow camera permission</p>
        </div>
      )}

      {/* Error state */}
      {cameraState === "error" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-800 p-4">
          <AlertCircle className="w-12 h-12 text-red-400 mb-4" />
          <p className="text-red-400 text-center mb-4">{errorMessage}</p>
          <Button onClick={startCamera} variant="outline">
            <RefreshCw className="w-4 h-4 mr-2" />
            Try Again
          </Button>
        </div>
      )}

      {/* Active state overlays */}
      {cameraState === "active" && (
        <>
          {/* Stop button */}
          <Button
            onClick={stopCamera}
            size="sm"
            variant="destructive"
            className="absolute top-4 left-4 z-10"
          >
            <VideoOff className="w-4 h-4 mr-2" />
            Stop
          </Button>

          {/* Analyzing indicator */}
          {isAnalyzing && (
            <div className="absolute top-4 right-4 flex items-center gap-2 bg-slate-900/80 px-3 py-1.5 rounded-full z-10">
              <RefreshCw className="w-4 h-4 animate-spin text-violet-400" />
              <span className="text-xs text-slate-300">Analyzing...</span>
            </div>
          )}

          {/* Detection results */}
          {currentDetection && currentDetection.faces_detected > 0 && (
            <div className="absolute bottom-4 left-4 right-4 bg-slate-900/90 backdrop-blur-sm rounded-lg p-3 z-10">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse" />
                  <span className="text-sm text-white">
                    {currentDetection.faces_detected} face(s) detected
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {emotionIcons[currentEmotion]}
                  <span className="text-sm text-white capitalize font-medium">
                    {currentEmotion}
                  </span>
                  <span className="text-xs text-slate-400">
                    ({((currentDetection.analysis?.[0]?.emotion_confidence || 0) * 100).toFixed(0)}%)
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* No face detected */}
          {currentDetection && currentDetection.faces_detected === 0 && (
            <div className="absolute bottom-4 left-4 right-4 bg-yellow-900/80 backdrop-blur-sm rounded-lg p-3 z-10">
              <div className="flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-yellow-400" />
                <span className="text-sm text-yellow-200">No face detected - look at the camera</span>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default CameraDetection;
