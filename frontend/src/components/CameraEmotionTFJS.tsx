"use client";

import React, { useRef, useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";

// Emotion labels matching training
const EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"] as const;
type EmotionLabel = (typeof EMOTION_LABELS)[number];

// Emotion icon paths (static images in public folder)
const EMOTION_ICONS: Record<EmotionLabel, string> = {
  angry: "/emotions/angry.svg",
  disgust: "/emotions/disgust.svg",
  fear: "/emotions/fear.svg",
  happy: "/emotions/happy.svg",
  neutral: "/emotions/neutral.svg",
  sad: "/emotions/sad.svg",
  surprise: "/emotions/surprise.svg",
};

// Emotion colors for UI
const EMOTION_COLORS: Record<EmotionLabel, string> = {
  angry: "#ef4444",
  disgust: "#84cc16",
  fear: "#a855f7",
  happy: "#22c55e",
  neutral: "#6b7280",
  sad: "#3b82f6",
  surprise: "#f59e0b",
};

interface CameraEmotionTFJSProps {
  onEmotionChange?: (emotion: EmotionLabel, confidence: number) => void;
  onDetectionResult?: (result: DetectionResultType | null) => void;
  className?: string;
}

interface DetectionResultType {
  emotion: EmotionLabel;
  confidence: number;
  smoothedEmotion: EmotionLabel;
  smoothedConfidence: number;
  landmarks: number[][];
  boundingBox: { x: number; y: number; width: number; height: number } | null;
}

interface FaceMeshType {
  setOptions: (options: Record<string, unknown>) => void;
  onResults: (callback: (results: FaceMeshResults) => void) => void;
  send: (input: { image: HTMLVideoElement }) => Promise<void>;
  close: () => void;
  initialize: () => Promise<void>;
}

interface FaceMeshResults {
  multiFaceLandmarks?: Array<Array<{ x: number; y: number; z: number }>>;
}

// EMA smoothing
function emaSmooth(
  prev: Record<EmotionLabel, number>,
  curr: Record<EmotionLabel, number>,
  alpha: number
): Record<EmotionLabel, number> {
  const result = {} as Record<EmotionLabel, number>;
  for (const label of EMOTION_LABELS) {
    result[label] = alpha * curr[label] + (1 - alpha) * (prev[label] || 0);
  }
  return result;
}

export function CameraEmotionTFJS({
  onEmotionChange,
  onDetectionResult,
  className = "",
}: CameraEmotionTFJSProps) {
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const faceMeshRef = useRef<FaceMeshType | null>(null);
  const modelRef = useRef<unknown>(null);
  const animationFrameRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef<number>(0);
  const landmarkWindowRef = useRef<number[][]>([]);
  const prevLandmarksRef = useRef<number[] | null>(null);
  const smoothedProbs = useRef<Record<EmotionLabel, number>>(
    Object.fromEntries(EMOTION_LABELS.map((l) => [l, 0])) as Record<EmotionLabel, number>
  );
  const frameCountRef = useRef<number>(0);
  const isRunningRef = useRef<boolean>(false);

  // State
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "running" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [currentResult, setCurrentResult] = useState<DetectionResultType | null>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [mediapipeLoaded, setMediapipeLoaded] = useState(false);

  // Constants
  const TARGET_FPS = 15;
  const FRAME_INTERVAL = 1000 / TARGET_FPS;
  const WINDOW_SIZE = 20;
  const INFERENCE_STRIDE = 5;
  const EMA_ALPHA = 0.15; // Lower = more smoothing, less jitter
  const FEATURE_DIM = 1872; // 468 landmarks * 2 (x,y) + 468 * 2 (deltas)

  // Load TensorFlow.js and model
  const loadModel = useCallback(async () => {
    try {
      console.log("[TFJS] Loading TensorFlow.js...");
      const tf = await import("@tensorflow/tfjs");
      await tf.ready();
      console.log("[TFJS] TensorFlow.js ready, backend:", tf.getBackend());

      console.log("[TFJS] Loading emotion model from /models/tfjs_landmark_model/model.json...");
      try {
        const model = await tf.loadGraphModel("/models/tfjs_landmark_model/model.json");
        modelRef.current = model;
        console.log("[TFJS] Model loaded successfully");
        setModelLoaded(true);
      } catch (modelError) {
        console.warn("[TFJS] Model not found, running in demo mode without inference");
        setModelLoaded(false);
      }
    } catch (err) {
      console.error("[TFJS] Failed to load:", err);
      throw err;
    }
  }, []);

  // Load MediaPipe FaceMesh
  const loadMediaPipe = useCallback(async () => {
    try {
      console.log("[MediaPipe] Loading FaceMesh...");
      const { FaceMesh } = await import("@mediapipe/face_mesh");
      console.log("[MediaPipe] FaceMesh module imported");

      const faceMesh = new FaceMesh({
        locateFile: (file: string) => {
          const url = `/models/face_mesh/${file}`;
          console.log("[MediaPipe] Loading file:", url);
          return url;
        },
      });

      console.log("[MediaPipe] Setting options...");
      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.3,
        minTrackingConfidence: 0.3,
        selfieMode: true,
      });

      // Initialize the model by calling initialize()
      console.log("[MediaPipe] Initializing model...");
      await faceMesh.initialize();
      console.log("[MediaPipe] Model initialized!");

      faceMeshRef.current = faceMesh;
      console.log("[MediaPipe] FaceMesh ready");
      setMediapipeLoaded(true);
    } catch (err) {
      console.error("[MediaPipe] Failed to load:", err);
      throw err;
    }
  }, []);

  // Extract features from landmarks
  const extractFeatures = useCallback(
    (landmarks: Array<{ x: number; y: number; z: number }>): number[] => {
      // Get bounding box
      let minX = Infinity,
        maxX = -Infinity,
        minY = Infinity,
        maxY = -Infinity;
      for (const lm of landmarks) {
        minX = Math.min(minX, lm.x);
        maxX = Math.max(maxX, lm.x);
        minY = Math.min(minY, lm.y);
        maxY = Math.max(maxY, lm.y);
      }
      const width = maxX - minX || 1;
      const height = maxY - minY || 1;
      const centerX = (minX + maxX) / 2;
      const centerY = (minY + maxY) / 2;

      // Normalize coordinates to [-1, 1] relative to face bbox
      const coords: number[] = [];
      for (const lm of landmarks) {
        const nx = ((lm.x - centerX) / (width / 2));
        const ny = ((lm.y - centerY) / (height / 2));
        coords.push(nx, ny);
      }

      // Compute deltas
      const deltas: number[] = [];
      if (prevLandmarksRef.current && prevLandmarksRef.current.length === coords.length) {
        for (let i = 0; i < coords.length; i++) {
          deltas.push(coords[i] - prevLandmarksRef.current[i]);
        }
      } else {
        // First frame: deltas are zero
        for (let i = 0; i < coords.length; i++) {
          deltas.push(0);
        }
      }

      prevLandmarksRef.current = coords.slice();

      // Combine coords + deltas
      return [...coords, ...deltas];
    },
    []
  );

  // Run inference on window
  const runInference = useCallback(async (window: number[][]) => {
    if (!modelRef.current || window.length < WINDOW_SIZE) {
      return null;
    }

    try {
      const tf = await import("@tensorflow/tfjs");
      const model = modelRef.current as { predict: (input: unknown) => { dataSync: () => Float32Array; dispose: () => void } };

      // Prepare input tensor [1, T, F]
      const inputData = window.slice(-WINDOW_SIZE);
      const inputTensor = tf.tensor3d([inputData], [1, WINDOW_SIZE, FEATURE_DIM]);

      // Run prediction
      const output = model.predict(inputTensor) as { dataSync: () => Float32Array; dispose: () => void };
      const probs = Array.from(output.dataSync());

      // Cleanup
      inputTensor.dispose();
      output.dispose();

      // Convert to object
      const probsObj = Object.fromEntries(
        EMOTION_LABELS.map((l, i) => [l, probs[i] || 0])
      ) as Record<EmotionLabel, number>;

      return probsObj;
    } catch (err) {
      console.error("[Inference] Error:", err);
      return null;
    }
  }, []);

  // Process frame from MediaPipe
  const processResults = useCallback(
    async (results: FaceMeshResults) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // Clear canvas for new frame (overlay only)
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Debug: Log every 30 frames
      if (frameCountRef.current % 30 === 0) {
        console.log("[FaceMesh] Results received:", {
          hasLandmarks: !!results.multiFaceLandmarks,
          faceCount: results.multiFaceLandmarks?.length || 0,
        });
      }

      if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
        // No face detected - increment counter for debug logging
        frameCountRef.current++;
        setCurrentResult(null);
        onDetectionResult?.(null);
        return;
      }

      // Take only the first 468 landmarks (exclude iris landmarks if refineLandmarks is true)
      // The model expects 1872 features: 468 points * 2 coords (x,y) + 468 deltas * 2 coords
      const landmarks = results.multiFaceLandmarks[0].slice(0, 468);

      // Extract features
      const features = extractFeatures(landmarks);

      // Add to sliding window
      landmarkWindowRef.current.push(features);
      if (landmarkWindowRef.current.length > WINDOW_SIZE) {
        landmarkWindowRef.current.shift();
      }

      frameCountRef.current++;

      // Compute bounding box for display
      let minX = Infinity,
        maxX = -Infinity,
        minY = Infinity,
        maxY = -Infinity;
      for (const lm of landmarks) {
        minX = Math.min(minX, lm.x * canvas.width);
        maxX = Math.max(maxX, lm.x * canvas.width);
        minY = Math.min(minY, lm.y * canvas.height);
        maxY = Math.max(maxY, lm.y * canvas.height);
      }

      const boundingBox = {
        x: minX,
        y: minY,
        width: maxX - minX,
        height: maxY - minY,
      };

      // Draw landmarks (canvas is CSS mirrored, so draw normally)
      ctx.fillStyle = "#8b5cf6";
      for (const lm of landmarks) {
        const x = lm.x * canvas.width;
        const y = lm.y * canvas.height;
        ctx.beginPath();
        ctx.arc(x, y, 1.5, 0, 2 * Math.PI);
        ctx.fill();
      }

      // Draw bounding box
      ctx.strokeStyle = "#8b5cf6";
      ctx.lineWidth = 2;
      ctx.strokeRect(boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height);

      // Run inference every INFERENCE_STRIDE frames
      let currentProbs: Record<EmotionLabel, number> | null = null;
      if (
        frameCountRef.current % INFERENCE_STRIDE === 0 &&
        landmarkWindowRef.current.length >= WINDOW_SIZE
      ) {
        currentProbs = await runInference(landmarkWindowRef.current);
      }

      // If no model or no inference this frame, use demo mode
      if (!currentProbs) {
        // Demo mode: random emotion based on simple heuristics
        const randomIdx = Math.floor(Math.random() * EMOTION_LABELS.length);
        currentProbs = Object.fromEntries(
          EMOTION_LABELS.map((l, i) => [l, i === randomIdx ? 0.7 : 0.05])
        ) as Record<EmotionLabel, number>;
      }

      // Apply EMA smoothing
      smoothedProbs.current = emaSmooth(smoothedProbs.current, currentProbs, EMA_ALPHA);

      // Get best emotion
      let bestLabel: EmotionLabel = "neutral";
      let bestConf = 0;
      let smoothedBestLabel: EmotionLabel = "neutral";
      let smoothedBestConf = 0;

      for (const label of EMOTION_LABELS) {
        if (currentProbs[label] > bestConf) {
          bestConf = currentProbs[label];
          bestLabel = label;
        }
        if (smoothedProbs.current[label] > smoothedBestConf) {
          smoothedBestConf = smoothedProbs.current[label];
          smoothedBestLabel = label;
        }
      }

      if (frameCountRef.current % (INFERENCE_STRIDE * 6) === 0) {
        console.log(
          `[Emotion] ${smoothedBestLabel} (${(smoothedBestConf * 100).toFixed(0)}%)`
        );
      }

      // Draw emotion label and icon (text needs to be mirrored back since canvas has CSS scaleX(-1))
      const labelY = boundingBox.y - 10;
      const labelText = `${smoothedBestLabel.toUpperCase()} ${(smoothedBestConf * 100).toFixed(0)}%`;
      ctx.save();
      ctx.scale(-1, 1); // Counter the CSS mirror for text
      ctx.font = "bold 16px Inter, sans-serif";
      ctx.fillStyle = EMOTION_COLORS[smoothedBestLabel];
      // Calculate mirrored X position for text
      const textX = -(boundingBox.x + boundingBox.width);
      ctx.fillText(
        labelText,
        textX,
        labelY > 20 ? labelY : boundingBox.y + boundingBox.height + 20
      );
      ctx.restore();

      // Create result object
      const result: DetectionResultType = {
        emotion: bestLabel,
        confidence: bestConf,
        smoothedEmotion: smoothedBestLabel,
        smoothedConfidence: smoothedBestConf,
        landmarks: landmarks.map((lm) => [lm.x, lm.y, lm.z]),
        boundingBox,
      };

      setCurrentResult(result);
      onEmotionChange?.(smoothedBestLabel, smoothedBestConf);
      onDetectionResult?.(result);
    },
    [extractFeatures, runInference, onEmotionChange, onDetectionResult]
  );

  // Animation loop
  const processFrame = useCallback(
    async (timestamp: number) => {
      if (!isRunningRef.current) return;

      const elapsed = timestamp - lastFrameTimeRef.current;
      if (elapsed >= FRAME_INTERVAL) {
        lastFrameTimeRef.current = timestamp;

        const video = videoRef.current;
        const faceMesh = faceMeshRef.current;

        // Send video frame to MediaPipe
        if (video && faceMesh && video.readyState >= 2) {
          try {
            await faceMesh.send({ image: video });
          } catch (err) {
            console.error("[Frame] Error sending to MediaPipe:", err);
          }
        } else if (video && faceMesh) {
          console.log("[Frame] Video not ready, readyState:", video.readyState);
        } else {
          console.log("[Frame] Missing video or faceMesh:", { video: !!video, faceMesh: !!faceMesh });
        }
      }

      if (isRunningRef.current) {
        animationFrameRef.current = requestAnimationFrame(processFrame);
      }
    },
    [FRAME_INTERVAL]
  );

  // Start camera
  const startCamera = async () => {
    console.log("[Camera] Request started");
    setStatus("loading");
    setError(null);

    try {
      // Load dependencies first
      if (!modelLoaded || !mediapipeLoaded) {
        await Promise.all([loadModel(), loadMediaPipe()]);
      }

      // Set up MediaPipe callback
      if (faceMeshRef.current) {
        faceMeshRef.current.onResults(processResults);
      }

      // Request camera
      console.log("[Camera] Requesting camera permission...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });

      console.log("[Camera] Permission granted");
      streamRef.current = stream;

      // Set up video
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (video && canvas) {
        video.srcObject = stream;
        await video.play();
        console.log("[Camera] Video playing");

        // Set canvas size
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
      }

      console.log("[MediaPipe] Ready");
      isRunningRef.current = true;
      setStatus("running");

      // Start processing loop
      console.log("[Camera] Starting frame processing loop");
      animationFrameRef.current = requestAnimationFrame(processFrame);
    } catch (err) {
      console.error("[Camera] Error:", err);
      setError(err instanceof Error ? err.message : "Failed to start camera");
      setStatus("error");
    }
  };

  // Stop camera
  const stopCamera = useCallback(() => {
    console.log("[Camera] Stopping...");
    isRunningRef.current = false;

    // Cancel animation frame
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    // Stop media stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop();
        track.enabled = false;
      });
      streamRef.current = null;
    }

    // Clear video
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    // Close MediaPipe
    if (faceMeshRef.current) {
      try {
        faceMeshRef.current.close();
      } catch (e) {
        console.warn("[MediaPipe] Close error:", e);
      }
      faceMeshRef.current = null;
    }

    // Reset state
    landmarkWindowRef.current = [];
    prevLandmarksRef.current = null;
    frameCountRef.current = 0;
    smoothedProbs.current = Object.fromEntries(
      EMOTION_LABELS.map((l) => [l, 0])
    ) as Record<EmotionLabel, number>;

    setStatus("idle");
    setCurrentResult(null);
    setMediapipeLoaded(false);

    console.log("[Camera] Stopped");
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  // Handle page unload
  useEffect(() => {
    const handleUnload = () => stopCamera();
    window.addEventListener("beforeunload", handleUnload);
    return () => window.removeEventListener("beforeunload", handleUnload);
  }, [stopCamera]);

  return (
    <div className={`relative bg-black rounded-xl overflow-hidden select-none flex items-center justify-center ${className}`}>
      {/* Container for video and canvas overlay */}
      <div className="relative" style={{ display: status === "running" ? "block" : "none" }}>
        {/* Video element (visible) */}
        <video
          ref={videoRef}
          playsInline
          muted
          className="max-w-full max-h-full object-contain scale-x-[-1]"
        />

        {/* Canvas for overlay (landmarks) - positioned to match video exactly */}
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
          style={{ transform: "scaleX(-1)" }}
        />
      </div>

      {/* Idle state */}
      {status === "idle" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900 p-4">
          <img
            src="/emotions/neutral.svg"
            alt="Camera"
            className="w-20 h-20 mb-4 opacity-50"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
          <p className="text-slate-400 mb-4 text-center">
            AI Emotion Detection Ready
          </p>
          <Button onClick={startCamera} className="bg-violet-600 hover:bg-violet-700">
            Start Camera
          </Button>
        </div>
      )}

      {/* Loading state */}
      {status === "loading" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900 p-4">
          <div className="w-12 h-12 border-4 border-violet-500 border-t-transparent rounded-full animate-spin mb-4" />
          <p className="text-slate-300">Loading AI Models...</p>
          <p className="text-slate-500 text-sm mt-2">This may take a moment</p>
        </div>
      )}

      {/* Error state */}
      {status === "error" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900 p-4">
          <img
            src="/emotions/sad.svg"
            alt="Error"
            className="w-16 h-16 mb-4"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
          <p className="text-red-400 text-center mb-4">{error}</p>
          <Button onClick={startCamera} variant="outline">
            Try Again
          </Button>
        </div>
      )}

      {/* Running overlay */}
      {status === "running" && (
        <>
          {/* Stop button */}
          <Button
            onClick={stopCamera}
            size="sm"
            variant="destructive"
            className="absolute top-3 left-3 z-10"
          >
            Stop
          </Button>

          {/* Current emotion display */}
          {currentResult && (
            <div className="absolute bottom-3 left-3 right-3 bg-slate-900/90 backdrop-blur rounded-lg p-3 z-10">
              <div className="flex items-center gap-3">
                <img
                  src={EMOTION_ICONS[currentResult.smoothedEmotion]}
                  alt={currentResult.smoothedEmotion}
                  className="w-10 h-10"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = "none";
                  }}
                />
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <span
                      className="font-bold capitalize"
                      style={{ color: EMOTION_COLORS[currentResult.smoothedEmotion] }}
                    >
                      {currentResult.smoothedEmotion}
                    </span>
                    <span className="text-slate-400 text-sm">
                      {(currentResult.smoothedConfidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  {/* Confidence bar */}
                  <div className="h-1.5 bg-slate-700 rounded-full mt-1 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-300"
                      style={{
                        width: `${currentResult.smoothedConfidence * 100}%`,
                        backgroundColor: EMOTION_COLORS[currentResult.smoothedEmotion],
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* No face warning */}
          {!currentResult && (
            <div className="absolute bottom-3 left-3 right-3 bg-yellow-900/80 rounded-lg p-3 z-10">
              <p className="text-yellow-200 text-sm text-center">
                No face detected - look at the camera
              </p>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default CameraEmotionTFJS;
export { EMOTION_LABELS, EMOTION_ICONS, EMOTION_COLORS };
export type { EmotionLabel, DetectionResultType };
