const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000/api';

interface ApiResponse<T = unknown> {
  data?: T;
  error?: string;
}

async function fetchApi<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<ApiResponse<T>> {
  const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;

  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...(token && { Authorization: `Bearer ${token}` }),
    ...options.headers,
  };

  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers,
    });

    const data = await response.json();

    if (!response.ok) {
      return { error: data.error || 'An error occurred' };
    }

    return { data };
  } catch (error) {
    return { error: 'Network error. Please try again.' };
  }
}

export const authService = {
  async register(userData: {
    email: string;
    username: string;
    password: string;
    first_name?: string;
    last_name?: string;
  }) {
    return fetchApi<{
      user: User;
      access_token: string;
      refresh_token: string;
    }>('/auth/register', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  },

  async login(credentials: { email: string; password: string }) {
    return fetchApi<{
      user: User;
      access_token: string;
      refresh_token: string;
    }>('/auth/login', {
      method: 'POST',
      body: JSON.stringify(credentials),
    });
  },

  async logout() {
    return fetchApi('/auth/logout', { method: 'POST' });
  },

  async getCurrentUser() {
    return fetchApi<{ user: User }>('/auth/me');
  },

  async refreshToken() {
    const refreshToken = localStorage.getItem('refresh_token');
    return fetchApi<{ access_token: string }>('/auth/refresh', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${refreshToken}`,
      },
    });
  },
};

export const userService = {
  async getDashboard() {
    return fetchApi<DashboardData>('/user/dashboard');
  },

  async updateProfile(data: Partial<User>) {
    return fetchApi<{ user: User }>('/user/profile', {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  async getDetectionHistory(page = 1, perPage = 20) {
    return fetchApi<{
      detections: Detection[];
      total: number;
      pages: number;
      current_page: number;
    }>(`/user/detection-history?page=${page}&per_page=${perPage}`);
  },
};

export const detectionService = {
  async analyzeFrame(imageData: string, authenticated = true) {
    const endpoint = authenticated ? '/detection/analyze' : '/detection/analyze-guest';
    return fetchApi<DetectionResult>(endpoint, {
      method: 'POST',
      body: JSON.stringify({ image: imageData }),
    });
  },

  async getStats() {
    return fetchApi<DetectionStats>('/detection/stats');
  },
};

export const chatService = {
  async sendMessage(message: string, emotion?: string) {
    return fetchApi<ChatResponse>('/chat/message', {
      method: 'POST',
      body: JSON.stringify({ message, emotion }),
    });
  },

  async getSuggestions() {
    return fetchApi<{ suggestions: string[] }>('/chat/suggestions');
  },
};

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  emotion?: string;
  timestamp: string;
}

export interface ChatResponse {
  success: boolean;
  response: string;
  emotion_detected?: string;
}

// Types
export interface User {
  id: number;
  email: string;
  username: string;
  first_name?: string;
  last_name?: string;
  avatar_url?: string;
  is_active: boolean;
  is_verified: boolean;
  created_at: string;
  last_login?: string;
}

export interface Detection {
  id: number;
  detection_type: string;
  result_data: {
    faces_count: number;
    emotions: string[];
  };
  confidence: number;
  created_at: string;
}

export interface DetectionResult {
  success: boolean;
  faces_detected: number;
  faces: Array<{
    x: number;
    y: number;
    width: number;
    height: number;
    confidence: number;
  }>;
  analysis: Array<{
    emotion: string;
    emotion_confidence: number;
    landmark_count: number;
  }>;
  assistance_message: string;
  image_size: {
    width: number;
    height: number;
  };
}

export interface DetectionStats {
  total_detections: number;
  face_detections: number;
  emotion_distribution: Record<string, number>;
}

export interface DashboardData {
  user: User;
  stats: {
    total_detections: number;
    face_detections: number;
    emotion_detections: number;
  };
  recent_detections: Detection[];
}
