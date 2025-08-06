const API_URL = "http://localhost:5000";
const VIDEO_DB_AUTH_KEY = "sk-M-gFHvoXEFzspHtSFV_5ASIfS7s_3sNWL8m8QOOV7Fc";

export async function uploadVideo(file) {
  try {
    const formData = new FormData();
    formData.append("video", file);
    const res = await fetch(`${API_URL}/upload`, {
      method: "POST",
      body: formData,
      headers: {
        "VideoDB-Auth": VIDEO_DB_AUTH_KEY
      }
    });

    if (!res.ok) {
      const errorData = await res.json();
      throw new Error(errorData.error || 'Upload failed');
    }

    return res.json();
  } catch (error) {
    console.error('Upload error:', error);
    throw error;
  }
}

export async function uploadVideoFromURL(videoURL) {
  try {
    const res = await fetch(`${API_URL}/upload`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "VideoDB-Auth": VIDEO_DB_AUTH_KEY
      },
      body: JSON.stringify({ video_url: videoURL })
    });

    if (!res.ok) {
      const errorData = await res.json();
      throw new Error(errorData.error || 'URL upload failed');
    }

    return res.json();
  } catch (error) {
    console.error('URL upload error:', error);
    throw error;
  }
}

export async function fetchVideos() {
  try {
    const res = await fetch(`${API_URL}/videos`, {
      headers: {
        "VideoDB-Auth": VIDEO_DB_AUTH_KEY
      }
    });

    if (!res.ok) {
      throw new Error('Failed to fetch videos');
    }

    return res.json();
  } catch (error) {
    console.error('Fetch videos error:', error);
    return [];
  }
}

export async function fetchAnalysis(videoId) {
  try {
    const res = await fetch(`${API_URL}/analysis/${videoId}`, {
      headers: {
        "VideoDB-Auth": VIDEO_DB_AUTH_KEY
      }
    });

    if (!res.ok) {
      throw new Error('Failed to fetch analysis');
    }

    return res.json();
  } catch (error) {
    console.error('Fetch analysis error:', error);
    return { unsafe_segments: [] };
  }
}

export async function fetchVideosFromVideoDB() {
  const res = await fetch("http://localhost:5000/videos_videodb");
  if (!res.ok) throw new Error("Failed to fetch videos from VideoDB");
  return res.json();
}

export function getThumbnailUrl(videoId, timestamp) {
  return `${API_URL}/thumbnail/${videoId}/${timestamp}`;
}
