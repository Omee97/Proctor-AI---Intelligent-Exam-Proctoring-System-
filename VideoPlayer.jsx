import React from "react";

function VideoPlayer({ video }) {
  if (!video) return (
    <div style={{
      background: "#f6fafd",
      borderRadius: 12,
      padding: 40,
      textAlign: "center",
      color: "#666",
      fontSize: 18
    }}>
      Please select a video to play
    </div>
  );
  
  return (
    <div style={{
      background: "#f6fafd",
      borderRadius: 12,
      boxShadow: "0 2px 12px rgba(44, 108, 223, 0.08)",
      padding: 16,
      marginBottom: 24,
      textAlign: "center"
    }}>
      <h3 style={{ color: "#2d6cdf", marginBottom: 16 }}>Now Playing: {video.filename}</h3>
      <video 
        width="100%" 
        style={{ maxWidth: 600, borderRadius: 8 }} 
        controls
        onError={(e) => console.error("Video error:", e)}
      >
        <source src={`http://localhost:5000/uploads/${video.filename}`} type="video/mp4" />
        <source src={`http://localhost:5000/uploads/${video.filename}`} type="video/webm" />
        <source src={`http://localhost:5000/uploads/${video.filename}`} type="video/ogg" />
        Your browser does not support the video tag.
      </video>
    </div>
  );
}

export default VideoPlayer;
