import React from "react";
import ThumbnailPreview from "./ThumbnailPreview";

function Timeline({ analysis, videoId }) {
  if (!analysis || !analysis.suspicious_incidents) return null;
  
  return (
    <div style={{ margin: "20px 0" }}>
      <h3 style={{ color: "#2c3e50", fontWeight: 600, marginBottom: 16 }}>
        📊 Incident Timeline
      </h3>
      <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
        {analysis.suspicious_incidents.map((incident, idx) => (
          <div
            key={idx}
            style={{
              border: "2px solid #e74c3c",
              background: "#fdf2f2",
              borderRadius: 8,
              padding: "10px 16px",
              minWidth: 200,
              fontWeight: 500,
              color: "#c0392b"
            }}
          >
            <div style={{ fontSize: 14, marginBottom: 4 }}>
              {incident.description}
            </div>
            <div style={{ fontSize: 12, color: "#7f8c8d" }}>
              {incident.start}s - {incident.end}s | {Math.round(incident.confidence * 100)}% confidence
            </div>
            <ThumbnailPreview videoId={videoId} timestamp={incident.start} />
          </div>
        ))}
      </div>
    </div>
  );
}

export default Timeline;
