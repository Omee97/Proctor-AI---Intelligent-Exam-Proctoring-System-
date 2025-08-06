import React, { useState } from "react";
import { getThumbnailUrl } from "../api";

function ThumbnailPreview({ videoId, timestamp }) {
  const [show, setShow] = useState(false);
  const [thumbnailData, setThumbnailData] = useState(null);

  const handleMouseEnter = async () => {
    setShow(true);
    try {
      const response = await fetch(getThumbnailUrl(videoId, timestamp));
      const data = await response.json();
      setThumbnailData(data.message);
    } catch (error) {
      setThumbnailData("Thumbnail not available");
    }
  };

  return (
    <span
      onMouseEnter={handleMouseEnter}
      onMouseLeave={() => setShow(false)}
      style={{ marginLeft: 12, cursor: "pointer", color: "#2d6cdf", fontWeight: 600 }}
    >
      [Preview]
      {show && (
        <div style={{
          width: 140,
          borderRadius: 8,
          boxShadow: "0 2px 12px rgba(44, 108, 223, 0.18)",
          position: "absolute",
          zIndex: 10,
          border: "2px solid #2d6cdf",
          background: "white",
          marginTop: 8,
          padding: 8,
          fontSize: 12,
          color: "#666"
        }}>
          {thumbnailData || "Loading..."}
        </div>
      )}
    </span>
  );
}

export default ThumbnailPreview;
