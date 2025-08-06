import React from "react";

function VideoDropdown({ videos, onSelect }) {
  return (
    <select onChange={e => onSelect(videos.find(v => v.id === e.target.value))}>
      <option value="">Select a video</option>
      {videos.map(v => (
        <option key={v.id} value={v.id}>{v.filename}</option>
      ))}
    </select>
  );
}

export default VideoDropdown;
