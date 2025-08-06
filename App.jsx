import React, { useEffect, useState } from "react";
import { uploadVideo, uploadVideoFromURL, fetchVideos, fetchAnalysis } from "./api";
import VideoPlayer from "./components/VideoPlayer";
import Timeline from "./components/Timeline";
import VideoDropdown from "./components/VideoDropdown";

const COLORS = {
  primary: "#6C63FF",
  secondary: "#232946",
  accent: "#F4F6FB",
  glass: "rgba(255,255,255,0.15)",
  glassDark: "rgba(44,54,85,0.7)",
  danger: "#FF6B6B",
  warning: "#FFD166",
  success: "#06D6A0",
  info: "#118AB2"
};

const FONT = {
  main: "'Inter', 'Segoe UI', 'Roboto', 'Arial', sans-serif"
};

function App() {
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [videoURL, setVideoURL] = useState("");
  const [uploadMethod, setUploadMethod] = useState("file");
  const [urlTestResult, setUrlTestResult] = useState(null);
  const [darkMode, setDarkMode] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchVideos().then(setVideos);
  }, []);

  useEffect(() => {
    if (selectedVideo) {
      setLoading(true);
      fetchAnalysis(selectedVideo.id).then((res) => {
        setAnalysis(res);
        setLoading(false);
      });
    }
  }, [selectedVideo]);

  const testURL = async () => {
    if (!videoURL.trim()) {
      setUploadStatus("Please enter a valid video URL");
      setTimeout(() => setUploadStatus(""), 3000);
      return;
    }
    setUploadStatus("Testing URL accessibility...");
    try {
      const response = await fetch("http://localhost:5000/test-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: videoURL })
      });
      const result = await response.json();
      setUrlTestResult(result);
      if (result.accessible) {
        setUploadStatus("URL is accessible! You can now analyze the exam recording.");
      } else {
        setUploadStatus("URL is not accessible. Please check the URL and try again.");
      }
      setTimeout(() => setUploadStatus(""), 5000);
    } catch (error) {
      setUploadStatus("Error testing URL: " + error.message);
      setTimeout(() => setUploadStatus(""), 5000);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadStatus("Uploading and analyzing exam recording...");
      setLoading(true);
      try {
        const result = await uploadVideo(file);
        setUploadStatus("Exam recording uploaded and analyzed for cheating behavior!");
        const vids = await fetchVideos();
        setVideos(vids);
        setTimeout(() => setUploadStatus(""), 3000);
      } catch (error) {
        setUploadStatus("Upload failed: " + error.message);
        setTimeout(() => setUploadStatus(""), 5000);
      }
      setLoading(false);
    }
  };

  const handleURLUpload = async () => {
    if (!videoURL.trim()) {
      setUploadStatus("Please enter a valid video URL");
      setTimeout(() => setUploadStatus(""), 3000);
      return;
    }
    setUploadStatus("Downloading and analyzing exam recording from URL...");
    setLoading(true);
    try {
      const result = await uploadVideoFromURL(videoURL);
      setUploadStatus("Exam recording downloaded and analyzed for cheating behavior!");
      setVideoURL("");
      setUrlTestResult(null);
      const vids = await fetchVideos();
      setVideos(vids);
      setTimeout(() => setUploadStatus(""), 3000);
    } catch (error) {
      setUploadStatus("Upload failed: " + error.message);
      setTimeout(() => setUploadStatus(""), 5000);
    }
    setLoading(false);
  };

  const loadSampleURL = () => {
    setVideoURL("https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4");
  };

  // Analytics: Pie chart data for incident types
  const getIncidentStats = () => {
    if (!analysis || !analysis.suspicious_incidents) return {};
    const stats = {};
    analysis.suspicious_incidents.forEach((inc) => {
      stats[inc.type] = (stats[inc.type] || 0) + 1;
    });
    return stats;
  };

  // Pie chart SVG
  const IncidentPieChart = () => {
    const stats = getIncidentStats();
    const total = Object.values(stats).reduce((a, b) => a + b, 0);
    if (!total) return <div style={{ color: '#888', fontSize: 14 }}>No incidents</div>;
    const colors = [COLORS.danger, COLORS.warning, COLORS.info, COLORS.success, COLORS.primary];
    let acc = 0;
    let i = 0;
    return (
      <svg width="120" height="120" viewBox="0 0 32 32">
        {Object.entries(stats).map(([type, count], idx) => {
          const start = acc;
          const value = (count / total) * 100;
          acc += value;
          const x1 = 16 + 16 * Math.cos(2 * Math.PI * (start / 100));
          const y1 = 16 + 16 * Math.sin(2 * Math.PI * (start / 100));
          const x2 = 16 + 16 * Math.cos(2 * Math.PI * ((start + value) / 100));
          const y2 = 16 + 16 * Math.sin(2 * Math.PI * ((start + value) / 100));
          const large = value > 50 ? 1 : 0;
          const path = `M16,16 L${x1},${y1} A16,16 0 ${large} 1 ${x2},${y2} z`;
          return (
            <path key={type} d={path} fill={colors[idx % colors.length]}>
              <title>{type}: {count}</title>
            </path>
          );
        })}
      </svg>
    );
  };

  // Dark mode styles
  const darkStyles = darkMode ? {
    background: COLORS.secondary,
    color: COLORS.accent
  } : {};

  // Glassmorphism style
  const glass = {
    background: darkMode ? COLORS.glassDark : COLORS.glass,
    boxShadow: "0 8px 32px 0 rgba(31, 38, 135, 0.18)",
    backdropFilter: "blur(12px)",
    WebkitBackdropFilter: "blur(12px)",
    borderRadius: 18,
    border: darkMode ? "1.5px solid #23294655" : "1.5px solid #fff3",
  };

  // Animated loading spinner
  const Spinner = () => (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 120 }}>
      <div className="lds-roller">
        {[...Array(8)].map((_, i) => <div key={i}></div>)}
      </div>
      <style>{`
        .lds-roller { display: inline-block; position: relative; width: 80px; height: 80px; }
        .lds-roller div {
          animation: lds-roller 1.2s cubic-bezier(0.5, 0, 0.5, 1) infinite;
          transform-origin: 40px 40px;
        }
        .lds-roller div:after {
          content: " ";
          display: block;
          position: absolute;
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background: ${COLORS.primary};
          margin: -4px 0 0 -4px;
        }
        .lds-roller div:nth-child(1) { animation-delay: -0.036s; }
        .lds-roller div:nth-child(1):after { top: 63px; left: 63px; }
        .lds-roller div:nth-child(2) { animation-delay: -0.072s; }
        .lds-roller div:nth-child(2):after { top: 68px; left: 56px; }
        .lds-roller div:nth-child(3) { animation-delay: -0.108s; }
        .lds-roller div:nth-child(3):after { top: 71px; left: 48px; }
        .lds-roller div:nth-child(4) { animation-delay: -0.144s; }
        .lds-roller div:nth-child(4):after { top: 72px; left: 40px; }
        .lds-roller div:nth-child(5) { animation-delay: -0.18s; }
        .lds-roller div:nth-child(5):after { top: 71px; left: 32px; }
        .lds-roller div:nth-child(6) { animation-delay: -0.216s; }
        .lds-roller div:nth-child(6):after { top: 68px; left: 24px; }
        .lds-roller div:nth-child(7) { animation-delay: -0.252s; }
        .lds-roller div:nth-child(7):after { top: 63px; left: 17px; }
        .lds-roller div:nth-child(8) { animation-delay: -0.288s; }
        .lds-roller div:nth-child(8):after { top: 56px; left: 12px; }
        @keyframes lds-roller {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );

  // Main return
  return (
    <div style={{
      minHeight: "100vh",
      background: darkMode
        ? "linear-gradient(135deg, #181c2f 0%, #232946 100%)"
        : "linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%)",
      fontFamily: FONT.main,
      transition: "background 0.5s",
      ...darkStyles
    }}>
      {/* Sidebar */}
      <div style={{
        position: "fixed",
        left: 0,
        top: 0,
        height: "100vh",
        width: 220,
        background: darkMode ? "#181c2f" : "#fff",
        borderRight: darkMode ? "1.5px solid #23294655" : "1.5px solid #e0eafc",
        boxShadow: "2px 0 24px 0 rgba(44, 108, 223, 0.04)",
        zIndex: 10,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "32px 0 0 0"
      }}>
        <div style={{ fontWeight: 800, fontSize: 28, color: COLORS.primary, marginBottom: 32, letterSpacing: 1 }}>
          <span style={{ fontSize: 32 }}>🎓</span> ProctorAI
        </div>
        <button
          onClick={() => setDarkMode((d) => !d)}
          style={{
            background: darkMode ? COLORS.primary : COLORS.secondary,
            color: darkMode ? "#fff" : "#fff",
            border: "none",
            borderRadius: 8,
            padding: "10px 18px",
            fontWeight: 600,
            fontSize: 16,
            marginBottom: 24,
            cursor: "pointer",
            boxShadow: darkMode ? "0 2px 8px #0002" : "0 2px 8px #23294611"
          }}
        >
          {darkMode ? "☀️ Light Mode" : "🌙 Dark Mode"}
        </button>
        <div style={{ marginTop: "auto", marginBottom: 32, color: "#888", fontSize: 13, textAlign: "center" }}>
          <div>Made for Hackathons</div>
          <div style={{ fontWeight: 700, color: COLORS.primary }}>by Your Team</div>
        </div>
      </div>
      {/* Main Content */}
      <div style={{
        marginLeft: 220,
        minHeight: "100vh",
        padding: "0 0 32px 0",
        transition: "background 0.5s"
      }}>
        {/* Topbar */}
        <div style={{
          ...glass,
          margin: "0 32px 32px 32px",
          padding: "24px 32px 16px 32px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          position: "sticky",
          top: 0,
          zIndex: 5
        }}>
          <div style={{ fontWeight: 700, fontSize: 28, color: darkMode ? COLORS.accent : COLORS.secondary, letterSpacing: 1 }}>
            AI Exam Proctoring Dashboard
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
            <span style={{ fontSize: 18, color: COLORS.primary, fontWeight: 600 }}>
              {videos.length} Recordings
            </span>
            <span style={{ fontSize: 18, color: COLORS.danger, fontWeight: 600 }}>
              {analysis && analysis.suspicious_incidents ? analysis.suspicious_incidents.length : 0} Incidents
            </span>
          </div>
        </div>
        {/* Upload Section */}
        <div style={{ ...glass, margin: "0 32px 32px 32px", padding: 32, borderRadius: 24 }}>
          <h2 style={{ color: COLORS.primary, fontWeight: 700, marginBottom: 16, fontSize: 24, letterSpacing: 0.5 }}>
            Upload Exam Recording
          </h2>
          <div style={{ marginBottom: 16, display: "flex", gap: 8 }}>
            <button
              onClick={() => setUploadMethod("file")}
              style={{
                padding: "8px 16px",
                borderRadius: 6,
                border: "none",
                background: uploadMethod === "file" ? COLORS.primary : COLORS.accent,
                color: uploadMethod === "file" ? "#fff" : COLORS.secondary,
                cursor: "pointer",
                fontWeight: 500,
                fontSize: 16,
                boxShadow: uploadMethod === "file" ? "0 2px 8px #6C63FF22" : "none"
              }}
            >
              📁 Upload File
            </button>
            <button
              onClick={() => setUploadMethod("url")}
              style={{
                padding: "8px 16px",
                borderRadius: 6,
                border: "none",
                background: uploadMethod === "url" ? COLORS.primary : COLORS.accent,
                color: uploadMethod === "url" ? "#fff" : COLORS.secondary,
                cursor: "pointer",
                fontWeight: 500,
                fontSize: 16,
                boxShadow: uploadMethod === "url" ? "0 2px 8px #6C63FF22" : "none"
              }}
            >
              🔗 Video URL
            </button>
          </div>
          {uploadMethod === "file" && (
            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
              <label style={{
                background: COLORS.primary,
                color: "white",
                padding: "10px 22px",
                borderRadius: 8,
                cursor: "pointer",
                fontWeight: 600,
                fontSize: 18,
                boxShadow: "0 2px 8px #6C63FF22"
              }}>
                📹 Choose Exam Recording
                <input type="file" accept="video/*" onChange={handleFileUpload} style={{ display: "none" }} />
              </label>
            </div>
          )}
          {uploadMethod === "url" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                <input
                  type="url"
                  value={videoURL}
                  onChange={(e) => setVideoURL(e.target.value)}
                  placeholder="Enter exam recording URL (direct video link)"
                  style={{
                    flex: 1,
                    padding: "10px 14px",
                    borderRadius: 8,
                    border: `1.5px solid ${COLORS.primary}55`,
                    fontSize: 16,
                    background: darkMode ? COLORS.glassDark : COLORS.glass,
                    color: darkMode ? COLORS.accent : COLORS.secondary
                  }}
                />
                <button
                  onClick={loadSampleURL}
                  style={{
                    background: COLORS.info,
                    color: "white",
                    padding: "8px 12px",
                    borderRadius: 6,
                    border: "none",
                    cursor: "pointer",
                    fontWeight: 500,
                    fontSize: 14
                  }}
                >
                  Sample
                </button>
                <button
                  onClick={testURL}
                  style={{
                    background: COLORS.success,
                    color: "white",
                    padding: "8px 16px",
                    borderRadius: 6,
                    border: "none",
                    cursor: "pointer",
                    fontWeight: 500,
                    fontSize: 14
                  }}
                >
                  Test URL
                </button>
                <button
                  onClick={handleURLUpload}
                  style={{
                    background: COLORS.primary,
                    color: "white",
                    padding: "8px 18px",
                    borderRadius: 8,
                    border: "none",
                    cursor: "pointer",
                    fontWeight: 600,
                    fontSize: 18
                  }}
                >
                  Analyze Recording
                </button>
              </div>
              {urlTestResult && (
                <div style={{
                  padding: "8px 12px",
                  borderRadius: 6,
                  background: urlTestResult.accessible ? COLORS.success + "22" : COLORS.danger + "22",
                  color: urlTestResult.accessible ? COLORS.success : COLORS.danger,
                  fontSize: 14
                }}>
                  <strong>URL Test Result:</strong> {urlTestResult.accessible ? "Accessible" : "Not accessible"}
                  {urlTestResult.content_type && ` | Type: ${urlTestResult.content_type}`}
                  {urlTestResult.content_length && ` | Size: ${urlTestResult.content_length} bytes`}
                </div>
              )}
              <div style={{ fontSize: 12, color: "#666", fontStyle: "italic" }}>
                Note: For best results, use direct video file URLs (ending in .mp4, .avi, etc.)
                <br />
                Click "Sample" to load a test video URL.
              </div>
            </div>
          )}
          {uploadStatus && (
            <div style={{
              padding: "10px 16px",
              borderRadius: 8,
              marginTop: 18,
              background: uploadStatus.includes("analyzed") ? COLORS.success + "22" : COLORS.danger + "22",
              color: uploadStatus.includes("analyzed") ? COLORS.success : COLORS.danger,
              fontWeight: 600,
              fontSize: 16,
              boxShadow: "0 2px 8px #0001"
            }}>
              {uploadStatus}
            </div>
          )}
        </div>
        {/* Main Dashboard Content */}
        <div style={{ display: "flex", gap: 32, margin: "0 32px" }}>
          {/* Left: Video & Timeline */}
          <div style={{ flex: 2, minWidth: 0 }}>
            <div style={{ ...glass, marginBottom: 32, padding: 24, borderRadius: 20 }}>
              <VideoDropdown videos={videos} onSelect={setSelectedVideo} />
              <div style={{ marginTop: 24 }}>
                <VideoPlayer video={selectedVideo} />
              </div>
              <div style={{ marginTop: 24 }}>
                <Timeline analysis={analysis} videoId={selectedVideo && selectedVideo.id} />
              </div>
            </div>
          </div>
          {/* Right: Analytics & Incidents */}
          <div style={{ flex: 1, minWidth: 320 }}>
            <div style={{ ...glass, marginBottom: 32, padding: 24, borderRadius: 20 }}>
              <div style={{ fontWeight: 700, fontSize: 20, color: COLORS.primary, marginBottom: 12 }}>
                Incident Analytics
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
                <IncidentPieChart />
                <div style={{ fontSize: 15, color: darkMode ? COLORS.accent : COLORS.secondary }}>
                  {Object.entries(getIncidentStats()).map(([type, count]) => (
                    <div key={type} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                      <span style={{ fontWeight: 700 }}>{type.replace(/_/g, ' ').toUpperCase()}</span>
                      <span style={{ background: COLORS.primary + "22", color: COLORS.primary, borderRadius: 8, padding: "2px 8px", fontWeight: 600, fontSize: 13 }}>{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div style={{ ...glass, padding: 24, borderRadius: 20 }}>
              <div style={{ fontWeight: 700, fontSize: 20, color: COLORS.primary, marginBottom: 12 }}>
                Suspicious Incidents
              </div>
              {loading ? (
                <Spinner />
              ) : analysis && analysis.suspicious_incidents && analysis.suspicious_incidents.length > 0 ? (
                analysis.suspicious_incidents.map((incident, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: "16px",
                      borderRadius: 12,
                      border: `2px solid #6C63FF22`,
                      background: COLORS.accent,
                      marginBottom: 18,
                      boxShadow: "0 2px 8px #6C63FF11",
                      display: "flex",
                      alignItems: "flex-start",
                      gap: 16,
                      transition: "box-shadow 0.2s"
                    }}
                  >
                    <span style={{ fontSize: 24 }}>⚠️</span>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 700, color: COLORS.secondary, marginBottom: 4, fontSize: 17 }}>
                        {incident.description}
                      </div>
                      <div style={{ fontSize: 14, color: "#666", marginBottom: 8 }}>
                        Time: {incident.start}s - {incident.end}s | Duration: {(incident.end - incident.start).toFixed(1)}s
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{
                          background: COLORS.primary,
                          color: "white",
                          padding: "2px 8px",
                          borderRadius: 12,
                          fontSize: 11,
                          fontWeight: 600
                        }}>
                          Confidence: {Math.round(incident.confidence * 100)}%
                        </span>
                      </div>
                    </div>
                    <button
                      onClick={() => {
                        const video = document.querySelector('video');
                        if (video) {
                          video.currentTime = incident.start;
                        }
                      }}
                      style={{
                        background: COLORS.primary,
                        color: "white",
                        border: "none",
                        borderRadius: 6,
                        padding: "6px 12px",
                        cursor: "pointer",
                        fontSize: 12,
                        fontWeight: 500,
                        marginLeft: 8
                      }}
                    >
                      Jump to {incident.start}s
                    </button>
                  </div>
                ))
              ) : (
                <div style={{
                  padding: "20px",
                  borderRadius: 8,
                  background: COLORS.success + "22",
                  color: COLORS.success,
                  textAlign: "center",
                  fontWeight: 500,
                  fontSize: 16
                }}>
                  ✅ No suspicious incidents detected in this exam recording
                  <br />
                  <span style={{ fontSize: 14, fontWeight: 400 }}>
                    The enhanced detection system found no evidence of cheating behavior.
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
        <footer style={{
          textAlign: "center",
          color: COLORS.secondary,
          fontWeight: 500,
          padding: 24,
          fontSize: 15,
          marginTop: 48
        }}>
          &copy; {new Date().getFullYear()} ProctorAI &mdash; Hackathon Edition
        </footer>
      </div>
    </div>
  );
}

export default App;