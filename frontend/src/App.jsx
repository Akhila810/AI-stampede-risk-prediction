import { useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";
import "./App.css";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Area,
  AreaChart
} from "recharts";

const socket = io("https://stampede-backend1.onrender.com");

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const audioRef = useRef(null);
  const prevRiskRef = useRef("LOW");

  const [history, setHistory] = useState([]);

  const [chartData, setChartData] = useState([]);
  const [risk, setRisk] = useState("LOW");
  const [videoURL, setVideoURL] = useState(null);
  const [page, setPage] = useState("live");

  // -------------------------------
  // RISK → NUMBER
  // -------------------------------
  const riskToValue = (risk) => {
    if (risk === "LOW") return 1;
    if (risk === "MEDIUM") return 2;
    if (risk === "HIGH") return 3;
    return 0;
  };

  // -------------------------------
  // AUDIO INIT
  // -------------------------------
 useEffect(() => {
  const audio = new Audio("/alarm.mp3");
  audio.loop = true;
  audio.preload = "auto";   // 🔥 important
  audioRef.current = audio;
}, []);

  useEffect(() => {
  if (page === "history") {
    const fetchHistory = () => {
      fetch("https://stampede-backend1.onrender.com/history")
        .then(res => res.json())
        .then(data => {
          console.log("HISTORY FRONTEND:", data); // DEBUG
          setHistory(data);
        })
        .catch(err => console.error("History error:", err));
    };

    fetchHistory(); // initial fetch

    const interval = setInterval(fetchHistory, 2000); // keep updating

    return () => clearInterval(interval);
  }
}, [page]);

  // Unlock audio (browser restriction)
  useEffect(() => {
    const unlockAudio = () => {
      if (audioRef.current) {
        audioRef.current.play().then(() => {
          audioRef.current.pause();
          audioRef.current.currentTime = 0;
        }).catch(() => {});
      }
      document.removeEventListener("click", unlockAudio);
    };

    document.addEventListener("click", unlockAudio);
  }, []);

  // -------------------------------
  // SOCKET LISTENER
  // -------------------------------
  useEffect(() => {
    socket.on("frame_update", (data) => {
      console.log("DATA:", data);

      // 🔹 GRAPH UPDATE
      setChartData((prev) => {
        const newPoint = {
          time: new Date().toLocaleTimeString(),
          value: riskToValue(data.risk)
        };
        return [...prev, newPoint].slice(-20);
      });

      // 🔹 RISK + SOUND
      if (typeof data.risk === "string") {
        setRisk(data.risk);

        if (data.risk === "HIGH") {
  audioRef.current?.play().catch(err => {
    console.log("AUDIO ERROR:", err);
  });
} else {
  audioRef.current?.pause();
  audioRef.current.currentTime = 0;
}

        prevRiskRef.current = data.risk;
      }

      drawBoxes(data.detections || [], data.risk);
    });

    return () => socket.off("frame_update");
  }, []);

  // -------------------------------
  // VIDEO UPLOAD
  // -------------------------------
  const handleVideoUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    setVideoURL(url);

    const formData = new FormData();
    formData.append("video", file);

    await fetch("https://stampede-backend1.onrender.com/start", {
      method: "POST",
      body: formData,
    });
  };

  // -------------------------------
  // DRAW BOXES
  // -------------------------------
  const drawBoxes = (detections, riskLevel = "LOW") => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext("2d");

    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;

    ctx.lineWidth = 2;

    if (riskLevel === "HIGH") {
      ctx.strokeStyle = "red";
      ctx.fillStyle = "rgba(255,0,0,0.3)";
    } else if (riskLevel === "MEDIUM") {
      ctx.strokeStyle = "orange";
      ctx.fillStyle = "rgba(255,165,0,0.3)";
    } else {
      ctx.strokeStyle = "lime";
      ctx.fillStyle = "rgba(0,255,0,0.3)";
    }

    detections.forEach((box) => {
      const x = box.x1 * scaleX;
      const y = box.y1 * scaleY;
      const w = (box.x2 - box.x1) * scaleX;
      const h = (box.y2 - box.y1) * scaleY;

      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
    });
  };

  // -------------------------------
  // UI
  // -------------------------------
  return (
    
    <div className="dashboard">

      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo">AI</div>
            <h2>Monitor</h2>
          </div>

        <div className={`menu-item ${page === "live" ? "active" : ""}`}
          onClick={() => setPage("live")}>
          Live Feed
        </div>

        <div className={`menu-item ${page === "analytics" ? "active" : ""}`}
          onClick={() => setPage("analytics")}>
          Analytics
        </div>

        <div className={`menu-item ${page === "history" ? "active" : ""}`}
          onClick={() => setPage("history")}>
          History
        </div>
      </aside>

      {/* MAIN */}
      <div className="main-content">

        {/* TOPBAR */}
        <div className="topbar">
          <h1>Stampede Risk Monitoring</h1>
          
          {videoURL && (
            <div className={`risk-badge ${risk.toLowerCase()}`}>
              {risk}
            </div>
          )}
          </div>

        {/* CONTENT */}
        <div className="content">

          {/* LIVE */}
          {page === "live" && (
            <>
              <div className="video-section">
  {videoURL ? (
    <div className="video-card">
      <video
        ref={videoRef}
        src={videoURL}
        autoPlay
        muted
        controls
        className="video"
      />
      <canvas ref={canvasRef} className="canvas" />
    </div>
  ) : (
    <div className="upload-section">
      <label className="upload-card">
        <div className="upload-icon">📤</div>
        <h3>Upload Video</h3>
        <p>Select a video file to start monitoring</p>
        <span className="upload-btn-inner">Choose File</span>
        <input
          type="file"
          accept="video/*"
          hidden
          onChange={handleVideoUpload}
        />
      </label>
    </div>
  )}
</div>

              <div className={`stats-panel1 ${risk.toLowerCase()}`}>
                  <h3 className="live-title">
                  {videoURL && <span className={`live-dot ${risk.toLowerCase()}`}></span>}
                      Live Stats
                  </h3>

                  <p>
                    <span>Risk Level:</span> {risk}
                  </p>

                  <p>
                    <span>Status:</span> {videoURL ? "Running" : "Idle"}
                  </p>
               </div>
            </>
          )}

          {/* ANALYTICS (REAL GRAPH) */}
          {page === "analytics" && (
  <div className="stats-panel">
    <h3>Risk Trend</h3>

    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={chartData}>

        {/* GRADIENT */}
        <defs>
          <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
            <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
          </linearGradient>
        </defs>

        {/* GRID */}
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />

        {/* AXIS */}
        <XAxis dataKey="time" hide />
        <YAxis domain={[1, 3]} tick={{ fill: "#9ca3af" }} />

        {/* TOOLTIP */}
        <Tooltip
          contentStyle={{
            backgroundColor: "#020617",
            border: "1px solid #1f2937",
            borderRadius: "8px",
            color: "#fff"
          }}
        />

        {/* AREA (FILL) */}
        <Area
          type="monotone"
          dataKey="value"
          stroke="#ef4444"
          fill="url(#riskGradient)"
          strokeWidth={3}
          dot={false}
        />

      </AreaChart>
    </ResponsiveContainer>

    <p style={{ marginTop: "10px", color: "#9ca3af" }}>
      LOW → MEDIUM → HIGH
    </p>
  </div>
)}

          {/* HISTORY */}
          {page === "history" && (
  <div className="stats-panel">
    <h3>Event History</h3>

    {history.length === 0 ? (
      <p>No data yet</p>
    ) : (
      <div className="history-list">
        {history.slice().reverse().map((item, index) => (
          <div key={index} className={`history-item ${item.risk.toLowerCase()}`}>
            <span>{item.time}</span>
            <span>{item.risk}</span>
            <span>{item.people} people</span>
          </div>
        ))}
      </div>
    )}
  </div>
)}

        </div>
      </div>
    </div>
  );
}

export default App;