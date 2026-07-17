"use client";

import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import { Play, Pause, Download, RotateCcw } from "lucide-react";

interface AudioWaveformPlayerProps {
  src: string;
  speakerName: string;
}

export default function AudioWaveformPlayer({ src, speakerName }: AudioWaveformPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const waveSurferRef = useRef<WaveSurfer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState("0:00");
  const [duration, setDuration] = useState("0:00");
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;

    // Build the absolute API URL
    const fileUrl = src.startsWith("http") ? src : `http://localhost:8000${src}`;

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "rgba(168, 85, 247, 0.3)", // Purple translucent
      progressColor: "rgb(168, 85, 247)", // Purple
      cursorColor: "rgb(59, 130, 246)", // Blue cursor
      barWidth: 2,
      barGap: 3,
      barRadius: 4,
      height: 64,
      normalize: true,
      url: fileUrl,
    });

    ws.on("play", () => setIsPlaying(true));
    ws.on("pause", () => setIsPlaying(false));
    
    ws.on("ready", () => {
      setIsReady(true);
      const formatTime = (time: number) => {
        const mins = Math.floor(time / 60);
        const secs = Math.floor(time % 60);
        return `${mins}:${secs < 10 ? "0" : ""}${secs}`;
      };
      setDuration(formatTime(ws.getDuration()));
    });

    ws.on("audioprocess", () => {
      const formatTime = (time: number) => {
        const mins = Math.floor(time / 60);
        const secs = Math.floor(time % 60);
        return `${mins}:${secs < 10 ? "0" : ""}${secs}`;
      };
      setCurrentTime(formatTime(ws.getCurrentTime()));
    });

    waveSurferRef.current = ws;

    return () => {
      ws.destroy();
    };
  }, [src]);

  const handlePlayPause = () => {
    if (waveSurferRef.current && isReady) {
      waveSurferRef.current.playPause();
    }
  };

  const handleRestart = () => {
    if (waveSurferRef.current && isReady) {
      waveSurferRef.current.setTime(0);
      if (!isPlaying) {
        waveSurferRef.current.play();
      }
    }
  };

  const downloadUrl = src.startsWith("http") ? src : `http://localhost:8000${src}`;

  return (
    <div className="flex flex-col gap-3 p-4 bg-slate-900/60 backdrop-blur-md rounded-xl border border-purple-500/20 shadow-lg transition-all hover:border-purple-500/40">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold text-purple-200">{speakerName}</h4>
        <span className="text-xs text-slate-400 font-mono">{currentTime} / {duration}</span>
      </div>
      
      <div ref={containerRef} className="w-full bg-slate-950/40 rounded-lg p-2 min-h-[64px]" />
      
      <div className="flex items-center gap-3">
        <button
          onClick={handlePlayPause}
          disabled={!isReady}
          className="flex items-center justify-center w-10 h-10 rounded-full bg-purple-600 hover:bg-purple-500 text-white transition-all shadow-md active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isPlaying ? <Pause size={18} /> : <Play size={18} className="ml-0.5" />}
        </button>
        
        <button
          onClick={handleRestart}
          disabled={!isReady}
          className="flex items-center justify-center w-10 h-10 rounded-full bg-slate-800 hover:bg-slate-700 text-slate-300 transition-all active:scale-95 disabled:opacity-50"
          title="Restart"
        >
          <RotateCcw size={16} />
        </button>
        
        <a
          href={downloadUrl}
          download={`${speakerName.toLowerCase().replace(" ", "_")}.wav`}
          target="_blank"
          rel="noopener noreferrer"
          className="ml-auto flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 border border-blue-500/30 text-xs font-medium transition-all active:scale-95"
        >
          <Download size={14} />
          Download WAV
        </a>
      </div>
    </div>
  );
}
