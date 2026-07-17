"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  FileAudio,
  ArrowRight,
  Headphones,
  Cpu,
  Layers,
  Sparkles,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Info,
  Network,
  Target,
  ArrowDown,
  VolumeX
} from "lucide-react";

import AnimatedWaveformBackground from "@/components/AnimatedWaveformBackground";
import AudioWaveformPlayer from "@/components/AudioWaveformPlayer";

// Architecture steps definition
const ARCHITECTURE_STEPS = [
  { title: "Mixed Audio", desc: "Overlapping multi-speaker audio", icon: Headphones, color: "from-blue-500 to-indigo-500" },
  { title: "Conv Encoder", desc: "1D Convolution feature extraction", icon: Cpu, color: "from-indigo-500 to-purple-500" },
  { title: "Shared MossFormer", desc: "Attention & FSMN-based temporal processing", icon: Network, color: "from-purple-500 to-pink-500" },
  { title: "TDA Head", desc: "Estimates speaker queries & existence", icon: Target, color: "from-pink-500 to-red-500" },
  { title: "Mask Generator", desc: "Computes dynamic masks per speaker", icon: Layers, color: "from-red-500 to-orange-500" },
  { title: "Conv Decoder", desc: "Reconstructs speech in time-domain", icon: Cpu, color: "from-orange-500 to-yellow-500" },
  { title: "Separated Tracks", desc: "Independent audio files per speaker", icon: Sparkles, color: "from-yellow-500 to-green-500" }
];

export default function Home() {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [numSpeakers, setNumSpeakers] = useState<string>("auto");
  const [fileStats, setFileStats] = useState<{ size: string; duration: string } | null>(null);
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [error, setError] = useState<string | null>(null);
  
  const [results, setResults] = useState<{
    detected_speakers: number;
    model_used: string;
    processing_time: number;
    inference_time: number;
    separated_audio_files: string[];
    original_duration: number;
    presence_probs: number[];
  } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Format file size
  const formatBytes = (bytes: number, decimals = 2) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
  };

  // Format duration in seconds to MM:SS
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs < 10 ? "0" : ""}${secs}`;
  };

  // Handle file selection
  const processFile = (selectedFile: File) => {
    if (!selectedFile.name.toLowerCase().endsWith(".wav")) {
      setError("Please upload a valid .wav file.");
      return;
    }
    
    setError(null);
    setFile(selectedFile);
    
    // Get file size
    const sizeStr = formatBytes(selectedFile.size);
    
    // Get duration
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const reader = new FileReader();
    reader.onload = function(e) {
      if (e.target?.result) {
        audioContext.decodeAudioData(e.target.result as ArrayBuffer, (buffer) => {
          setFileStats({
            size: sizeStr,
            duration: formatDuration(buffer.duration)
          });
        }, () => {
          // Fallback if decoding fails
          setFileStats({ size: sizeStr, duration: "Unknown" });
        });
      }
    };
    reader.readAsArrayBuffer(selectedFile);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  // Trigger separation call
  const handleSeparate = async () => {
    if (!file) return;

    setIsProcessing(true);
    setCurrentStep(1); // Preprocessing
    setResults(null);
    setError(null);

    // Dynamic step simulator (phases 1, 2, 3)
    const interval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev < 4) {
          return prev + 1;
        }
        return prev;
      });
    }, 1500);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("num_speakers", numSpeakers);

    try {
      const response = await fetch("http://localhost:8000/separate", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errDetail = await response.json();
        throw new Error(errDetail.detail || "Server error occurred during separation.");
      }

      const data = await response.json();
      
      clearInterval(interval);
      setCurrentStep(4); // Speech Separation completed
      
      setTimeout(() => {
        setCurrentStep(5); // Audio Generation
        setTimeout(() => {
          setResults(data);
          setIsProcessing(false);
        }, 800);
      }, 800);

    } catch (err: any) {
      clearInterval(interval);
      setIsProcessing(false);
      setError(err.message || "An unexpected error occurred. Please make sure the FastAPI server is running.");
    }
  };

  const resetState = () => {
    setFile(null);
    setFileStats(null);
    setResults(null);
    setError(null);
    setIsProcessing(false);
    setCurrentStep(0);
    setNumSpeakers("auto");
  };

  return (
    <div className="min-h-screen bg-[#050508] bg-gradient-to-tr from-[#080714] via-[#050508] to-[#030616] text-white flex flex-col relative overflow-x-hidden">
      
      {/* Background animated canvas */}
      <AnimatedWaveformBackground />

      {/* Decorative Blur Orbs */}
      <div className="absolute top-1/4 left-1/4 w-[40vw] h-[40vw] rounded-full bg-purple-900/10 blur-[120px] pointer-events-none -z-10 animate-pulse duration-[8000ms]" />
      <div className="absolute bottom-1/4 right-1/4 w-[40vw] h-[40vw] rounded-full bg-blue-900/10 blur-[120px] pointer-events-none -z-10 animate-pulse duration-[6000ms]" />

      {/* Header */}
      <header className="w-full max-w-7xl mx-auto px-6 py-6 flex items-center justify-between border-b border-white/5 relative z-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-purple-600 to-blue-500 flex items-center justify-center shadow-lg shadow-purple-500/20">
            <VolumeX className="text-white" size={20} />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-wider bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-blue-400">MTC-NET</h1>
            <p className="text-[10px] text-slate-400 uppercase tracking-widest">Speech Separation</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="px-4 py-2 rounded-lg bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 text-xs font-semibold text-slate-300 transition-all"
          >
            API Docs
          </a>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 w-full max-w-7xl mx-auto px-6 py-12 relative z-10 flex flex-col gap-16">
        
        {/* Hero Section */}
        <section className="text-center flex flex-col items-center max-w-3xl mx-auto gap-6 mt-6">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-300 text-xs font-semibold"
          >
            <Sparkles size={12} />
            <span>State-of-the-Art Neural Audio Separation</span>
          </motion.div>

          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="text-5xl md:text-6xl font-extrabold tracking-tight"
          >
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-purple-400 via-violet-300 to-blue-400">
              MTC-Net
            </span>
          </motion.h2>

          <motion.h3
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-xl md:text-2xl font-medium text-slate-300"
          >
            AI-Powered Multi-Speaker Speech Separation
          </motion.h3>

          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="text-slate-400 text-sm md:text-base leading-relaxed"
          >
            MTC-Net leverages a shared MossFormer backbone with Transformer Decoder Attractors (TDA) 
            to separate overlapping speech into individual speaker tracks while dynamically estimating 
            the number of active speakers.
          </motion.p>
        </section>

        {/* Upload and Workflow Section */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* File Upload card */}
          <div className="lg:col-span-7 flex flex-col gap-6">
            <div className="bg-slate-900/40 backdrop-blur-xl border border-slate-800/80 rounded-2xl p-6 md:p-8 shadow-2xl flex flex-col gap-6 relative overflow-hidden">
              <h3 className="text-lg font-bold text-slate-200 flex items-center gap-2">
                <FileAudio size={18} className="text-purple-400" />
                Upload Mixture Audio
              </h3>
              
              <div
                onDragEnter={handleDrag}
                onDragOver={handleDrag}
                onDragLeave={handleDrag}
                onDrop={handleDrop}
                onClick={triggerFileInput}
                className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-300 relative ${
                  dragActive 
                    ? "border-purple-500 bg-purple-500/5" 
                    : "border-slate-800 bg-slate-950/20 hover:border-slate-700/60"
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".wav"
                  onChange={handleFileInput}
                  className="hidden"
                />
                
                <div className="w-16 h-16 rounded-full bg-slate-900/80 flex items-center justify-center border border-slate-800 text-slate-400 group-hover:text-white transition-all shadow-inner mb-4">
                  <Upload size={24} className="text-purple-400 animate-bounce" />
                </div>
                
                <p className="text-sm font-semibold text-slate-200">
                  Drag and drop your audio mixture file here, or <span className="text-purple-400 hover:text-purple-300 underline">browse</span>
                </p>
                <p className="text-xs text-slate-500 mt-2">Supports `.wav` format, mono or stereo, resampled dynamically.</p>
              </div>

              {/* Display selected file */}
              {file && (
                <div className="p-4 bg-slate-950/50 rounded-xl border border-slate-800 flex items-center justify-between gap-4">
                  <div className="flex items-center gap-3 overflow-hidden">
                    <div className="w-10 h-10 rounded-lg bg-purple-500/10 flex items-center justify-center border border-purple-500/20 text-purple-400">
                      <FileAudio size={18} />
                    </div>
                    <div className="overflow-hidden">
                      <p className="text-sm font-semibold truncate text-slate-200">{file.name}</p>
                      <div className="flex items-center gap-3 text-xs text-slate-400 mt-0.5">
                        <span>Size: {fileStats?.size || "Calculating..."}</span>
                        <span className="w-1 h-1 rounded-full bg-slate-600" />
                        <span>Duration: {fileStats?.duration || "Calculating..."}</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Speaker selection */}
              {file && (
                <div className="flex flex-col gap-2">
                  <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Target Speaker Model</label>
                  <select
                    value={numSpeakers}
                    onChange={(e) => setNumSpeakers(e.target.value)}
                    disabled={isProcessing}
                    className="w-full p-3 rounded-xl bg-slate-950/60 border border-slate-800 text-sm text-slate-200 focus:border-purple-500/50 outline-none transition-all disabled:opacity-50"
                  >
                    <option value="auto">Auto-Detect (First-Pass Analysis)</option>
                    <option value="2">Force 2 Speakers (MiniLibriMix Model)</option>
                    <option value="3">Force 3 Speakers (Libri3Mix Model)</option>
                  </select>
                </div>
              )}

              {error && (
                <div className="p-4 bg-red-950/20 rounded-xl border border-red-500/30 flex gap-3 text-red-200 text-sm">
                  <AlertCircle size={18} className="shrink-0 text-red-400" />
                  <p>{error}</p>
                </div>
              )}

              <button
                onClick={handleSeparate}
                disabled={!file || isProcessing}
                className="w-full py-4 rounded-xl font-bold text-sm bg-gradient-to-r from-purple-600 to-blue-500 hover:from-purple-500 hover:to-blue-400 text-white shadow-lg shadow-purple-500/20 hover:shadow-purple-500/30 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.98] flex items-center justify-center gap-2"
              >
                {isProcessing ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    Separating Speech Tracks...
                  </>
                ) : (
                  <>
                    <Sparkles size={16} />
                    Run Speech Separation
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Inference Flow / Loading Card */}
          <div className="lg:col-span-5 flex flex-col gap-6">
            <div className="bg-slate-900/40 backdrop-blur-xl border border-slate-800/80 rounded-2xl p-6 md:p-8 shadow-2xl h-full flex flex-col gap-6">
              <h3 className="text-lg font-bold text-slate-200 flex items-center gap-2">
                <Loader2 size={18} className={`text-blue-400 ${isProcessing ? 'animate-spin' : ''}`} />
                Separation Pipeline Status
              </h3>

              {!isProcessing && !results && (
                <div className="flex-1 flex flex-col items-center justify-center text-center p-8 text-slate-500">
                  <Info size={32} className="mb-3 text-slate-600" />
                  <p className="text-sm">Upload an audio mixture file and click "Run Speech Separation" to start the pipeline.</p>
                </div>
              )}

              {(isProcessing || results) && (
                <div className="flex flex-col gap-5 flex-1 justify-center">
                  {[
                    { step: 1, label: "Audio Preprocessing", desc: "Converting samplerate to 8kHz, mono downmixing" },
                    { step: 2, label: "Speaker Analysis", desc: "Estimating initial speaker count with MossFormer backbone" },
                    { step: 3, label: "Automatic Model Selection", desc: "Routing mixture to the optimal trained checkpoint" },
                    { step: 4, label: "Speech Separation", desc: "Computing dynamic time-frequency attractor masks" },
                    { step: 5, label: "Audio Generation", desc: "Reconstructing time-domain waveforms for output" }
                  ].map((s) => {
                    const isCompleted = currentStep > s.step;
                    const isActive = currentStep === s.step;
                    const isPending = currentStep < s.step;

                    return (
                      <div key={s.step} className="flex gap-4 items-start">
                        <div className="shrink-0 mt-0.5">
                          {isCompleted ? (
                            <div className="w-5 h-5 rounded-full bg-green-500/20 border border-green-500 flex items-center justify-center text-green-400">
                              <CheckCircle2 size={12} />
                            </div>
                          ) : isActive ? (
                            <div className="w-5 h-5 rounded-full bg-blue-500/20 border border-blue-500 flex items-center justify-center text-blue-400">
                              <Loader2 size={12} className="animate-spin" />
                            </div>
                          ) : (
                            <div className="w-5 h-5 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-500 text-xs">
                              {s.step}
                            </div>
                          )}
                        </div>
                        <div>
                          <p className={`text-sm font-semibold transition-all ${
                            isCompleted ? "text-green-400" : isActive ? "text-blue-400" : "text-slate-500"
                          }`}>
                            {s.label}
                          </p>
                          <p className="text-xs text-slate-500 mt-0.5">{s.desc}</p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Results Page */}
        <AnimatePresence>
          {results && (
            <motion.section
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 30 }}
              transition={{ duration: 0.5 }}
              className="flex flex-col gap-8 scroll-mt-6"
              id="results-section"
            >
              <div className="flex items-center justify-between">
                <h3 className="text-2xl font-bold tracking-tight text-slate-100 flex items-center gap-3">
                  <Sparkles className="text-purple-400 animate-pulse" size={24} />
                  Separated Speaker Outputs
                </h3>
                <button
                  onClick={resetState}
                  className="text-xs font-semibold text-slate-400 hover:text-slate-200 transition-all border border-slate-800 bg-slate-900/40 px-3 py-1.5 rounded-lg"
                >
                  Clear Results
                </button>
              </div>

              {/* Diagnostics Grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: "Detected Speaker Count", value: `${results.detected_speakers} Speakers`, color: "text-purple-400" },
                  { label: "Model Checkpoint Used", value: results.model_used, color: "text-blue-400" },
                  { label: "Total Processing Time", value: `${results.processing_time.toFixed(2)}s`, color: "text-indigo-400" },
                  { label: "Core Model Inference", value: `${results.inference_time.toFixed(2)}s`, color: "text-pink-400" }
                ].map((stat, i) => (
                  <div key={i} className="p-4 bg-slate-900/30 backdrop-blur-xl border border-slate-800/80 rounded-xl shadow-lg">
                    <p className="text-xs text-slate-500 font-medium">{stat.label}</p>
                    <p className={`text-base md:text-lg font-bold mt-1 ${stat.color}`}>{stat.value}</p>
                  </div>
                ))}
              </div>

              {/* Informational Message Banner if speakers > 3 */}
              {results.detected_speakers > 3 && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="p-4 bg-yellow-950/20 rounded-xl border border-yellow-500/20 flex gap-3 text-yellow-300 text-xs md:text-sm leading-relaxed"
                >
                  <Info size={20} className="shrink-0 text-yellow-400 mt-0.5" />
                  <p>
                    <strong>Note:</strong> The uploaded recording appears to contain more speakers than the available pretrained models were trained for. The current system contains dedicated checkpoints for 2-speaker and 3-speaker mixtures. Separation has still been performed using the closest available model, so output quality may decrease for more complex mixtures.
                  </p>
                </motion.div>
              )}

              {/* Separated Audio Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {results.separated_audio_files.map((fileUrl, index) => (
                  <AudioWaveformPlayer
                    key={index}
                    src={fileUrl}
                    speakerName={`Separated Speaker Track ${index + 1}`}
                  />
                ))}
              </div>
            </motion.section>
          )}
        </AnimatePresence>

        {/* MTC-Net Architecture Section */}
        <section className="bg-slate-900/30 backdrop-blur-xl border border-slate-800/80 rounded-2xl p-8 shadow-2xl flex flex-col gap-8 relative overflow-hidden">
          <div className="text-center max-w-xl mx-auto flex flex-col gap-2">
            <h3 className="text-2xl font-bold tracking-tight">Model Processing Architecture</h3>
            <p className="text-sm text-slate-400">
              The internal neural pipeline of MTC-Net, from the combined input waveform to the reconstructed speaker tracks.
            </p>
          </div>

          {/* Connected Cards */}
          <div className="hidden lg:flex flex-row justify-between items-center gap-1.5 relative py-6">
            {ARCHITECTURE_STEPS.map((step, idx) => {
              const Icon = step.icon;
              return (
                <div key={idx} className="flex items-center gap-1.5 flex-1 last:flex-initial">
                  {/* Card */}
                  <div className="flex flex-col items-center text-center p-4 bg-slate-900/60 backdrop-blur-md rounded-xl border border-slate-800 flex-1 min-h-[140px] justify-center transition-all hover:border-purple-500/30 group">
                    <div className={`w-10 h-10 rounded-lg bg-gradient-to-tr ${step.color} flex items-center justify-center shadow-lg group-hover:scale-110 transition-all duration-300`}>
                      <Icon className="text-white" size={18} />
                    </div>
                    <h4 className="text-xs font-semibold text-slate-200 mt-3">{step.title}</h4>
                    <p className="text-[9px] text-slate-500 mt-1 max-w-[120px]">{step.desc}</p>
                  </div>
                  {/* Connector Arrow */}
                  {idx < ARCHITECTURE_STEPS.length - 1 && (
                    <ArrowRight className="text-slate-600 shrink-0 mx-1" size={14} />
                  )}
                </div>
              );
            })}
          </div>

          {/* Mobile view for Architecture (Vertical flow) */}
          <div className="flex lg:hidden flex-col gap-3">
            {ARCHITECTURE_STEPS.map((step, idx) => {
              const Icon = step.icon;
              return (
                <div key={idx} className="flex flex-col items-center">
                  <div className="flex items-center gap-4 w-full p-4 bg-slate-900/60 backdrop-blur-md rounded-xl border border-slate-800">
                    <div className={`w-10 h-10 rounded-lg bg-gradient-to-tr ${step.color} flex items-center justify-center shadow-lg shrink-0`}>
                      <Icon className="text-white" size={18} />
                    </div>
                    <div>
                      <h4 className="text-sm font-semibold text-slate-200">{step.title}</h4>
                      <p className="text-xs text-slate-500 mt-0.5">{step.desc}</p>
                    </div>
                  </div>
                  {idx < ARCHITECTURE_STEPS.length - 1 && (
                    <ArrowDown className="text-slate-600 my-2" size={16} />
                  )}
                </div>
              );
            })}
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="w-full border-t border-white/5 bg-slate-950/40 relative z-10 py-8 mt-12 text-center text-xs text-slate-500 flex flex-col gap-2">
        <p className="font-semibold text-slate-400">MTC-Net Research Demonstration</p>
        <p>Powered by PyTorch + FastAPI • Responsive Dark Glassmorphism UI</p>
      </footer>
    </div>
  );
}
