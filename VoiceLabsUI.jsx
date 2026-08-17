import React, { useState } from "react";
import { Mic, Upload, Music, HelpCircle, Bell, User, Play, Volume2 } from "lucide-react";

export default function VoiceLabsUI() {
  const [emotion, setEmotion] = useState("Happy");
  const [recording, setRecording] = useState(false);
  const emotions = ["Default", "Happy", "Sad", "Calm", "Angry", "Excited", "Formal"];

  return (
    <div className="min-h-screen bg-slate-100 p-6">
      <div className="max-w-6xl mx-auto bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
        {/* Top bar */}
        <div className="flex items-center justify-between px-8 py-4 border-b border-slate-200">
          <div className="flex items-center gap-2">
            <div className="flex items-end gap-0.5">
              <span className="w-1 h-3 bg-indigo-900 rounded-sm"></span>
              <span className="w-1 h-5 bg-indigo-900 rounded-sm"></span>
              <span className="w-1 h-4 bg-indigo-900 rounded-sm"></span>
            </div>
            <span className="font-bold text-slate-900 text-lg">VOICE <span className="text-blue-600">LABS</span></span>
          </div>
          <div className="flex items-center gap-4 text-slate-500">
            <HelpCircle size={20} />
            <div className="relative">
              <Bell size={20} />
              <span className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full"></span>
            </div>
            <div className="w-8 h-8 rounded-full bg-slate-300 flex items-center justify-center">
              <User size={16} className="text-slate-600" />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 p-8">
          {/* Left: Voice Input */}
          <div>
            <h2 className="text-xl font-bold text-slate-900">Voice Input (Cloning)</h2>
            <p className="text-slate-500 font-medium mb-4">Step 1: Record or Upload Your Voice</p>

            <div className="grid grid-cols-2 gap-4">
              {/* Mic Record Card */}
              <div className="border border-slate-200 rounded-xl p-4 flex flex-col">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-slate-800">Mic Record</h3>
                  <HelpCircle size={16} className="text-slate-400" />
                </div>
                <div className="flex justify-center mb-4">
                  <div className="w-20 h-20 rounded-full border-2 border-slate-200 flex items-center justify-center">
                    <div className="w-14 h-14 rounded-full bg-slate-50 border border-slate-200 flex items-center justify-center">
                      <Mic size={22} className="text-slate-500" />
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => setRecording(!recording)}
                  className="mx-auto mb-4 flex items-center gap-2 bg-indigo-950 text-white text-sm font-semibold px-4 py-1.5 rounded-full"
                >
                  <span className="w-2 h-2 bg-red-500 rounded-full"></span> REC
                </button>
                <div className="flex items-center justify-center gap-0.5 h-8 mb-2">
                  {[6, 10, 14, 8, 16, 10, 6, 12, 8, 14, 10, 6, 12, 16, 8].map((h, i) => (
                    <span key={i} className="w-0.5 bg-blue-400 rounded-full" style={{ height: `${h}px` }}></span>
                  ))}
                </div>
                <div className="flex justify-between text-xs text-slate-400 mb-2">
                  <span>Duration</span>
                  <span>00:00 / 01:00</span>
                </div>
                <button className="bg-slate-700 text-white text-sm font-medium py-2 rounded-lg mb-4">
                  Start Recording
                </button>
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center overflow-hidden">
                    <User size={16} className="text-slate-500" />
                  </div>
                  <span className="text-sm text-slate-600 flex items-center gap-1">
                    <Volume2 size={12} /> sample_voice_A
                  </span>
                </div>
                <button className="bg-blue-600 text-white text-sm font-semibold py-2 rounded-lg">
                  Clone Voice
                </button>
              </div>

              {/* Upload File Card */}
              <div className="border border-slate-200 rounded-xl p-4 flex flex-col">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-slate-800">Upload File</h3>
                  <Upload size={16} className="text-slate-400" />
                </div>
                <div className="border-2 border-dashed border-slate-200 rounded-lg flex flex-col items-center justify-center py-6 mb-4">
                  <Upload size={28} className="text-slate-300 mb-2" />
                  <span className="font-semibold text-slate-700 text-sm">Upload zone</span>
                  <span className="text-xs text-slate-400">sample.mp3</span>
                  <span className="text-xs text-slate-400">12.5 MB • 32s</span>
                </div>
                <div className="flex gap-2 mb-4">
                  <button className="flex-1 border border-slate-300 text-slate-700 text-sm font-medium py-2 rounded-lg">
                    Browse File
                  </button>
                  <button className="flex-1 border border-slate-300 text-slate-700 text-sm font-medium py-2 rounded-lg">
                    Discard
                  </button>
                </div>
                <div className="flex items-center gap-2 mb-3">
                  <Music size={16} className="text-slate-400" />
                  <div>
                    <p className="text-sm text-slate-700 leading-tight">sample.mp3</p>
                    <p className="text-xs text-slate-400 leading-tight">12.5 MB • 32s</p>
                  </div>
                </div>
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center overflow-hidden">
                    <User size={16} className="text-slate-500" />
                  </div>
                  <span className="text-sm text-slate-600">sample_voice_A</span>
                </div>
                <button className="bg-blue-600 text-white text-sm font-semibold py-2 rounded-lg">
                  Clone Voice
                </button>
              </div>
            </div>
          </div>

          {/* Right: Generate Speech */}
          <div>
            <h2 className="text-xl font-bold text-slate-900 mb-4">Step 2: Generate Speech from Cloned Voice</h2>

            <div className="border border-slate-200 rounded-xl p-5">
              <label className="text-sm font-semibold text-slate-800">
                Enter Text <span className="text-slate-400 font-normal">(max 2000 chars)</span>
              </label>
              <textarea
                className="w-full mt-2 mb-4 border border-slate-200 rounded-lg p-3 text-sm text-slate-700 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
                rows={4}
                defaultValue="Welcome to the Voice Labs platform. Here you can clone any voice and generate natural-sounding speech with emotion..."
              />

              <p className="text-sm font-semibold text-slate-800 mb-2">Emotion Tags</p>
              <div className="flex flex-wrap gap-2 mb-5">
                {emotions.map((tag) => (
                  <button
                    key={tag}
                    onClick={() => setEmotion(tag)}
                    className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
                      emotion === tag
                        ? "bg-indigo-950 text-white"
                        : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                    }`}
                  >
                    {tag}
                  </button>
                ))}
              </div>

              <button className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg mb-6">
                Generate Speech
              </button>

              <h3 className="font-bold text-slate-900 mb-3">Generated Speech</h3>
              <div className="border border-slate-200 rounded-xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-full bg-slate-100 flex items-center justify-center">
                      <Mic size={16} className="text-slate-500" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-slate-800">Welcome to the...</p>
                      <p className="text-xs text-slate-400">sample_voice_A_happy</p>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <button className="border border-slate-300 text-sm font-medium px-3 py-1.5 rounded-lg text-slate-700">
                      Download Audio (MP3)
                    </button>
                    <button className="border border-slate-300 text-sm font-medium px-3 py-1.5 rounded-lg text-slate-700">
                      Regenerate
                    </button>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <button className="w-8 h-8 rounded-full bg-slate-800 flex items-center justify-center text-white">
                    <Play size={14} fill="white" />
                  </button>
                  <span className="text-xs text-slate-400 w-16">0:03 / 0:15</span>
                  <div className="flex items-center gap-0.5 flex-1 h-6">
                    {[4, 8, 12, 6, 10, 14, 8, 6, 10, 4, 8, 12, 6, 10, 4, 8, 6, 10, 4, 8].map((h, i) => (
                      <span
                        key={i}
                        className={`w-0.5 rounded-full ${i < 6 ? "bg-slate-700" : "bg-slate-200"}`}
                        style={{ height: `${h}px` }}
                      ></span>
                    ))}
                  </div>
                  <Volume2 size={16} className="text-slate-400" />
                  <div className="w-16 h-1 bg-slate-200 rounded-full relative">
                    <div className="absolute left-0 top-0 h-1 bg-slate-500 rounded-full w-3/4"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
