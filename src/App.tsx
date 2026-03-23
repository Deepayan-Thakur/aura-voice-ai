import { useState, useEffect, useRef } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality } from '@google/genai';
import { Mic, Square, Loader2, Globe, Sparkles, Volume2 } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

function base64ToFloat32Array(base64: string) {
  const binary = atob(base64);
  const uint8 = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    uint8[i] = binary.charCodeAt(i);
  }
  const int16 = new Int16Array(uint8.buffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / (int16[i] < 0 ? 0x8000 : 0x7FFF);
  }
  return float32;
}

class AudioPlayer {
  context: AudioContext | null = null;
  nextPlayTime: number = 0;
  activeSources: AudioBufferSourceNode[] = [];

  init() {
    if (!this.context) {
      this.context = new AudioContext({ sampleRate: 24000 });
      this.nextPlayTime = this.context.currentTime;
    }
  }

  playChunk(float32Data: Float32Array) {
    if (!this.context || float32Data.length === 0) return;
    const buffer = this.context.createBuffer(1, float32Data.length, 24000);
    buffer.getChannelData(0).set(float32Data);
    
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    
    const startTime = Math.max(this.nextPlayTime, this.context.currentTime);
    source.start(startTime);
    this.nextPlayTime = startTime + buffer.duration;
    
    this.activeSources.push(source);
    source.onended = () => {
      this.activeSources = this.activeSources.filter(s => s !== source);
    };
  }

  interrupt() {
    this.activeSources.forEach(s => {
      try { s.stop(); } catch (e) {}
    });
    this.activeSources = [];
    if (this.context) {
      this.nextPlayTime = this.context.currentTime;
    }
  }
  
  close() {
    this.interrupt();
    if (this.context) {
      this.context.close();
      this.context = null;
    }
  }
}

class AudioRecorder {
  context: AudioContext | null = null;
  stream: MediaStream | null = null;
  processor: AudioWorkletNode | null = null;
  source: MediaStreamAudioSourceNode | null = null;
  onData: ((base64: string) => void) | null = null;

  async start() {
    this.context = new AudioContext({ sampleRate: 16000 });
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.source = this.context.createMediaStreamSource(this.stream);

    const workletCode = `
      class PCMProcessor extends AudioWorkletProcessor {
        process(inputs, outputs) {
          const input = inputs[0];
          if (input && input.length > 0) {
            const channelData = input[0];
            const data = new Float32Array(channelData);
            this.port.postMessage(data);
          }
          const output = outputs[0];
          if (output && output.length > 0) {
            for (let i = 0; i < output[0].length; i++) {
              output[0][i] = 0;
            }
          }
          return true;
        }
      }
      registerProcessor('pcm-processor', PCMProcessor);
    `;
    const blob = new Blob([workletCode], { type: 'application/javascript' });
    const workletUrl = URL.createObjectURL(blob);
    await this.context.audioWorklet.addModule(workletUrl);
    
    this.processor = new AudioWorkletNode(this.context, 'pcm-processor');
    this.processor.port.onmessage = (e) => {
      if (!this.onData) return;
      const float32Data = e.data as Float32Array;
      const pcm16 = new Int16Array(float32Data.length);
      for (let i = 0; i < float32Data.length; i++) {
        let s = Math.max(-1, Math.min(1, float32Data[i]));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      const uint8 = new Uint8Array(pcm16.buffer);
      let binary = '';
      const chunkSize = 0x8000;
      for (let i = 0; i < uint8.length; i += chunkSize) {
        binary += String.fromCharCode.apply(null, Array.from(uint8.subarray(i, i + chunkSize)));
      }
      const base64Data = btoa(binary);
      this.onData(base64Data);
    };

    this.source.connect(this.processor);
    this.processor.connect(this.context.destination);
  }

  stop() {
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }
    if (this.source) {
      this.source.disconnect();
      this.source = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
    }
    if (this.context) {
      this.context.close();
      this.context = null;
    }
  }
}

export default function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<{id: string, text: string, sender: 'user' | 'model' | 'system'}[]>([]);
  
  const aiRef = useRef<GoogleGenAI | null>(null);
  const sessionRef = useRef<any>(null);
  const audioPlayerRef = useRef<AudioPlayer | null>(null);
  const audioRecorderRef = useRef<AudioRecorder | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const addLog = (text: string, sender: 'user' | 'model' | 'system') => {
    setLogs(prev => [...prev, { id: Math.random().toString(36).substring(7), text, sender }]);
  };

  const connect = async () => {
    try {
      setIsConnecting(true);
      setError(null);
      
      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey) {
        throw new Error("GEMINI_API_KEY is missing");
      }
      
      aiRef.current = new GoogleGenAI({ apiKey });
      audioPlayerRef.current = new AudioPlayer();
      audioPlayerRef.current.init();
      
      audioRecorderRef.current = new AudioRecorder();
      
      const config: any = {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: { prebuiltVoiceConfig: { voiceName: "Zephyr" } },
        },
        systemInstruction: "You are Aura, a highly intelligent, conversational AI assistant. You understand and speak fluently in English and Hindi. You have access to Google Search. You MUST use Google Search proactively whenever the user asks for facts, news, data, or specific details. When you search, analyze the results and provide a comprehensive, detailed, and accurate answer containing all the important information the user requested. Be extremely helpful and provide all the data the user asks for.",
        tools: [{ googleSearch: {} }],
        outputAudioTranscription: {},
        inputAudioTranscription: {},
      };

      const sessionPromise = aiRef.current.live.connect({
        model: "gemini-2.5-flash-native-audio-preview-12-2025",
        config,
        callbacks: {
          onopen: async () => {
            addLog("Connected to Aura.", "system");
            setIsConnected(true);
            setIsConnecting(false);
            
            audioRecorderRef.current!.onData = (base64Data) => {
              sessionPromise.then(session => {
                session.sendRealtimeInput({
                  audio: { data: base64Data, mimeType: 'audio/pcm;rate=16000' }
                });
              });
            };
            await audioRecorderRef.current!.start();
          },
          onmessage: (message: LiveServerMessage) => {
            // Audio output
            const base64Audio = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (base64Audio) {
              const float32Data = base64ToFloat32Array(base64Audio);
              audioPlayerRef.current?.playChunk(float32Data);
            }
            
            // Interruption
            if (message.serverContent?.interrupted) {
              audioPlayerRef.current?.interrupt();
              addLog("Model interrupted.", "system");
            }
            
            // Output Transcription
            const outputText = message.serverContent?.modelTurn?.parts?.map(p => p.text).filter(Boolean).join('');
            if (outputText) {
              addLog(outputText, "model");
            }
          },
          onerror: (err) => {
            console.error("Live API Error:", err);
            setError(err.message || "An error occurred");
            disconnect();
          },
          onclose: () => {
            addLog("Connection closed.", "system");
            disconnect();
          }
        }
      });
      
      sessionRef.current = await sessionPromise;
      
    } catch (err: any) {
      console.error(err);
      setError(err.message);
      setIsConnecting(false);
      disconnect();
    }
  };

  const disconnect = () => {
    if (sessionRef.current) {
      try { sessionRef.current.close(); } catch (e) {}
      sessionRef.current = null;
    }
    if (audioRecorderRef.current) {
      audioRecorderRef.current.stop();
      audioRecorderRef.current = null;
    }
    if (audioPlayerRef.current) {
      audioPlayerRef.current.close();
      audioPlayerRef.current = null;
    }
    setIsConnected(false);
    setIsConnecting(false);
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-50 flex flex-col items-center justify-center p-4 font-sans">
      {/* Header */}
      <div className="absolute top-0 left-0 w-full p-6 flex justify-between items-center">
        <div className="flex items-center gap-2 text-xl font-medium tracking-tight">
          <Sparkles className="w-6 h-6 text-blue-400" />
          <span>Aura</span>
        </div>
        <div className="flex items-center gap-4 text-sm text-neutral-400">
          <div className="flex items-center gap-1">
            <Globe className="w-4 h-4" />
            <span>English & Hindi</span>
          </div>
        </div>
      </div>

      {/* Main Orb */}
      <div className="relative flex items-center justify-center mb-12 mt-16">
        <motion.div
          animate={{
            scale: isConnected ? [1, 1.1, 1] : 1,
            opacity: isConnected ? [0.5, 0.8, 0.5] : 0.2,
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut"
          }}
          className={`absolute w-64 h-64 rounded-full blur-3xl ${isConnected ? 'bg-blue-500/30' : 'bg-neutral-800/50'}`}
        />
        <button
          onClick={isConnected ? disconnect : connect}
          disabled={isConnecting}
          className={`relative z-10 w-32 h-32 rounded-full flex items-center justify-center transition-all duration-500 shadow-2xl border border-white/10 ${
            isConnected 
              ? 'bg-gradient-to-br from-blue-500 to-indigo-600 hover:scale-95' 
              : 'bg-neutral-900 hover:bg-neutral-800 hover:scale-105'
          }`}
        >
          {isConnecting ? (
            <Loader2 className="w-10 h-10 animate-spin text-white/70" />
          ) : isConnected ? (
            <Square className="w-10 h-10 text-white fill-white/20" />
          ) : (
            <Mic className="w-10 h-10 text-white/70" />
          )}
        </button>
      </div>

      {/* Status Text */}
      <div className="text-center mb-8 h-12">
        <AnimatePresence mode="wait">
          <motion.p
            key={isConnecting ? 'connecting' : isConnected ? 'connected' : 'disconnected'}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="text-lg font-medium text-neutral-300"
          >
            {isConnecting ? "Connecting to Aura..." : 
             isConnected ? "Listening... Speak naturally." : 
             "Tap to start conversation"}
          </motion.p>
        </AnimatePresence>
        {error && (
          <p className="text-red-400 text-sm mt-2">{error}</p>
        )}
      </div>

      {/* Logs / Transcripts */}
      <div className="w-full max-w-2xl bg-neutral-900/50 border border-white/5 rounded-2xl p-4 h-64 overflow-y-auto backdrop-blur-sm">
        {logs.length === 0 ? (
          <div className="h-full flex items-center justify-center text-neutral-500 text-sm">
            Transcripts will appear here
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {logs.map(log => (
              <div key={log.id} className={`flex gap-3 text-sm ${log.sender === 'system' ? 'text-neutral-500' : 'text-neutral-300'}`}>
                <span className="shrink-0 mt-0.5">
                  {log.sender === 'model' ? <Volume2 className="w-4 h-4 text-blue-400" /> : 
                   log.sender === 'user' ? <Mic className="w-4 h-4 text-green-400" /> : 
                   <Sparkles className="w-4 h-4" />}
                </span>
                <span>{log.text}</span>
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        )}
      </div>
    </div>
  );
}
