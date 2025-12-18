import React, { useEffect, useRef, useState } from 'react';

// --- CẤU HÌNH ĐƯỜNG NỐI (CONNECTIONS) ---
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Ngón cái
  [0, 5], [5, 6], [6, 7], [7, 8],       // Ngón trỏ
  [0, 9], [9, 10], [10, 11], [11, 12],  // Ngón giữa
  [0, 13], [13, 14], [14, 15], [15, 16],// Ngón áp út
  [0, 17], [17, 18], [18, 19], [19, 20] // Ngón út
];

const POSE_CONNECTIONS = [
  [11, 13], [13, 15], // Tay trái
  [12, 14], [14, 16], // Tay phải
  [11, 12],           // Vai
  [23, 24],           // Hông
  [11, 23], [12, 24]  // Thân
];

interface Point { x: number; y: number; }
interface FrameData {
  pose: Point[];
  left_hand: Point[];
  right_hand: Point[];
}

interface SkeletonPlayerProps {
  jsonPath: string; 
  width?: number;
  height?: number;
}

const SkeletonPlayer: React.FC<SkeletonPlayerProps> = ({ jsonPath, width = 300, height = 250 }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [frames, setFrames] = useState<FrameData[]>([]);
  const frameIndexRef = useRef(0);
  const requestRef = useRef<number>();

  // 1. Load JSON
  useEffect(() => {
    frameIndexRef.current = 0;
    fetch(jsonPath)
      .then(res => res.json())
      .then(data => {
        console.log("Loaded Skeleton Frames:", data.length);
        setFrames(data);
      })
      .catch(err => console.error("Lỗi load JSON:", err));
  }, [jsonPath]);

  // 2. Hàm vẽ
  const draw = (ctx: CanvasRenderingContext2D, frame: FrameData) => {
    // Xóa màn hình cũ
    ctx.clearRect(0, 0, width, height);
    
    // Config nét vẽ
    ctx.lineWidth = 2;
    ctx.lineCap = "round";

    // Hàm kiểm tra điểm hợp lệ (Khác 0)
    const isValid = (p: Point) => p && (p.x !== 0 || p.y !== 0);

    // Hàm vẽ đường nối
    const drawConnectors = (points: Point[], connections: number[][], color: string) => {
      ctx.strokeStyle = color;
      ctx.beginPath();
      connections.forEach(([start, end]) => {
        const p1 = points[start];
        const p2 = points[end];
        
        if (isValid(p1) && isValid(p2)) {
          ctx.moveTo(p1.x * width, p1.y * height);
          ctx.lineTo(p2.x * width, p2.y * height);
        }
      });
      ctx.stroke();
    };

    // Hàm vẽ khớp (chấm tròn)
    const drawLandmarks = (points: Point[], color: string) => {
      ctx.fillStyle = color;
      points.forEach(p => {
        if (isValid(p)) {
          ctx.beginPath();
          ctx.arc(p.x * width, p.y * height, 2, 0, 2 * Math.PI);
          ctx.fill();
        }
      });
    };

    // --- VẼ ---
    // 1. Pose (Thân mình) - Màu trắng
    if (frame.pose) {
        drawConnectors(frame.pose, POSE_CONNECTIONS, "rgba(255, 255, 255, 0.7)");
        drawLandmarks(frame.pose, "white");
    }

    // 2. Tay trái - Màu Cam
    if (frame.left_hand) {
        drawConnectors(frame.left_hand, HAND_CONNECTIONS, "orange");
        drawLandmarks(frame.left_hand, "orange");
    }

    // 3. Tay phải - Màu Xanh Cyan
    if (frame.right_hand) {
        drawConnectors(frame.right_hand, HAND_CONNECTIONS, "cyan");
        drawLandmarks(frame.right_hand, "cyan");
    }
  };

  // 3. Animation Loop
  const animate = () => {
    if (frames.length === 0) return;

    const canvas = canvasRef.current;
    if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
            // Lấy frame hiện tại
            const currentFrame = frames[frameIndexRef.current];
            if (currentFrame) draw(ctx, currentFrame);
        }
    }

    // Tăng index, lặp lại nếu hết
    frameIndexRef.current = (frameIndexRef.current + 1) % frames.length;

    // Chỉnh tốc độ: setTimeout giúp video không chạy quá nhanh
    setTimeout(() => {
        requestRef.current = requestAnimationFrame(animate);
    }, 50); // 50ms ~ 20fps (Tốc độ vừa phải)
  };

  useEffect(() => {
    if (frames.length > 0) {
      requestRef.current = requestAnimationFrame(animate);
    }
    return () => cancelAnimationFrame(requestRef.current!);
  }, [frames]);

  return (
    <div style={{ textAlign: 'center', marginTop: '10px' }}>
        <canvas 
          ref={canvasRef} 
          width={width} 
          height={height}
          style={{ 
            background: '#1e1e1e', 
            borderRadius: '10px',
            border: '2px solid #333',
            boxShadow: '0 0 10px rgba(0,0,0,0.5)'
          }} 
        />
    </div>
  );
};

export default SkeletonPlayer;