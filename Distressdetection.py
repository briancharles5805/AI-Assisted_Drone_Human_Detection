import cv2
import numpy as np
import time
import threading
import requests
import os
import sys
import subprocess
import signal

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "yolov8n-pose.onnx"
CONF_THRESHOLD = 0.45
TELEGRAM_TOKEN = "your token"
CHAT_ID = "your id"

# Gesture logic
GESTURE_HOLD_SEC = 1.0 
MAX_ALERTS_PER_EVENT = 3

class SafetyState:
    def __init__(self):
        self.running = True
        self.gesture_start = 0
        self.alert_active = False
        self.count = 0

state = SafetyState()

def signal_handler(sig, frame):
    print("\n[System] Shutdown signal received. Cleaning up resources...")
    state.running = False

signal.signal(signal.SIGINT, signal_handler)

# ==========================================
# 2. OPTIMIZED KEYPOINT PARSING & HB2 LOGIC
# ==========================================
def parse_keypoints(predictions, frame_shape):
    """
    Parses keypoints from YOLOv8-pose prediction tensor outputs.
    Scales coordinates back to the visual frame dimensions.
    """
    h, w = frame_shape
    scores = predictions[:, 4]
    idx = np.where(scores > CONF_THRESHOLD)[0]
    if len(idx) == 0:
        return None
        
    best_idx = idx[np.argmax(scores[idx])]
    kpts = predictions[best_idx, 5:].reshape((17, 3))
    
    # Scale x, y relative to original frame coordinates
    kpts[:, 0] *= (w / 640)
    kpts[:, 1] *= (h / 640)
    
    return kpts

def check_gesture_hb2(kpts):
    """
    HB2 Logic: Hand-to-Body-Ratio Detection
    Uses torso height (shoulders to hips) to create a dynamic threshold for 'Hands Up'.
    This avoids distance-based false positives.
    """
    # Keypoint mappings:
    # 5,6: Shoulders | 11,12: Hips | 9,10: Wrists
    l_sh, r_sh = kpts[5], kpts[6]
    l_hip, r_hip = kpts[11], kpts[12]
    l_wrist, r_wrist = kpts[9], kpts[10]
    
    # Verify core coordinates have enough detection confidence
    required_pts = [l_sh, r_sh, l_wrist, r_wrist]
    if any(pt[2] < 0.5 for pt in required_pts):
        return False
        
    # 1. Calculate torso height reference scale
    avg_shoulder_y = (l_sh[1] + r_sh[1]) / 2
    avg_hip_y = (l_hip[1] + r_hip[1]) / 2 if (l_hip[2] > 0.4 and r_hip[2] > 0.4) else avg_shoulder_y + 100
    torso_height = abs(avg_hip_y - avg_shoulder_y)
    
    # 2. HB2 Threshold (Wrists must be at least 20% of torso height above shoulders)
    hb2_threshold = avg_shoulder_y - (torso_height * 0.2)
    
    # 3. Gesture State Evaluation
    is_left_up = l_wrist[1] < hb2_threshold
    is_right_up = r_wrist[1] < hb2_threshold
    
    return is_left_up and is_right_up

# ==========================================
# 3. TELEGRAM ALERTING
# ==========================================
def send_alert(img_path):
    """Sends the saved capture over to Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        with open(img_path, 'rb') as f:
            data = {'chat_id': CHAT_ID, 'caption': "🚨 GREG ALERT: Panic Gesture (HB2) Detected!"}
            requests.post(url, data=data, files={'photo': f}, timeout=15)
        if os.path.exists(img_path):
            os.remove(img_path)
    except Exception as e:
        print(f"[Network] Alert failed: {e}")

# ==========================================
# 4. CAMERA INITIALIZATION (RPI5 NATIVE PIPE)
# ==========================================
def get_libcamera_stream():
    """
    Launches a native 'libcamera-vid' process to handle the CSI camera,
    streaming continuous MJPEG chunks directly over stdout.
    """
    print("[System] Opening native libcamera pipe...")
    cmd = [
        'libcamera-vid',
        '-t', '0', 
        '--width', '640', 
        '--height', '480', 
        '--inline', 
        '--nopreview', 
        '--codec', 'mjpeg', 
        '--flush',
        '-o', '-'
    ]
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        return process
    except Exception as e:
        print(f"[!] Pipe Error: {e}")
        return None

# ==========================================
# 5. MAIN LOOP
# ==========================================
def main():
    print("[System] Initializing Safety Suite (HB2 Proportional Logic)...")
    
    # Clean up background system camera locks
    subprocess.run(['sudo', 'pkill', '-9', 'libcamera'], capture_output=True)
    subprocess.run(['sudo', 'pkill', '-9', 'gst-launch-1.0'], capture_output=True)
    
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Error: {MODEL_PATH} not found in current directory.")
        return

    # Set up OpenCV to utilize optimized CPU-only backends
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    pipe_process = get_libcamera_stream()
    if not pipe_process:
        print("[!] CRITICAL: Could not start libcamera-vid.")
        return

    print("[System] Processing started. Monitoring for HB2 gestures...")
    
    buffer = b""
    
    try:
        while state.running:
            chunk = pipe_process.stdout.read(4096)
            if not chunk: break
            buffer += chunk
            
            # Find boundaries of the current JPEG frame in the incoming stream
            a = buffer.find(b'\xff\xd8')
            b = buffer.find(b'\xff\xd9')
            
            if a != -1 and b != -1:
                jpg = buffer[a:b+2]
                buffer = buffer[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    start_time = time.time()
                    
                    # Convert to OpenCV blob (640x640 input resolution format)
                    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
                    net.setInput(blob)
                    output = net.forward()
                    
                    preds = np.transpose(output[0])
                    kpts = parse_keypoints(preds, frame.shape[:2])
                    
                    if kpts is not None:
                        # Draw keypoint circles for debugging visual feedback
                        for i in range(17):
                            x, y, conf = kpts[i]
                            if conf > 0.5:
                                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)
                        
                        # Process spatial relationships
                        if check_gesture_hb2(kpts):
                            if state.gesture_start == 0:
                                state.gesture_start = time.time()
                            
                            # Debounce gesture over the hold period
                            if (time.time() - state.gesture_start) >= GESTURE_HOLD_SEC:
                                if not state.alert_active:
                                    print("🚨 HB2 GESTURE TRIGGERED")
                                    state.alert_active = True
                                
                                # Burst mode limits
                                if state.count < MAX_ALERTS_PER_EVENT:
                                    img_name = f"alert_{int(time.time())}.jpg"
                                    cv2.imwrite(img_name, frame)
                                    state.count += 1
                                    threading.Thread(target=send_alert, args=(img_name,)).start()
                        else:
                            state.gesture_start = 0
                            state.alert_active = False
                            state.count = 0

                    fps = 1.0 / (time.time() - start_time)
                    cv2.putText(frame, f"HB2 FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("GREG System", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    finally:
        state.running = False
        if pipe_process: 
            pipe_process.terminate()
        cv2.destroyAllWindows()
        print("[System] Shutdown complete.")

if __name__ == "__main__":
    main()