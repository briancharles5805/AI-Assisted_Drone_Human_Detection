# AI-Assisted_Drone_Human_Detection
The AI-Assisted Drone for Human Distress Detection is an aerial prototype system engineered to support search-and-rescue (SAR) operations in hazardous, natural, or man-made disaster environments by providing real time information such as  number of Humans identified, if they are safe or in distress, timely alerts and live gps coordinates in gmap
##  Hardware Architecture & Specifications

The drone platform is built upon a high-thrust propulsion blueprint engineered to sustain heavy edge-computing loads without undermining aerodynamic stability or response times[cite: 10].

| Component | Specification | Purpose |
| :--- | :--- | :--- |
| **Airframe** | F450 Quadcopter Frame[cite: 9, 10] | Lightweight structural chassis with integrated power distribution[cite: 10]. |
| **Flight Brain** | Pixhawk 2.4.8 (ArduCopter Firmware)[cite: 9, 10] | Manages core flight dynamics, sensor feedback, and Loiter hold[cite: 10]. |
| **Companion Computer** | Raspberry Pi 5 (8GB / 16GB)[cite: 9, 10] | Drives the onboard Python pipeline and handles neural network inference[cite: 9, 10]. |
| **Camera Module** | Arducam IMX519 (16MP)[cite: 9, 10] | Delivers raw video frames at high frequency (720p @ 80fps)[cite: 9, 10]. |
| **Propulsion Unit** | 1000KV BLDC Motors + 30A SimonK ESCs[cite: 9, 10] | Delivers an optimal 2.46:1 thrust-to-weight lift margin[cite: 10]. |
| **Power Plant** | 5200mAh 3S LiPo Battery (35C)[cite: 9, 10] | Supplies continuous high-current power through a 5V/3A BEC[cite: 10]. |
| **Telemetry & GPS** | Neo-7M GPS Module[cite: 9, 10] | Provides precise live localized latitude and longitude streaming[cite: 10]. |

---

## Software Stack & Edge Vision Pipeline

The system runs entirely at the edge to circumvent the latency penalties and structural hazards of cloud-dependent architectures in remote disaster zones[cite: 10].

* **Core Language:** Python 3.11+ (Multi-threaded processing to isolate frames and maintain non-blocking telemetry acquisition)[cite: 9].
* **Neural Framework:** Ultralytics YOLOv8-Pose (Quantized version optimized for hardware acceleration)[cite: 9, 10].
* **Protocols:** MAVLink / PyMAVLink for deep flight-controller interprocess telemetry communication, UDP for fast video streaming, and HTTP for remote API dispatching[cite: 9, 10].

### The Kinematic Inference Logic

   [ Camera Frame: 720p @ 80 FPS ]
                  │
                  ▼
     [ YOLOv8-Pose Extraction ] ──(No Person Detected)──► [ Loop Refresh ]
                  │
        (Confidence > 0.6)
                  ▼
 [ Joint Spatial Extraction (17 Keypoints) ]

(Extracts relative Y-coordinates of Wrists & Shoulders)
│
▼
[ Spatial Threshold Filter ]
Is Y_Wrist < Y_Shoulder (Both Arms Up)?
├── YES ──► [ Temporal Verification Loop (>1.0s) ] ──► [ TRIGGER DISTRESS ALERT ]
└── NO  ──► [ Tag Scene Safe / Maintain Patrol ]


---

##  Automated Alerting Workflow

When a victim matching the distress profile is confirmed, the system instantly generates an escape pipeline bypassing complex ground terminal screens[cite: 10]:

1. **Telemetry Intercept:** PyMAVLink queries the Pixhawk register to freeze the instantaneous latitude and longitude matrix[cite: 10].
2. **Dynamic URL Formatting:** The raw coordinates are parsed into an actionable, cross-platform Google Maps hyper-string: `https://www.google.com/maps?q=latitude,longitude`[cite: 10].
3. **Payload Compression:** A visual frame containing bounding box annotations (Red for Distress, Green for Safe) is isolated, compressed, and assigned a timestamp metadata header[cite: 9].
4. **API Gateway Transmission:** The asynchronous packet is fired using HTTP POST protocols via the Telegram Bot API to notify emergency field responders globally in under 3 seconds[cite: 9, 10].

---

##  Performance Matrix & Field Testing

Field validation runs performed at the Ramaiah Playground confirmed the performance stability of the system across structural altitude zones[cite: 9, 10]:

* **6 to 12 Meters:** **90% Detection & Classification Accuracy**. Skeletons are tightly pinned down with absolute geometric tracking stability[cite: 10].
* **12 to 16 Meters:** **76% Accuracy**. High altitude profiles introduce keypoint spatial jitter due to pixel pixelization

## Core Application Implementation

The primary execution and edge inference script is structured within the root source directory:

*  <a href="./Distressdetection.py" target="_blank">Distressdetection.py</a> — The production-ready Python deployment script executing multi-threaded processing layers at the edge. It concurrently coordinates high-frequency video capture loops, real-time 17-keypoint skeletal tracking, kinematic posture classification filters, and asynchronous PyMAVLink autopilot register telemetry queries.

---

###  Core Modules & Logic Streams

1. **Multi-Threaded Video Pipeline:** Captures raw visual inputs directly via high-fps camera interfaces, optimizing CPU memory allocation by segregating individual image frames into a clean, non-blocking inference queue.
2. **Kinematic Threshold Engine:** Parses real-time node strings output by the quantized YOLOv8-Pose model, calculating the exact spatial differences between wrists and shoulders to isolate valid rescue signals.
3. **Emergency Payload Broadcaster:** Intercepts live latitude and longitude streams via the local Pixhawk instrumentation layer to assemble dynamic Google Maps hyperlinks, routing them instantly alongside annotated image metadata buffers through the Telegram Bot API.
## Project Documentation

The complete engineering documentation and presentation materials are structured in the <a href="./Documentation/" target="_blank">Documentation</a> directory:

*  <a href="./Documentation/Human%20Distress%20Detection%20Thesis.pdf" target="_blank">Project Thesis</a> — A comprehensive engineering dissertation detailing the mathematical core (kinematic spatial tracking), quantized YOLOv8-Pose model optimization, and experimental field validation[cite: 10].
*  <a href="./Documentation/Human%20Detection%20presentation.pdf" target="_blank">Seminar Slides</a> — Technical presentation slides outlining the core hardware architecture (F450 chassis), Edge-AI inference loops, and real-time MAVLink telemetry alerting systems[cite: 9, 10].


