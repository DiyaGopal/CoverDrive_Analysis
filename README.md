# 🏏 AthleteRise – Real-Time Cover Drive Analysis

This project analyzes a cricket **cover drive shot** in real-time using **OpenCV** and **MediaPipe Pose Estimation**.  
It generates an annotated video with pose skeleton and live metric overlays, plus a JSON evaluation with scores and feedback.  
A bonus elbow angle plot is also provided for smoothness analysis.

---

## ⚙️ Setup & Run Instructions

1. **Install Python 3.9 – 3.11**  
   (Recommended: Python 3.10 for compatibility with OpenCV and MediaPipe)

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # On Windows
   source .venv/bin/activate   # On Linux/Mac
   
3. **Upgrade pip and install dependencies**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt

4. **Download the input video**
   ```bash
   python -m yt_dlp -o input_video.mp4 https://youtube.com/shorts/vSX3IRxGnNY

5. **Run the analysis**
```bash
python cover_drive_analysis.py
```

---

## 📝 Notes on Assumptions / Limitations

- **Python Compatibility:** Tested on Python 3.10. Other versions (3.9–3.11) should work. Python 3.12+ may not have full library support.  
- **Pose Estimation:** Uses MediaPipe Pose. Accuracy may decrease if the player’s body parts are occluded or partially visible.  
- **Metrics:** Computed values (elbow angle, spine lean, head-knee alignment, foot direction) are approximate and based on 2D video frames, not true biomechanical-grade measurements.  
- **Environment:** Requires stable internet connection if using `yt-dlp` for video download.  
- **Performance:** Achieves ~10–15 FPS on CPU; real-time speed depends on system hardware.  
- **Scoring & Feedback:** Evaluation scores are heuristic-based and intended for demonstration, not professional coaching accuracy.
