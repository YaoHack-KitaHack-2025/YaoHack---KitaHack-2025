# 🎙️ Spok.io - Your AI Public Speaking Coach

Welcome!  
This is **Spok.io**, developed by **YaoHack Team**, a smart web platform to help everyone improve their public speaking skills using AI posture detection, speech analysis, and community support.

> 💡 **Spok.io** helps you:
> - Analyze your posture in real-time ✅
> - Get instant feedback on your speaking clarity ✅
> - Practice with live video + speech feedback ✅
> - Join community forum for public speaking tips ✅

---

## 🚀 Features

- ✅ **AI Posture Detection** (using MediaPipe & OpenCV)
- ✅ **Speech Analysis** (Google Speech-to-Text API)
- ✅ **Real-time Video Feedback**
- ✅ **Google Gemini AI Integration** (for chatbot, feedback & future roadmap!)
- ✅ **Community Forum Demo Page**
- ✅ Clean & modern frontend design

---

## 🧩 Tech Stack

| Tech                   | Description                          |
|------------------------|--------------------------------------|
| Python                 | Backend server using Flask           |
| OpenCV                 | Real-time video frame analysis       |
| MediaPipe              | Pose estimation for posture feedback |
| Google Cloud Platform  | Speech-to-text & Vision API          |
| Google Gemini API      | AI responses and feedback            |
| HTML / CSS / JS        | Frontend templates                   |
| Flask                  | Web application framework            |

---

## ⚙️ Setup Instructions

### 1️⃣ Clone this repository

```bash
git clone https://github.com/YaoHack-KitaHack-2025/YaoHack---KitaHack-2025.git
cd spokio
```

### 2️⃣ Install Python dependencies

Make sure Python 3.x is installed.

```
pip install -r requirements.txt
```

> If you don’t have `requirements.txt`, create it manually:
>
> ```
> flask
> opencv-python
> mediapipe
> google-cloud-speech
> google-cloud-vision
> google-generativeai
> ```

### 3️⃣ Google Cloud Platform Setup

1. Create a project on **Google Cloud Platform (GCP)**.
2. Enable the following APIs:
   - ✅ Cloud Vision API
   - ✅ Cloud Speech-to-Text API

3. Generate a **Service Account JSON file**:
   - Go to **IAM & Admin > Service Accounts > Keys**
   - Create a new key (JSON), download the file.
   - Place it in your project folder.

4. **Edit your `app.py`**:
   Update the credentials path:
   ```python
   os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"YOUR_CREDENTIALS_FILE.json"
   ```

5. Add your **Gemini API Key** inside `app.py`:
   ```python
   genai.configure(api_key='YOUR_GEMINI_API_KEY')
   ```

### 4️⃣ Run the application

```
python app.py
```

Then, open your browser and visit:

```
http://127.0.0.1:5000/
```

🎉 Enjoy your AI public speaking coach!

---

## 🌐 Pages Available

| Page      | URL                  | Description                             |
|-----------|----------------------|-----------------------------------------|
| Home      | `/`                  | Welcome landing page                    |
| AI Coach  | `/ai_coach`          | Live posture + speech feedback          |
| Real-Time | `/real_time`         | Real-time video feedback page           |
| Forum     | `/forum`             | Community forum page (demo)             |

---

## 🤖 Future Improvements

- ✅ Chatbot frontend integration
- ✅ Forum database for real posts & replies
- ✅ Data export & progress dashboard
- ✅ Dark mode & animations
- ✅ Camera switch & reconnect options

---

## 🤝 Contributing

We welcome contributions!  
If you have ideas to improve Spok.io, feel free to fork the repo and submit a pull request.

1. Fork this repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a Pull Request

---

## 📢 Credits

Made with 💙 by **YaoHack Team**
2025 - UTM x Spok.io Project

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
```
