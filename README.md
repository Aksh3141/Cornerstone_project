# 🎥 Cornerstone Video Moderation Project

A full-stack application designed to analyze and moderate video content using a React frontend and a Django-powered Machine Learning backend.

---

## 🏗️ Project Structure

This repository contains both the frontend and backend source code.

### 💻 Frontend (React + Vite + Tailwind v4)
```text
frontend/
├── public/              # Static assets
├── src/
│   ├── assets/          # Global images and styles
│   ├── components/      # UI Building blocks
│   │   ├── common/      # Buttons, Inputs, Loaders
│   │   ├── upload/      # Video upload logic
│   │   ├── results/     # Analysis display
│   │   ├── video/       # Video player components
│   │   └── layout/      # Navbar and Footer
│   ├── pages/           # Home & Analyze views
│   ├── services/        # API calls (Django integration)
│   ├── hooks/           # Custom hooks (useVideoAnalysis)
│   ├── utils/           # Helpers (formatTime)
│   ├── App.jsx          # Main Routing
│   └── main.jsx         # Entry point
└── package.json         # Dependencies
```

### ⚙️ Backend (Django + REST Framework)
```text
backend/
├── manage.py            # Django CLI tool
├── requirements.txt     # Python dependencies
├── config/              # Project settings and routing
├── moderation/          # Main application logic
│   ├── models.py        # Database schema for videos/results
│   ├── views.py         # API endpoints
│   ├── serializers.py   # Data transformation logic
│   ├── ml/              # Machine Learning integration
│   │   ├── model.py     # ML Model architecture
│   │   └── inference.py # Prediction logic
│   └── tasks.py         # Background processing tasks
└── media/               # User-uploaded content (Ignored by Git)
```

### 🚀 System Architecture & Flow
The following diagram represents the end-to-end data flow when a user interacts with the platform:
```text
graph TD
    A[User Selects Video] --> B[React: POST Request]
    B --> C[Django: Save Video to Media]
    C --> D[ML: Inference Engine]
    D --> E[Django: Save Results to DB]
    E --> F[API: Return JSON Response]
    F --> G[React: Update UI & Display Results]
```
