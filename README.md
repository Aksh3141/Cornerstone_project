# Cornerstone_project
🏗️ Project Structure
This project is divided into a React (Vite) frontend and a Django backend.

💻 Frontend
frontend/
├── public/              # Static assets (favicons, etc.)
├── src/
│   ├── assets/          # Images, global styles
│   ├── components/      # Reusable UI components
│   │   ├── common/      # Buttons, Inputs, Loaders
│   │   ├── upload/      # Video upload logic
│   │   ├── results/     # Analysis display components
│   │   ├── video/       # Video player components
│   │   └── layout/      # Navbar, Footer, Wrappers
│   ├── pages/           # Main view components (Home, Analyze)
│   ├── services/        # API communication (Django integration)
│   ├── hooks/           # Custom React hooks (useVideoAnalysis)
│   ├── utils/           # Helper functions (formatTime)
│   ├── App.jsx          # Main application routing
│   └── main.jsx         # Application entry point
├── .env                 # Environment variables (Ignored by Git)
└── package.json         # Frontend dependencies

⚙️ Backend
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

🚀 System Architecture & Flow
The following diagram represents the end-to-end data flow when a user interacts with the platform:
graph TD
    A[User Selects Video] --> B[React: POST Request]
    B --> C[Django: Save Video to Media]
    C --> D[ML: Inference Engine]
    D --> E[Django: Save Results to DB]
    E --> F[API: Return JSON Response]
    F --> G[React: Update UI & Display Results]