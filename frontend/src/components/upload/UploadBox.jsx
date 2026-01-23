import { useState } from "react";
import { useNavigate } from "react-router-dom";

const MAX_FILE_SIZE_MB = 100;
const ALLOWED_TYPES = ["video/mp4", "video/webm", "video/ogg"];

export default function UploadBox({ compact = false }) {
  const [file, setFile] = useState(null);
  const [error, setError] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const navigate = useNavigate();


  const handleFileChange = (e) => {
    const selected = e.target.files[0];

    setError("");

    if (!selected) {
      setFile(null);
      return;
    }

    // Type validation
    if (!ALLOWED_TYPES.includes(selected.type)) {
      setError("Only MP4, WebM, or OGG video files are allowed.");
      setFile(null);
      return;
    }

    // Size validation
    const sizeMB = selected.size / (1024 * 1024);
    if (sizeMB > MAX_FILE_SIZE_MB) {
      setError("File size must be under 100MB.");
      setFile(null);
      return;
    }

    setFile(selected);
  };

  const handleUpload = () => {
  if (!file) return;

  // ✅ For now: directly go to results page
  navigate("/results");
};

  return (
    <div
      className={`w-full max-w-xl mx-auto rounded-xl bg-white shadow-sm ${
        compact
          ? "p-4"
          : "p-6 border border-gray-200"
      }`}
    >
      <h3
        className={`font-semibold ${
          compact ? "text-base mb-3" : "text-lg mb-4"
        }`}
      >
        Upload a video
      </h3>


      <label className="flex flex-col items-center justify-center gap-2 border-2 border-dashed rounded-lg p-6 cursor-pointer hover:border-gray-400 transition">
        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          className="hidden"
        />

        <span className="text-sm text-gray-600">
          {file ? file.name : "Click to select a video file"}
        </span>

        <span className="text-xs text-gray-400">
          MP4, WebM, OGG (max 100MB)
        </span>
      </label>

      {error && (
        <p className="mt-3 text-sm text-red-600">{error}</p>
      )}

      <button
      onClick={handleUpload}
      disabled={!file || isUploading}
      className={`mt-4 w-full rounded-lg font-medium text-white transition
        ${
          compact
            ? "px-4 py-2 text-sm"
            : "px-6 py-3 text-base"
        }
        ${
          !file || isUploading
            ? "bg-gray-400 cursor-not-allowed"
            : "bg-black hover:bg-gray-900"
        }`}
    >
      {isUploading ? "Analyzing video..." : "Upload & Analyze"}
    </button>

    </div>
  );
}
