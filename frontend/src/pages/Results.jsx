import MarketingNavbar from "../components/layout/MarketingNavbar";
import Footer from "../components/layout/Footer";
import UploadBox from "../components/upload/UploadBox";
import { useLocation } from "react-router-dom";
import { useEffect, useRef, useState } from "react";

const CLASS_COLORS = {
  sexual: "#ec4899",
  violence: "#8B5CF6",
  hate: "#06b6d4",
  neutral: "#84cc16",
};

const mapScores = (scores = {}) => ({
  sexual: scores.sexual_content || 0,
  violence: scores.violence || 0,
  hate: scores.hate_speech || 0,
  neutral: scores.neutral || 0,
});

function PieChart({ data, size = 120, thickness = 22 }) {
  const radius = (size - thickness) / 2;
  const center = size / 2;
  let cumulativeAngle = 0;

  const slices = Object.entries(data).map(([label, value]) => {
    const angle = value * 360;
    const startAngle = cumulativeAngle;
    const endAngle = cumulativeAngle + angle;
    cumulativeAngle += angle;

    const largeArcFlag = angle > 180 ? 1 : 0;

    const startX = center + radius * Math.cos((Math.PI / 180) * startAngle);
    const startY = center + radius * Math.sin((Math.PI / 180) * startAngle);

    const endX = center + radius * Math.cos((Math.PI / 180) * endAngle);
    const endY = center + radius * Math.sin((Math.PI / 180) * endAngle);

    const pathData = `
      M ${center} ${center}
      L ${startX} ${startY}
      A ${radius} ${radius} 0 ${largeArcFlag} 1 ${endX} ${endY}
      Z
    `;

    return <path key={label} d={pathData} fill={CLASS_COLORS[label]} />;
  });

  return (
    <svg width={size} height={size}>
      {slices}
      <circle cx={center} cy={center} r={radius - thickness} fill="white" />
    </svg>
  );
}

export default function Results() {
  const location = useLocation();
  const videoRef = useRef(null);
  const currentRequestId = useRef(0);

  const { videoURL, file } = location.state || {};

  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(!!file);

  useEffect(() => {
    if (!file) return;
    setResults(null);
    setLoading(true);
  }, [file]);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [location.key]);

  useEffect(() => {
    if (!file) return;

    const requestId = ++currentRequestId.current;

    const runAnalysis = async () => {
      try {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch("http://localhost:8000/api/moderate-video", {
          method: "POST",
          body: formData,
        });

        const data = await res.json();
        console.log("API DATA:", data);

        // ✅ Only update if this is latest request
        if (requestId === currentRequestId.current) {
          setResults(data);
        }

      } catch (err) {
        console.error(err);
      } finally {
        // ✅ Only stop loading if latest request
        if (requestId === currentRequestId.current) {
          setLoading(false);
        }
      }
    };

    runAnalysis();
  }, [file]);

  const data = results;
  const segments = data?.segments || [];
  const modalities = data?.modalities || {};
  const finalScores = data?.final_scores || {};
  const finalLabel = data?.verdict || "neutral";

  return (
    <>
      <MarketingNavbar />

      <main className="px-6 pt-24 pb-32 bg-slate-50">
        <div className="mx-auto max-w-5xl space-y-16">

          {/* VIDEO */}
          <section className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="md:col-span-2">
              <div className="aspect-video rounded-xl bg-black overflow-hidden">
                {videoURL && (
                  <video
                    ref={videoRef}
                    src={videoURL}
                    controls
                    className="w-full h-full"
                  />
                )}
              </div>
            </div>

            <div>
              <h2 className="text-sm font-semibold mb-4">
                Flagged segments
              </h2>
              <p className="text-sm text-gray-500">(Coming soon)</p>
            </div>
          </section>

          {/* LOADING */}
          {loading && (
            <div className="text-center py-10">
              <p className="text-lg font-medium">Analyzing video...</p>
              <p className="text-sm text-gray-500">
                This may take a few seconds
              </p>
            </div>
          )}

          {/* RESULTS */}
          {!loading && data && (
            <>
              {/* FINAL + TRANSCRIPT */}
              <section className="grid grid-cols-1 md:grid-cols-2 gap-8">

                <div className={`rounded-xl p-6 shadow-sm border ${
                  finalLabel === "neutral"
                    ? "border-green-300 bg-green-50"
                    : "border-red-300 bg-red-50"
                }`}>
                  <h2 className="text-sm font-semibold mb-2">
                    Final classification
                  </h2>

                  <div className="text-2xl font-semibold capitalize mb-4">
                    {finalLabel}
                  </div>

                  {Object.entries(mapScores(finalScores)).map(([label, value]) => (
                    <div key={label} className="mb-2">
                      <div className="flex justify-between text-sm">
                        <span>{label}</span>
                        <span>{(value * 100).toFixed(1)}%</span>
                      </div>

                      <div className="w-full h-2 bg-gray-200 rounded-full">
                        <div
                          className="h-full bg-gray-500 rounded-full"
                          style={{ width: `${value * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                <div className="rounded-xl bg-white p-6 shadow-sm">
                  <h2 className="text-sm font-semibold mb-3">
                    Transcript
                  </h2>
                  <div className="text-sm max-h-40 overflow-y-auto">
                    {data.transcript || "No transcript available"}
                  </div>
                </div>

              </section>

              {/* MODALITIES */}
              <section>
                <h2 className="text-lg font-semibold mb-6">
                  Confidence by modality
                </h2>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                  {Object.entries(modalities).map(([mod, scores]) => {
                    if (!scores || Object.keys(scores).length === 0) return null;

                    const mapped = mapScores(scores);

                    return (
                      <div
                        key={mod}
                        className="rounded-xl bg-white p-6 flex flex-col items-center shadow-sm"
                      >
                        <h3 className="text-sm font-semibold mb-4 capitalize">
                          {mod} model
                        </h3>

                        {/* PIE */}
                        <PieChart data={mapped} />

                        {/* 🔥 LABELS + BARS */}
                        <ul className="mt-4 space-y-2 text-xs w-full">
                          {Object.entries(mapped).map(([label, value]) => (
                            <li key={label} className="space-y-1">

                              {/* label + % */}
                              <div className="flex justify-between">
                                <span className="capitalize flex items-center gap-2">
                                  <span
                                    className="h-2 w-2 rounded-full"
                                    style={{ backgroundColor: CLASS_COLORS[label] }}
                                  />
                                  {label}
                                </span>
                                <span>{(value * 100).toFixed(1)}%</span>
                              </div>

                              {/* bar */}
                              <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                                <div
                                  className="h-full rounded-full"
                                  style={{
                                    width: `${value * 100}%`,
                                    backgroundColor: CLASS_COLORS[label],
                                  }}
                                />
                              </div>

                            </li>
                          ))}
                        </ul>
                      </div>
                    );
                  })}
                </div>
              </section>

              {/* SEGMENTS */}
              <section>
                <h2 className="text-lg font-semibold mb-6">
                  Segment Analysis
                </h2>

                {segments.length === 0 ? (
                  <p className="text-sm text-gray-500">No segments available</p>
                ) : (
                  <div className="space-y-4">
                    {segments.map((seg, i) => (
                      <div key={i} className="bg-white p-4 rounded-xl border">
                        <div className="text-xs text-gray-500 mb-2">
                          {seg.start?.toFixed(1)}s - {seg.end?.toFixed(1)}s
                        </div>

                        <div className="text-sm mb-3">{seg.text}</div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                          {Object.entries(seg.modalities || {}).map(([mod, scores]) => (
                            <div key={mod}>
                              <div className="font-semibold mb-1 capitalize">{mod}</div>
                              {Object.entries(scores).map(([k, v]) => (
                                <div key={k} className="flex justify-between">
                                  <span>{k}</span>
                                  <span>{(v * 100).toFixed(1)}%</span>
                                </div>
                              ))}
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            </>
          )}

          {/* UPLOAD AGAIN */}
          <section className="pt-12 border-t">
            <UploadBox compact />
          </section>

        </div>
      </main>

      <Footer />
    </>
  );
}