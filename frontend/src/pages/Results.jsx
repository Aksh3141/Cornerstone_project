import MarketingNavbar from "../components/layout/MarketingNavbar";
import Footer from "../components/layout/Footer";
import UploadBox from "../components/upload/UploadBox";

const mockResults = {
  verdict: "Violent content detected",
  confidence: 0.82,
  harmful: true, // future-proof flag
  scores: {
    sexual: 0.08,
    hate: 0.12,
    violence: 0.82,
    neutral: 0.15,
  },
  segments: [
    { start: "00:12", end: "00:18", label: "Violence" },
    { start: "01:05", end: "01:22", label: "Violence" },
  ],
};
const CLASS_COLORS = {
  sexual: "#ec4899",   // pink
  violence: "#8B5CF6", 
  hate: "#06b6d4",     // orange
  neutral: "#84cc16",  // green
};
const CLASS_COLORS2 = {
  sexual: "#7c3aed",   // pink
  violence: "#2dd4bf", 
  hate: "#fb923c",     // orange
  neutral: "#375569",  // green
};

const mockModalities = {
  text: {
    sexual: 0.05,
    violence: 0.78,
    hate: 0.07,
    neutral: 0.10,
  },
  vision: {
    sexual: 0.12,
    violence: 0.81,
    hate: 0.02,
    neutral: 0.05,
  },
  audio: {
    sexual: 0.01,
    violence: 0.45,
    hate: 0.04,
    neutral: 0.50,
  },
};

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

    const startX =
      center + radius * Math.cos((Math.PI / 180) * startAngle);
    const startY =
      center + radius * Math.sin((Math.PI / 180) * startAngle);

    const endX =
      center + radius * Math.cos((Math.PI / 180) * endAngle);
    const endY =
      center + radius * Math.sin((Math.PI / 180) * endAngle);

    const pathData = `
      M ${center} ${center}
      L ${startX} ${startY}
      A ${radius} ${radius} 0 ${largeArcFlag} 1 ${endX} ${endY}
      Z
    `;

    return (
      <path
        key={label}
        d={pathData}
        fill={CLASS_COLORS[label]}
      />
    );
  });

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {slices}
      {/* inner hole */}
      <circle
        cx={center}
        cy={center}
        r={radius - thickness}
        fill="white"
      />
    </svg>
  );
}



export default function Results() {
  return (
    <>
      <MarketingNavbar />

      <main className="px-6 pt-24 pb-32 bg-slate-50">
        <div className="mx-auto max-w-5xl space-y-16">

          {/* VERDICT */}
          <section
            className={`rounded-xl border p-6 ${
              mockResults.harmful
                ? "border-red-200 bg-red-50"
                : "border-green-200 bg-green-50"
            }`}
          >
            <h1
              className={`text-xl font-semibold mb-2 ${
                mockResults.harmful ? "text-red-700" : "text-green-700"
              }`}
            >
              {mockResults.harmful ? "⚠️" : "✅"} {mockResults.verdict}
            </h1>
            <p
              className={`text-sm ${
                mockResults.harmful ? "text-red-600" : "text-green-600"
              }`}
            >
              Overall confidence: {(mockResults.confidence * 100).toFixed(1)}%
            </p>
          </section>

          {/* VIDEO + FLAGGED SEGMENTS */}
          <section className="grid grid-cols-1 md:grid-cols-3 gap-8">

            {/* VIDEO PLACEHOLDER */}
            <div className="md:col-span-2">
              <div className="aspect-video rounded-xl border border-dashed border-gray-300 bg-gray-50 flex items-center justify-center text-sm text-gray-500">
                Uploaded video preview will appear here
              </div>
            </div>

            {/* FLAGGED SEGMENTS */}
            <div>
              <h2 className="text-sm font-semibold text-black mb-4">
                Flagged segments
              </h2>

              {mockResults.segments.length === 0 ? (
                <p className="text-sm text-gray-500">
                  No problematic segments detected.
                </p>
              ) : (
                <ul className="space-y-3">
                  {mockResults.segments.map((seg, idx) => (
                    <li
                    key={idx}
                    className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm flex items-center justify-between gap-3"
                    >
                    <div className="flex flex-col">
                        <span className="text-gray-700">
                        {seg.start} – {seg.end}
                        </span>
                        <span className="text-xs font-medium text-red-600">
                        {seg.label}
                        </span>
                    </div>

                    <button
                        type="button"
                        className="flex items-center justify-center h-8 w-8 rounded-md border border-gray-300 text-gray-700 hover:border-black hover:text-black transition"
                        aria-label={`Play segment starting at ${seg.start}`}
                    >
                        ▶
                    </button>
                    </li>

                  ))}
                </ul>
              )}
            </div>
          </section>

          {/* CONFIDENCE SCORES */}
            <section>
            <h2 className="text-lg font-semibold text-black mb-6">
                Confidence by modality
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">

                {Object.entries(mockModalities).map(([modality, scores]) => (
                <div
                    key={modality}
                    className="rounded-xl bg-white p-6 flex flex-col items-center shadow-sm hover:shadow-md transition-shadow"
                >
                    <h3 className="text-sm font-semibold mb-4 capitalize">
                    {modality} model
                    </h3>

                    <div className="flex items-center justify-center py-4">
                    <PieChart data={scores} />
                    </div>

                    <ul className="mt-4 space-y-1 text-xs w-full">
                    {Object.entries(scores).map(([label, value]) => (
                        <li key={label} className="flex justify-between">
                        <span className="capitalize flex items-center gap-2">
                            <span
                            className="h-2 w-2 rounded-full"
                            style={{ backgroundColor: CLASS_COLORS[label] }}
                            />
                            {label}
                        </span>
                        <span>{(value * 100).toFixed(1)}%</span>
                        </li>
                    ))}
                    </ul>
                </div>
                ))}

            </div>
            </section>


          {/* ANALYZE ANOTHER VIDEO */}
        <section className="pt-12 border-t border-gray-200">
        <div className="max-w-xl">
            <h2 className="text-base font-semibold text-black mb-2">
            Analyze another video
            </h2>
            <p className="text-sm text-gray-600 mb-6">
            Upload a new video to run content moderation analysis again.
            </p>

            {/* Reuse UploadBox component */}
            <UploadBox compact />
        </div>
        </section>

        </div>
      </main>

      <Footer />
    </>
  );
}
