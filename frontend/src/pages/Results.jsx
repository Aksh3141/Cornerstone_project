import MarketingNavbar from "../components/layout/MarketingNavbar";
import Footer from "../components/layout/Footer";

const mockResults = {
  verdict: "Violent content detected",
  confidence: 0.82,
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

export default function Results() {
  return (
    <>
      <MarketingNavbar />

      <main className="px-6 pt-24 pb-32 bg-white">
        <div className="mx-auto max-w-5xl space-y-16">

          {/* VERDICT */}
          <section className="rounded-xl border border-red-200 bg-red-50 p-6">
            <h1 className="text-xl font-semibold text-red-700 mb-2">
              ⚠️ {mockResults.verdict}
            </h1>
            <p className="text-sm text-red-600">
              Overall confidence: {(mockResults.confidence * 100).toFixed(1)}%
            </p>
          </section>

          {/* CONFIDENCE SCORES */}
          <section>
            <h2 className="text-lg font-semibold text-black mb-6">
              Confidence scores
            </h2>

            <div className="space-y-4">
              {Object.entries(mockResults.scores).map(([label, value]) => (
                <div key={label}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="capitalize">{label}</span>
                    <span>{(value * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 rounded bg-gray-200">
                    <div
                      className="h-2 rounded bg-brand"
                      style={{ width: `${value * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* TIMELINE / SEGMENTS */}
          <section>
            <h2 className="text-lg font-semibold text-black mb-6">
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
                    className="flex justify-between rounded-lg border border-gray-200 p-4 text-sm"
                  >
                    <span>
                      {seg.start} – {seg.end}
                    </span>
                    <span className="font-medium text-red-600">
                      {seg.label}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </section>

          {/* ACTIONS */}
          <section className="flex gap-4">
            <a
              href="/"
              className="rounded-lg border border-gray-300 px-4 py-2 text-sm hover:border-black transition"
            >
              Analyze another video
            </a>
          </section>

        </div>
      </main>

      <Footer />
    </>
  );
}
