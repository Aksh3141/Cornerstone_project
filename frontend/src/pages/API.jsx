import MarketingNavbar from "../components/layout/MarketingNavbar";
import Footer from "../components/layout/Footer";

export default function API() {
  return (
    <>
      <MarketingNavbar />

      <main className="pt-24 pb-32 bg-slate-50">
        <div className="mx-auto max-w-4xl px-6">
          <div className="rounded-xl bg-white shadow-sm p-10 space-y-10">

            {/* Header */}
            <header className="space-y-3">
              <h1 className="text-3xl font-semibold text-black">
                Aegis AI API
              </h1>
              <p className="text-gray-600 max-w-2xl">
                Programmatic access to Aegis AI’s video content moderation
                system. Analyze videos at scale using the same multimodal
                models that power the web interface.
              </p>
            </header>

            {/* Coming Soon */}
            <section className="rounded-lg border border-dashed border-gray-300 bg-slate-50 p-6">
              <h2 className="text-lg font-semibold mb-2">
                🚧 Coming Soon
              </h2>
              <p className="text-sm text-gray-600">
                The Aegis AI API is currently under development and is not yet
                publicly available.
              </p>
            </section>

            {/* What the API will provide */}
            <section>
              <h2 className="text-xl font-semibold mb-4">
                Planned Capabilities
              </h2>

              <ul className="space-y-3 text-sm text-gray-700">
                <li>• REST-based video moderation endpoints</li>
                <li>• Multimodal analysis (text, vision, audio)</li>
                <li>• Confidence scores per moderation category</li>
                <li>• Timestamped flagged segments</li>
                <li>• Structured JSON responses</li>
              </ul>
            </section>

            {/* Output example (conceptual, not fake) */}
            <section>
              <h2 className="text-xl font-semibold mb-4">
                Response Structure
              </h2>

              <p className="text-sm text-gray-600 mb-4">
                API responses will closely mirror the results shown in the
                web interface, including verdicts, confidence scores, and
                flagged timestamps.
              </p>

              <div className="rounded-lg bg-gray-900 text-gray-100 text-sm p-4 overflow-x-auto">
                <pre>
{`{
  "verdict": "violent_content_detected",
  "confidence": 0.82,
  "modalities": {
    "text": { "violence": 0.78, "neutral": 0.10 },
    "vision": { "violence": 0.81, "neutral": 0.05 },
    "audio": { "violence": 0.45, "neutral": 0.50 }
  },
  "segments": [
    { "start": "00:12", "end": "00:18", "label": "violence" },
    { "start": "01:05", "end": "01:22", "label": "violence" }
  ]
}`}
                </pre>
              </div>

              <p className="mt-3 text-xs text-gray-500">
                Example shown for illustration only. Final schema may change.
              </p>
            </section>

            {/* CTA */}
            <section className="pt-6 border-t border-gray-200">
              <p className="text-sm text-gray-600 mb-4">
                Interested in early API access or integration?
              </p>

              <a
                href="https://github.com/Aksh3141/Cornerstone_project"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-block rounded-lg bg-black px-4 py-2 text-sm font-medium text-white hover:bg-gray-900 transition"
              >
                Follow development on GitHub
              </a>
            </section>

          </div>
        </div>
      </main>

      <Footer />
    </>
  );
}
