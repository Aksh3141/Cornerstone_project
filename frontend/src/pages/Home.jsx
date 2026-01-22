import MarketingNavbar from "../components/layout/MarketingNavbar";
import UploadBox from "../components/upload/UploadBox";
import Footer from "../components/layout/Footer";

export default function Home() {
  return (
    <>
      <MarketingNavbar />

      {/* HERO SECTION */}
      <section className="bg-blue-50">
        <div className="relative z-10 mx-auto max-w-7xl px-6 pt-28 pb-16 text-center">

          {/* Context / trust line */}
          <div className="mb-6 flex justify-center">
            <span className="rounded-full border border-gray-300 bg-white px-4 py-1 text-xs text-gray-600">
              Research preview · No data stored · ML-powered
            </span>
          </div>

          {/* Headline */}
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-black mb-6">
            AI-powered video content moderation
          </h1>

          {/* Description (HOW, not WHAT) */}
          <p className="mx-auto max-w-2xl text-gray-600 text-lg leading-relaxed mb-6">
            Aegis AI uses a multimodal approach that combines vision, audio, and
            text-based models to analyze video content holistically. This enables
            more reliable and context-aware moderation decisions.
          </p>

          {/* Capability bullets (WHAT the user gets) */}
          <div className="mt-6 flex flex-wrap justify-center gap-x-10 gap-y-4 text-sm text-gray-600">
            <div className="flex items-center gap-2">
              <span className="text-brand">•</span>
              <span>Classify sensitive content across multiple categories</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-brand">•</span>
              <span>Identify exact timestamps of problematic segments</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-brand">•</span>
              <span>Generate confidence scores with a final verdict</span>
            </div>
          </div>

          {/* CTAs */}
          <div className="mt-10 flex justify-center gap-4">
            <button
              className="group relative overflow-hidden rounded-lg bg-black px-6 py-3 text-sm font-medium text-white transition-all duration-300 ease-out hover:-translate-y-px hover:shadow-[0_10px_30px_rgba(0,0,0,0.2)]"
              onClick={() =>
                document
                  .getElementById("upload-section")
                  ?.scrollIntoView({ behavior: "smooth" })
              }
            >
              <span className="relative z-10">Try it now</span>
              <span className="pointer-events-none absolute inset-0 bg-linear-to-r from-transparent via-white/10 to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
            </button>

            <a
              href="#how-it-works"
              className="rounded-lg border border-gray-300 bg-white px-6 py-3 text-sm font-medium text-gray-700 hover:border-black hover:text-black transition"
            >
              How it works
            </a>
          </div>
        </div>
      </section>

      {/* UPLOAD SECTION */}
      <section
        id="upload-section"
        className="px-6 pt-16 pb-24 bg-white"
      >
        <div className="mx-auto max-w-xl text-center mb-10">
          <h2 className="text-2xl font-semibold text-black mb-2">
            Analyze a video
          </h2>
          <p className="text-sm text-gray-600">
            Your video is analyzed temporarily and is not stored on our servers.
          </p>
        </div>

        <UploadBox />
      </section>
      
      {/* HOW IT WORKS SECTION */}
      <section
        id="how-it-works"
        className="px-6 pb-32 bg-white"
      >
        <div className="mx-auto max-w-6xl text-center">
          <h2 className="text-2xl font-semibold text-black mb-16">
            How it works
          </h2>

          <div className="flex flex-col md:flex-row items-center justify-center gap-6">

            {/* Step 1 */}
            <div className="w-full md:w-1/3 rounded-xl bg-gradient-to-br from-blue-50 to-blue-100 p-6 text-left shadow-sm">
              <div className="text-2xl mb-3">📤</div>
              <h3 className="text-base font-medium text-black mb-2">
                Upload a video
              </h3>
              <p className="text-sm text-gray-600">
                Upload a video file that you want to analyze for potentially sensitive
                or unsafe content.
              </p>
            </div>

            {/* Arrow */}
            <div className="hidden md:block text-gray-400 text-2xl">
              →
            </div>

            {/* Step 2 */}
            <div className="w-full md:w-1/3 rounded-xl bg-gradient-to-br from-blue-50 to-blue-100 p-6 text-left shadow-sm">
              <div className="text-2xl mb-3">🧠</div>
              <h3 className="text-base font-medium text-black mb-2">
                Multimodal analysis
              </h3>
              <p className="text-sm text-gray-600">
                The system analyzes visual frames, audio signals, and extracted text
                using specialized machine learning models.
              </p>
            </div>

            {/* Arrow */}
            <div className="hidden md:block text-gray-400 text-2xl">
              →
            </div>

            {/* Step 3 */}
            <div className="w-full md:w-1/3 rounded-xl bg-gradient-to-br from-blue-50 to-blue-100 p-6 text-left shadow-sm">
              <div className="text-2xl mb-3">📊</div>
              <h3 className="text-base font-medium text-black mb-2">
                Results & verdict
              </h3>
              <p className="text-sm text-gray-600">
                Receive confidence scores, flagged timestamps, and a clear moderation
                verdict for the video.
              </p>
            </div>

          </div>
        </div>
      </section>



      <Footer />
    </>
  );
}
