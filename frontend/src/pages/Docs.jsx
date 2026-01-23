// Documentation
// ├─ Introduction
// ├─ Getting Started
// ├─ Moderation Categories
// ├─ Understanding Confidence Scores
// ├─ Multimodal Analysis
// ├─ Interpreting Results
// ├─ Limitations & Responsible Use
// ├─ Privacy & Data Usage
// └─ What’s Coming Next
import MarketingNavbar from "../components/layout/MarketingNavbar";
import Footer from "../components/layout/Footer";
import DocsSidebar from "../components/docs/DocsSidebar";
import DocsSection from "../components/docs/DocsSection";

export default function Docs() {
  return (
    <>
      <MarketingNavbar />

      <main className="pt-24 pb-32 bg-slate-50">
        <div className="mx-auto max-w-7xl px-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-10">

            {/* Sidebar */}
            <aside className="md:col-span-1">
              <DocsSidebar />
            </aside>

            {/* Content */}
            <section className="md:col-span-3">
              <div className="rounded-xl bg-white shadow-sm p-8 space-y-16">
                {/* Sections go here */}     
                <DocsSection id="introduction" title="Introduction">
                <p>
                    Aegis AI is an AI-powered video content moderation system designed
                    to help identify potentially harmful content at scale.
                </p>
                </DocsSection>

                <DocsSection id="getting-started" title="Getting Started">
                <p>
                    Upload a video, wait for analysis, and review the moderation results
                    including confidence scores and flagged timestamps.
                </p>
                </DocsSection>

                <DocsSection id="categories" title="Moderation Categories">
                <ul>
                    <li>Sexually Explicit Content</li>
                    <li>Violent Content</li>
                    <li>Hate Speech</li>
                    <li>Neutral (Safe Content)</li>
                </ul>
                </DocsSection>

                <DocsSection id="confidence-scores" title="Understanding Confidence Scores">
                <p>
                    Confidence scores represent the probability assigned by each model
                    to a specific category. They are not absolute judgments.
                </p>
                </DocsSection>

                <DocsSection id="multimodal-analysis" title="Multimodal Analysis">
                <p>
                    Aegis AI combines text, vision, and audio models to produce a more
                    reliable moderation signal.
                </p>
                </DocsSection>

                <DocsSection id="limitations" title="Limitations & Responsible Use">
                <p>
                    AI systems are not perfect and should be used alongside human review
                    for critical decisions.
                </p>
                </DocsSection>

                <DocsSection id="privacy" title="Privacy & Data Usage">
                <p>
                    Uploaded videos are processed for analysis and are not used for
                    training models.
                </p>
                </DocsSection>

                <DocsSection id="coming-next" title="What’s Coming Next">
                <ul>
                    <li>Public API access</li>
                    <li>Batch video processing</li>
                    <li>Improved explainability</li>
                </ul>
                </DocsSection>
              </div>
            </section>

          </div>
        </div>
      </main>

      <Footer />
    </>
  );
}
