const sections = [
  { id: "introduction", label: "Introduction" },
  { id: "getting-started", label: "Getting Started" },
  { id: "categories", label: "Moderation Categories" },
  { id: "confidence-scores", label: "Confidence Scores" },
  { id: "multimodal-analysis", label: "Multimodal Analysis" },
  { id: "interpreting-results", label: "Interpreting Results" },
  { id: "limitations", label: "Limitations" },
  { id: "privacy", label: "Privacy & Data Usage" },
  { id: "coming-next", label: "What’s Coming Next" },
];

export default function DocsSidebar() {
  return (
    <nav className="md:sticky md:top-28">
      <h2 className="text-sm font-semibold text-gray-900 mb-4">
        Documentation
      </h2>

      <ul className="space-y-2 text-sm">
        {sections.map((s) => (
          <li key={s.id}>
            <a
              href={`#${s.id}`}
              className="text-gray-600 hover:text-black transition"
            >
              {s.label}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
