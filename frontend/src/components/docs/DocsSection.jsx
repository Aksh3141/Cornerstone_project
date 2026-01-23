export default function DocsSection({ id, title, children }) {
  return (
    <section id={id} className="scroll-mt-28">
      <h2 className="text-2xl font-semibold text-black mb-4">
        {title}
      </h2>
      <div className="prose prose-slate max-w-none">
        {children}
      </div>
    </section>
  );
}
