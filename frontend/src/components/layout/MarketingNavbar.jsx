import { useEffect, useState } from "react";
import logoSmall from "../../assets/logo-small.png";
import { Link, useLocation } from "react-router-dom";


const NAV_ITEMS = [
  { id: "how-it-works", label: "How it works", type: "scroll" },
  { id: "api", label: "API", type: "route", path: "/api" },
  { id: "docs", label: "Docs", type: "route", path: "/docs" },
];

export default function Navbar() {
  const [activeSection, setActiveSection] = useState(null);
  const [hasShadow, setHasShadow] = useState(false);
  const location = useLocation();
  
  // Scroll shadow
  useEffect(() => {
    const handleScroll = () => {
      setHasShadow(window.scrollY > 8);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Active section observer
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      {
        rootMargin: "-40% 0px -50% 0px",
        threshold: 0,
      }
    );

    NAV_ITEMS.forEach(({ id }) => {
      const section = document.getElementById(id);
      if (section) observer.observe(section);
    });

    return () => observer.disconnect();
  }, []);

  return (
    <header
      className={`sticky top-0 z-50 bg-white border-b border-gray-200 transition-shadow ${
        hasShadow ? "shadow-sm" : ""
      }`}
    >
      <nav
        className={`mx-auto max-w-7xl px-6 flex items-center justify-between transition-all duration-300 ${
          hasShadow ? "py-2" : "py-4"
        }`}
      >


        {/* Brand */}
        <a
          href="#"
          className="flex items-center gap-3 hover:opacity-90 transition"
        >
          <img
            src={logoSmall}
            alt="Aegis AI logo"
            className="h-8 w-8 object-contain"
          />
          <span className="text-lg font-semibold tracking-tight text-black">
            Aegis AI
          </span>
        </a>


        {/* Navigation */}
        <ul className="hidden md:flex items-center gap-8 text-sm">
          {NAV_ITEMS.map(({ id, label, type, path }) => (
            <li key={id} className="relative group">
              {type === "scroll" ? (
              <a
                href={`#${id}`}
                className={`pb-1 transition ${
                  activeSection === id
                    ? "text-black"
                    : "text-gray-600 group-hover:text-black"
                }`}
              >
                {label}
              </a>
            ) : (
              <Link
                to={path}
                className={`pb-1 transition ${
                  location.pathname === path
                    ? "text-black"
                    : "text-gray-600 hover:text-black"
                }`}
              >
              {label}
              </Link>
            )}

              <span
                className={`
                absolute left-0 -bottom-1 h-0.5 transition-all duration-300
                ${
                  type === "scroll" && activeSection === id
                    ? "w-full bg-brand"
                    : "w-0 bg-black group-hover:w-full"
                }
              `}
              />
            </li>
          ))}

        {/* External GitHub */}
        <li className="relative group">
          <a
            href="https://github.com/Aksh3141/Cornerstone_project"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 pb-1 text-gray-600 transition group-hover:text-black"
          >
            GitHub <span className="text-xs">↗</span>
          </a>
          <span
            className="
              absolute left-0 -bottom-1 h-0.5
              w-0 bg-black
              transition-all duration-300
              group-hover:w-full
            "
          />
        </li>

      </ul>


        {/* CTA */}
        <button
          className="group relative overflow-hidden rounded-lg bg-black px-4 py-2 text-sm font-medium text-white transition-all duration-300 ease-out hover:-translate-y-px hover:shadow-[0_8px_24px_rgba(0,0,0,0.18)] active:translate-y-0"
          onClick={() =>
                document
                  .getElementById("upload-section")
                  ?.scrollIntoView({ behavior: "smooth" })
              }
        >
          <span className="relative z-10">Get started</span>
          <span className="pointer-events-none absolute inset-0 bg-linear-to-r from-transparent via-white/10 to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
        </button>


      </nav>
    </header>
  );
}

