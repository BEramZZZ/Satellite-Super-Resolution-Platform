import Link from "next/link";
import { ReactNode } from "react";

const links = [
  { href: "/lab", label: "Super-Resolution" },
  { href: "/report", label: "Research Report" },
  { href: "/benchmark", label: "Benchmark" },
  { href: "/concepts", label: "Concepts" }
];

export default function Shell({
  active,
  children
}: {
  active: string;
  children: ReactNode;
}) {
  return (
    <div className="page-shell">
      <header className="topbar">
        <div className="topbar-inner">
          <Link href="/lab" className="brand">
            <div className="brand-mark">SR</div>
            <div className="brand-copy">
              <div className="brand-title">Diffusion-Based Super-Resolution
                of Low-Cost Satellite Imagery</div>
              <div className="brand-subtitle">research / demo platform</div>
            </div>
          </Link>

          <nav className="nav">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`nav-link ${
                  active === link.href ? "nav-link-active" : ""
                }`}
              >
                {link.label}
              </Link>
            ))}
          </nav>
        </div>
      </header>

      <main className="main">{children}</main>

      <footer className="footer">
        Project 8 · Real-ESRGAN × Diffusion
      </footer>
    </div>
  );
}
