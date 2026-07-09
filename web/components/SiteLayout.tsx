import Link from "next/link";
import { Activity } from "lucide-react";

export function SiteLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative flex min-h-screen flex-col overflow-hidden bg-background">
      <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
        <div className="absolute left-[-10%] top-[-10%] h-[40%] w-[40%] rounded-full bg-accent/5 blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] h-[40%] w-[40%] rounded-full bg-blue-500/5 blur-[120px]" />
      </div>

      <header className="sticky top-0 z-50 border-b border-border/50 bg-background/80 backdrop-blur-md">
        <div className="mx-auto flex h-16 max-w-5xl items-center justify-between px-6">
          <Link href="/" className="group flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded border border-border bg-surfaceHover transition-colors group-hover:border-accent/50">
              <Activity className="h-5 w-5 text-accent" />
            </span>
            <span className="font-mono font-semibold tracking-tight text-primary">
              Tiny<span className="text-secondary">INR</span>
            </span>
          </Link>

          <nav className="flex items-center gap-5 font-mono text-sm text-secondary">
            <Link href="/articles" className="transition-colors hover:text-primary">
              Articles
            </Link>
            <span className="h-4 w-px bg-border" />
            <a
              href="https://github.com/m9ssah/TinyINR"
              aria-label="TinyINR on GitHub"
              className="font-semibold transition-colors hover:text-primary"
            >
              GH
            </a>
          </nav>
        </div>
      </header>

      <main className="relative z-10 flex-1">{children}</main>

      <footer
        id="about"
        className="relative z-10 mt-20 border-t border-border/50 bg-background"
      >
      </footer>
    </div>
  );
}
