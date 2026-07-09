import Link from "next/link";
import { Calendar, Clock } from "lucide-react";
import { SiteLayout } from "@/components/SiteLayout";
import { posts } from "@/data/posts";

export const metadata = {
  title: "Articles | TinyINR",
  description: "All TinyINR project notes and implementation posts.",
};

export default function ArticlesPage() {
  const sortedPosts = [...posts].sort(
    (a, b) => Date.parse(b.date) - Date.parse(a.date),
  );

  return (
    <SiteLayout>
      <section className="mx-auto max-w-5xl px-6 pb-24 pt-20">
        <div className="mb-12 border-b border-border/50 pb-10">
          <h1 className="mb-4 font-mono text-xs uppercase tracking-widest text-accent">
            Articles
          </h1>
          <p className="mt-6 max-w-2xl text-lg leading-8 text-secondary">
            Follow the project from concept notes into implementation logs,
            benchmarks, and CUDA experiments.
          </p>
        </div>

        <div className="space-y-5">
          {sortedPosts.map((post) => (
            <Link
              key={post.slug}
              href={`/post/${post.slug}`}
              className="group block rounded-xl border border-border bg-surface p-6 transition-all duration-300 hover:border-accent/40 hover:bg-surfaceHover"
            >
              <article className="grid gap-5 md:grid-cols-[160px_1fr] md:items-start">
                <div className="space-y-2 font-mono text-xs text-secondary">
                  <span className="flex items-center gap-2">
                    <Calendar className="h-3.5 w-3.5" />
                    {post.date}
                  </span>
                  <span className="flex items-center gap-2">
                    <Clock className="h-3.5 w-3.5" />
                    {post.readTime}
                  </span>
                </div>

                <div>
                  <h2 className="text-2xl font-bold leading-tight text-primary transition-colors group-hover:text-accent">
                    {post.title}
                  </h2>
                  <p className="mt-3 max-w-3xl leading-7 text-secondary">
                    {post.excerpt}
                  </p>
                  <div className="mt-5 flex flex-wrap gap-2">
                    {post.tags.map((tag) => (
                      <span
                        key={tag}
                        className="rounded border border-border bg-background px-2 py-1 font-mono text-[10px] text-secondary"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              </article>
            </Link>
          ))}
        </div>
      </section>
    </SiteLayout>
  );
}
