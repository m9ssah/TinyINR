import Link from "next/link";
import { ArrowRight, Calendar, ChevronRight, Clock } from "lucide-react";
import { MotionReveal } from "@/components/MotionReveal";
import { SiteLayout } from "@/components/SiteLayout";
import { VectorField } from "@/components/VectorField";
import { posts } from "@/data/posts";

export default function Home() {
  const featuredPost = posts.find((post) => post.featured);
  const recentPosts = posts.filter((post) => !post.featured);

  return (
    <SiteLayout>
      <div className="pb-20">
        <section className="relative flex h-[62vh] min-h-[520px] items-center justify-center overflow-hidden border-b border-border/50">
          <div className="absolute inset-0 z-0">
            <VectorField />
            <div className="absolute inset-0 bg-gradient-to-b from-background/20 via-background/60 to-background" />
          </div>

          <div className="relative z-10 mx-auto w-full max-w-5xl px-6 text-center">
            <MotionReveal>
              <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-accent/20 bg-accent/10 px-3 py-1 font-mono text-xs text-accent">
                <span className="relative flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-75" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
                </span>
                Flow matching for coordinate-value data
              </div>

              <h1 className="mb-6 text-5xl font-bold tracking-tight text-primary md:text-7xl">
                TinyINR <br />
              </h1>

              <p className="mx-auto mb-10 max-w-2xl text-lg leading-relaxed text-secondary md:text-xl">
                Research notes, systems sketches, and visual explanations for a
                CUDA-accelerated flow matching engine over irregular
                coordinate-value data.
              </p>
            </MotionReveal>
          </div>
        </section>

        <div className="mx-auto mt-20 max-w-5xl px-6">
          {featuredPost ? (
            <MotionReveal className="mb-24" delay={0.1}>
              <div className="mb-8 flex items-center gap-4">
                <div className="h-px flex-1 bg-border" />
                <h2 className="font-mono text-sm uppercase tracking-widest text-secondary">
                  Featured
                </h2>
                <div className="h-px flex-1 bg-border" />
              </div>

              <Link href={`/post/${featuredPost.slug}`} className="group block">
                <article className="relative overflow-hidden rounded-2xl border border-border bg-surface p-8 transition-all duration-300 hover:border-accent/50 hover:bg-surfaceHover md:p-12">
                  <div className="absolute right-0 top-0 h-64 w-64 rounded-full bg-accent/5 blur-[80px] transition-colors group-hover:bg-accent/10" />

                  <div className="relative z-10">
                    <div className="mb-6 flex flex-wrap items-center gap-4 font-mono text-xs text-secondary">
                      <span className="flex items-center gap-1.5">
                        <Calendar className="h-3.5 w-3.5" />
                        {featuredPost.date}
                      </span>
                      <span className="flex items-center gap-1.5">
                        <Clock className="h-3.5 w-3.5" />
                        {featuredPost.readTime}
                      </span>
                    </div>

                    <h3 className="mb-4 text-3xl font-bold text-primary transition-colors group-hover:text-accent md:text-4xl">
                      {featuredPost.title}
                    </h3>

                    <p className="mb-8 max-w-3xl text-lg leading-relaxed text-secondary">
                      {featuredPost.excerpt}
                    </p>

                    <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
                      <div className="flex flex-wrap gap-2">
                        {featuredPost.tags.map((tag) => (
                          <span
                            key={tag}
                            className="rounded-full border border-border bg-background px-3 py-1 font-mono text-xs text-secondary"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                      <div className="flex items-center gap-2 font-mono text-sm text-accent transition-transform group-hover:translate-x-2">
                        Read article <ArrowRight className="h-4 w-4" />
                      </div>
                    </div>
                  </div>
                </article>
              </Link>
            </MotionReveal>
          ) : null}

          <section>
            <div className="mb-8 flex items-center gap-4">
              <h2 className="font-mono text-sm uppercase tracking-widest text-secondary">
                Latest Articles
              </h2>
              <div className="h-px flex-1 bg-border" />
            </div>

            <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
              {recentPosts.map((post, index) => (
                <MotionReveal key={post.slug} delay={0.18 + index * 0.08}>
                  <Link href={`/post/${post.slug}`} className="group block h-full">
                    <article className="flex h-full flex-col rounded-xl border border-border bg-surface p-6 transition-all duration-300 hover:border-accent/30 hover:bg-surfaceHover">
                      <div className="mb-4 flex items-center gap-3 font-mono text-xs text-secondary">
                        <span>{post.date}</span>
                        <span className="h-1 w-1 rounded-full bg-border" />
                        <span>{post.readTime}</span>
                      </div>

                      <h3 className="mb-3 text-xl font-bold text-primary transition-colors group-hover:text-accent">
                        {post.title}
                      </h3>

                      <p className="mb-6 flex-1 text-sm leading-6 text-secondary">
                        {post.excerpt}
                      </p>

                      <div className="mt-auto flex items-center justify-between border-t border-border/50 pt-4">
                        <div className="flex gap-2 overflow-hidden">
                          {post.tags.slice(0, 2).map((tag) => (
                            <span
                              key={tag}
                              className="whitespace-nowrap rounded border border-border bg-background px-2 py-1 font-mono text-[10px] text-secondary"
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                        <ChevronRight className="h-4 w-4 text-secondary transition-colors group-hover:text-accent" />
                      </div>
                    </article>
                  </Link>
                </MotionReveal>
              ))}
            </div>
          </section>
        </div>
      </div>
    </SiteLayout>
  );
}
