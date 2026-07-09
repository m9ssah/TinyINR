import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft, Calendar, Clock, Share2 } from "lucide-react";
import { MathDisclosure } from "@/components/MathDisclosure";
import { SiteLayout } from "@/components/SiteLayout";
import { getPost, posts } from "@/data/posts";
import { FlowMatchingPostBody } from "./flow-matching";
import { MilestoneOnePostBody } from "./milestone-one";

interface PostPageProps {
  params: Promise<{
    slug: string;
  }>;
}

export function generateStaticParams() {
  return posts.map((post) => ({ slug: post.slug }));
}

export async function generateMetadata({ params }: PostPageProps) {
  const { slug } = await params;
  const post = getPost(slug);

  if (!post) {
    return {
      title: "Article not found | TinyINR",
    };
  }

  return {
    title: `${post.title} | TinyINR`,
    description: post.excerpt,
  };
}

export default async function PostPage({ params }: PostPageProps) {
  const { slug } = await params;
  const post = getPost(slug);

  if (!post) {
    notFound();
  }

  return (
    <SiteLayout>
      <article className="pb-20">
        <header className="relative overflow-hidden border-b border-border/50 pb-12 pt-20">
          <div className="absolute inset-0 z-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-accent/10 via-background to-background" />

          <div className="relative z-10 mx-auto max-w-3xl px-6">
            <Link
              href="/"
              className="mb-10 inline-flex items-center gap-2 font-mono text-sm text-secondary transition-colors hover:text-accent"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to articles
            </Link>

            <div className="mb-6 flex flex-wrap gap-2">
              {post.tags.map((tag) => (
                <span
                  key={tag}
                  className="rounded-full border border-border bg-surface px-3 py-1 font-mono text-xs text-accent"
                >
                  {tag}
                </span>
              ))}
            </div>

            <h1 className="mb-6 text-4xl font-bold leading-tight text-primary md:text-5xl">
              {post.title}
            </h1>

            <div className="mt-8 flex flex-col gap-6 border-t border-border/50 py-6 md:flex-row md:items-center md:justify-between">
              <div className="flex flex-wrap items-center gap-4 font-mono text-sm text-secondary">
                <span className="flex items-center gap-1.5">
                  <Calendar className="h-4 w-4" />
                  {post.date}
                </span>
                <span className="flex items-center gap-1.5">
                  <Clock className="h-4 w-4" />
                  {post.readTime}
                </span>
              </div>
              <button
                type="button"
                aria-label="Share article"
                className="flex h-10 w-10 items-center justify-center rounded-full border border-border bg-surface text-secondary transition-colors hover:border-accent/50 hover:text-accent"
              >
                <Share2 className="h-4 w-4" />
              </button>
            </div>
          </div>
        </header>

        <div className="mx-auto mt-12 max-w-3xl px-6">
          <div className="prose max-w-none">
            <p className="mb-8 text-xl leading-relaxed text-primary">
              {post.excerpt}
            </p>

            {slug === "what-is-flow-matching" ? (
              <FlowMatchingPostBody />
            ) : slug === "milestone-1-tensor-coordinate-fourier" ? (
              <MilestoneOnePostBody />
            ) : (
              <>
                <p>
                  TinyINR treats signals as coordinate-value relationships.
                  Instead of storing only pixels or samples, the project asks
                  what happens when a model learns a continuous function from
                  coordinates to values, then learns how those values move
                  under a generative flow.
                </p>

                <p>
                  That framing makes Fourier embeddings, coordinate batches,
                  and CUDA kernels part of the same story. The website is
                  organized as a notebook for those pieces: what each component
                  does, why the shape contract matters, and where GPU
                  acceleration begins to pay off.
                </p>

                <h2>The mental model</h2>

                <p>
                  For image-like data, each point can be written as a coordinate
                  and a value. A 32 by 32 RGB image becomes 1,024
                  coordinate-value pairs, where each coordinate stores an x-y
                  position and each value stores RGB.
                </p>

                <pre>
                  <code>{`coordinates: [B, H * W, 2]
values:      [B, H * W, 3]
embedding:   [B, H * W, 2 * D * F]`}</code>
                </pre>

                <div className="relative my-10 overflow-hidden rounded-xl border border-border bg-surface p-6">
                  <div
                    className="absolute inset-0 opacity-20"
                    style={{
                      backgroundImage:
                        "radial-gradient(#39d3bb 1px, transparent 1px)",
                      backgroundSize: "20px 20px",
                    }}
                  />
                  <div className="relative z-10 flex flex-col items-center text-center">
                    <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full border border-accent bg-accent/20">
                      <div className="h-2 w-2 animate-ping rounded-full bg-accent" />
                    </div>
                    <h4 className="mb-2 font-mono text-sm text-primary">
                      Coordinate Field View
                    </h4>
                    <p className="text-sm text-secondary">
                      The project learns how coordinate-indexed values can be
                      represented, sampled, embedded, and eventually
                      transported.
                    </p>
                  </div>
                </div>

                <h2>Where Fourier features fit</h2>

                <p>
                  Small neural networks often prefer smooth, low-frequency
                  functions. Fourier coordinate embeddings give the network a
                  richer basis by mapping every coordinate value through sine
                  and cosine bands.
                </p>

                <MathDisclosure title="Coordinate embedding formula">
                  <p>
                    For a coordinate value x and frequency band f, the current
                    Week 4 kernel target is:
                  </p>
                  <pre>
                    <code>{`sin(pi * frequency[f] * x)
cos(pi * frequency[f] * x)`}</code>
                  </pre>
                  <p>
                    The first CUDA kernel should compute only these Fourier
                    features. Keeping raw coordinate concatenation outside the
                    kernel makes the GPU path easier to verify.
                  </p>
                </MathDisclosure>

                <h2>Why this matters for flow matching</h2>

                <p>
                  Flow matching learns a vector field that moves samples from
                  one distribution to another. In TinyINR, the interesting
                  question is how to do this over coordinate-value structures
                  rather than fixed grids alone.
                </p>

                <ul>
                  <li>
                    <strong>Tensor</strong> gives every module shared CPU
                    storage and shape rules.
                  </li>
                  <li>
                    <strong>CoordinateBatch</strong> turns images and irregular
                    samples into coordinate-value training examples.
                  </li>
                  <li>
                    <strong>FourierEmbedding</strong> makes coordinates easier
                    for an MLP to model.
                  </li>
                  <li>
                    <strong>CUDA kernels</strong> make repeated coordinate
                    transforms measurable and scalable.
                  </li>
                </ul>

                <h3>A narrow Week 4 target</h3>

                <p>
                  The right first GPU milestone is not a complete tensor
                  runtime. It is one custom kernel that is correct, benchmarked,
                  and easy to explain.
                </p>

                <pre>
                  <code>{`CPU embedding time / CUDA kernel time
CPU embedding time / (H2D + kernel + D2H)`}</code>
                </pre>
              </>
            )}
          </div>
        </div>
      </article>
    </SiteLayout>
  );
}
