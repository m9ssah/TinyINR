export interface PostMeta {
  slug: string;
  title: string;
  excerpt: string;
  date: string;
  readTime: string;
  tags: string[];
  featured?: boolean;
}

export const posts: PostMeta[] = [
  {
    slug: "cicfm-as-loss",
    title: "CICFM as Loss",
    excerpt: "What is CICFM? Why do we use it for flow matching?",
    date: "2026-09-11",
    readTime: "mind read",
    tags: ["CICFM", "Flow Matching", "Loss Function"],
    featured: true,
  },
  {
    slug: "what-is-flow-matching",
    title: "What is Flow Matching?",
    excerpt: "An intuitive introduction to flow matching",
    date: "2026-07-08",
    readTime: "6 min read",
    tags: ["Flow Matching", "Velocity Fields"],
  },
  {
    slug: "milestone-2-cuda-coordinate-benchmark",
    title: "Milestone 2: Benchmarking CPU vs. CUDA Coordinate Embedding",
    excerpt:
      "Benchmarking TinyINR's first custom CUDA coordinate embedding kernel",
    date: "2026-07-27",
    readTime: "7 min read",
    tags: ["CUDA", "Benchmarking", "Fourier"],
  },
  {
    slug: "milestone-1-tensor-coordinate-fourier",
    title: "Milestone 1: Tensor core, Coordinate Batch, and Fourier Embedding",
    excerpt: "What we implemented for Milestone 1",
    date: "2026-07-09",
    readTime: "6 min read",
    tags: ["Tensor", "CoordinateBatch", "Fourier"],
  },
];

export function getPost(slug: string) {
  return posts.find((post) => post.slug === slug);
}
