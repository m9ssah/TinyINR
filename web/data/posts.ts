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
    slug: "what-is-flow-matching",
    title: "What is Flow Matching?",
    excerpt: "An intuitive introduction to flow matching",
    date: "2026-07-08",
    readTime: "6 min read",
    tags: ["Flow Matching", "Velocity Fields"],
    featured: true,
  },
  {
    slug: "flow-matching-for-coordinate-data",
    title: "Flow Matching for Coordinate-Value Data",
    excerpt:
      "A geometric introduction to TinyINR: learning flows over irregular coordinates, values, and adaptive latent assignments.",
    date: "2026-07-08",
    readTime: "8 min read", 
    tags: ["Flow Matching", "INR"],
  },
  {
    slug: "fourier-coordinate-embeddings",
    title: "Fourier Coordinate Embeddings",
    excerpt:
      "Why sine and cosine features help small networks capture sharp spatial detail in images, fields, and continuous signals.",
    date: "2026-07-02",
    readTime: "6 min read",
    tags: ["Fourier", "Coordinates"],
  },
  {
    slug: "building-the-tensor-core",
    title: "Building the TinyINR Tensor Core",
    excerpt:
      "The CPU-first tensor layer that gives CoordinateBatch, FourierEmbedding, and future CUDA kernels a shared memory model.",
    date: "2026-06-30",
    readTime: "7 min read",
    tags: ["C++", "Systems"],
  },
  {
    slug: "first-cuda-coordinate-kernel",
    title: "First CUDA Coordinate Embedding Kernel",
    excerpt:
      "A narrow Week 4 target: port Fourier coordinate embedding to CUDA, verify parity, and benchmark CPU versus GPU.",
    date: "2026-06-24",
    readTime: "10 min read",
    tags: ["CUDA", "Benchmark"],
  },
];

export function getPost(slug: string) {
  return posts.find((post) => post.slug === slug);
}
