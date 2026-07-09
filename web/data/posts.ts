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
