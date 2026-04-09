import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const journal = defineCollection({
  loader: glob({ pattern: '**/[^_]*.{md,mdx}', base: "./src/content/journal" }),
  schema: z.object({
    title: z.string(),
    date: z.date(),
    summary: z.string(),
    tags: z.array(z.string()).optional(),
  }),
});

export const collections = { journal };
