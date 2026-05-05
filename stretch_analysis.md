Embedding Visualization Analysis

The t-SNE visualization of the GloVe word embeddings reveals clear but imperfect clustering across the five BBC categories.

Business-related words form a relatively tight cluster in the lower-right region of the plot, indicating strong semantic similarity within this domain. Entertainment terms are also grouped closely in the lower-left area, suggesting consistent vocabulary usage across entertainment articles. Similarly, sports-related words appear concentrated in the upper-central region, forming a recognizable cluster.

In contrast, political terms are more widely dispersed across the space, appearing both in the upper-right and lower regions. This dispersion suggests that political language overlaps significantly with other domains, particularly business and technology. Words such as “budget,” “committee,” and “country” likely contribute to this overlap, as they are commonly used in multiple contexts.

Technology-related words form a cluster on the left side of the plot, but with some spread toward other categories. This indicates partial semantic cohesion, alongside shared terminology with other fields.

Additionally, several general-purpose words (e.g., “work,” “time,” “number”) appear between clusters, acting as outliers. These words have broad meanings and are used across multiple categories, which explains their position in mixed regions of the embedding space.

Overall, the visualization demonstrates that GloVe embeddings capture meaningful semantic structure, with clear category-based grouping, while also revealing natural overlap between related domains and the presence of context-independent vocabulary.