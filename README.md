# Similarity as Nearness

This repository contains data and code for a brief demonstration of three commonly used distance / similarity metrics, and how they can be used in the context of assessing text similarity.

Learning goals for the lesson include the following: 

1. Understand how different nearness measures can represent document similarity 
2. Compare three common nearness measures (Euclidean distance, Manhattan distance, and Cosine distance / similarity) and how they represent specific text features
3. Analyze what kinds of similarity each represents, and how they might be used for specific purposes
4. Examine some R code that implements text similarity measures

In addition to these goals, the material here can support:

1. Examining some Python code to see how html, pdf, and text files can be processed to create document-term matrices 
2. Thinking critically about the data visualizations used in the R code 
3. Considering the ethical consequences of using more complex and less transparent computational methods in context such as online commerce, social media, and Large Language Models (LLMS) and other AI systems

## Folder Contents

### pre-process

Contains Python scripts used to extract text data, including Part-of-Speech (POS) tags 
 
### book-reviews

Contains sub-folders for source `pdf` and `html` files; extracted `txt` files, and term-frequency `csv` files 

## Top-Level Files

`compare-distances.qmd`: Contains R code that explores text similarity of five book review excerpts, as well as the full reviews  

`compare-distances.html`: Render HTML from above

`review_metadata.csv`: Contains bibliographical information on the five book reviews visualized in `compare-distances.qmd`