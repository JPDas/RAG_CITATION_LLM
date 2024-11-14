# RAG_CITATION_LLM

Adding citations to LLM responses enhances trustworthiness, accuracy, and ethical compliance. By providing verifiable sources, LLMs can reduce the risk of misinformation and promote responsible AI usage.

This is a POC project to see the citations in the response. 
Langchain Retrieval Augmented Generations(RAG) chain with Citations Using OpenAI + FAISS VectorDB

Rquired finetune of the code : To generate citations, you just need to keep metadata (e.g doc name, URL, page number etc) along with your document-chunks in your vector-db. When you add chunks to the LLM context for the final answer generation, number them sequentially and ask the LLM to show granular citations for its final answer, referencing the chunk numbers. Then your code can look up the metadata of the cited chunks and display them after the LLM answer. You could even show the chunks themselves as excerpts.
    Display the citations in your answer in this format - "[link text](source idx)"

Example: https://www.perplexity.ai/
