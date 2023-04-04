import pandas as pd
import os
import openai
import numpy as np
import tiktoken

from constants.constants import *

openai.api_key = OPENAI_API_KEY

COMPLETIONS_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
system_content = '''
Imagine yourself as a legal advice consultant. If you think user question is not complete and may need more context, please ask follow up questions to maintain the dialogue.
'''

COMPLETIONS_API_PARAMS = {
    "temperature": 0.5,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL
}

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame, batch_size: int = 100) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    embeddings = {}
    batches = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]
    for batch in batches:
        texts = list(batch['Content'])
        results = openai.Embedding.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        for i, result in enumerate(results["data"]):
            embedding = result["embedding"]
            idx = batch.index[i]
            embeddings[idx] = embedding
    return embeddings

# Define a function to tokenize a sentence and count the tokens
def count_tokens(sentence):
    splitted = sentence.split()
    return len(splitted)


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities



MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003
encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.Content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(df['Article'][int(ind)] for ind in chosen_sections_indexes))
    
    sections = "\n".join(df['Article'][int(ind)] for ind in chosen_sections_indexes)
    chosen_articles = f"Selected {len(chosen_sections)} document sections:".join(sections)
    
    # header = """Answer the question as truthfully as possible using the provided context and imagine yourself as a legal advice consultant. If you think user question is not complete and may need more context, please ask follow up questions to maintain the dialogue. \n\nContext:\n"""
    header = ''' Answer the question as truthfully as possible using the provided context: \n '''

    return chosen_articles, header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:" 



def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    chosen_articles, prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    response = openai.ChatCompletion.create(
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                **COMPLETIONS_API_PARAMS
            )

    return chosen_articles + "\n".join(response["choices"][0]["message"]['content'].strip(" \n"))


class Adlet:
    def __init__(self) -> None:
        self.df = pd.read_excel(r'./models/civil_code_data_for_training.xlsx').drop(columns='Unnamed: 0')
        self.df["tokens"] = self.df["Content"].apply(count_tokens)

        embedding_df = pd.read_excel(r'./models/embeddings_df.xlsx').drop(columns='Unnamed: 0')
        new_dict = embedding_df.to_dict(orient='index')
        self.document_embeddings = {key: list(values.values()) for key, values in new_dict.items()}

    def answer(self, question: str) -> str:
        return answer_query_with_context(question, self.df, self.document_embeddings)
