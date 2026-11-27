import random
from pandas import col
import argparse
from pandas.api.typing.aliases import SequenceNotStr
from datasets import load_dataset
from collections import defaultdict
from openai import OpenAI
from tqdm.auto import tqdm, trange
from math import ceil
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.api.types import (
    
Documents,
    
EmbeddingFunction,
    
Embeddings
)

import numpy as np

class MyCustomEmbeddingFunction(EmbeddingFunction[Documents]):
    
    def __init__(self):
        """Initialize the embedding function."""

        
    def __call__(self, input: Documents) -> Embeddings:
        """Embed the input documents."""
        client = OpenAI(
            base_url="http://localhost:8081",
            api_key="xxx",
        )
        resp = client.embeddings.create(
            model="embeddinggemma-2-BF16.gguf",
            input=input,
            encoding_format="float",
            dimensions=3
        )

        embeddings = [np.array(x.embedding[0]) for x in resp]
        return embeddings

def load_tapaco() -> defaultdict[str, defaultdict[int, set[str]]]:
    """Load xed_emotions and shuffle them in a deterministic way"""
    ds = load_dataset("webdataset", data_files={"test": "https://zenodo.org/records/3707949/files/tapaco_v1.0.zip?download=1"}, split="test")
    result: defaultdict[str, defaultdict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in ds:
        # __key__: file name
        lang: str = row["__key__"].split("/")[1]
        # "txt": actual file
        line: str
        for line in row["txt"].strip().split("\n"):
            chunks = line.strip().split("\t")
            result[lang][int(chunks[0])].add(chunks[2])
    return result

if __name__ == "__main__":

    from argparse import ArgumentParser

    langs = [
        "af",
        "kab",
        "ar",
        "ko",
        "az",
        "kw",
        "be",
        "la",
        "ber",
        "lfn",
        "bg",
        "lt",
        "bn",
        "mk",
        "br",
        "mr",
        "ca",
        "nb",
        "cbk",
        "nds",
        "cmn",
        "nl",
        "cs",
        "orv",
        "da",
        "ota",
        "de",
        "pes",
        "el",
        "pl",
        "en",
        "pt",
        "eo",
        "rn",
        "es",
        "ro",
        "et",
        "ru",
        "eu",
        "sl",
        "fi",
        "sr",
        "fr",
        "sv",
        "gl",
        "tk",
        "gos",
        "tl",
        "he",
        "tlh",
        "hi",
        "toki",
        "hr",
        "tr",
        "hu",
        "tt",
        "hy",
        "ug",
        "ia",
        "uk",
        "id",
        "ur",
        "ie",
        "vi",
        "io",
        "vo",
        "is",
        "war",
        "it",
        "wuu",
        "ja",
        "yue",
        "jbo"
    ]
    parser = ArgumentParser()
    parser.add_argument("mode", choices=["test", "insert", "impostors"])
    args = parser.parse_args()
    
    chroma_client = chromadb.PersistentClient(path="chromadb")
    collection = chroma_client.get_or_create_collection(
    name="my_collection",
    embedding_function=MyCustomEmbeddingFunction()
)

    if args.mode == "insert":

        tapaco = load_tapaco()
        all_texts: set[tuple[str, str]] = set()


        for lang, parasets in tapaco.items():
            if lang not in langs:
                continue
            for paraset in parasets.values():
                for text in paraset:
                    all_texts.add((lang, text))

        all_texts: list[tuple[str, str]] = sorted(all_texts)

        batch_size = 256
        for batch in trange(ceil(len(all_texts) / batch_size)):
            texts = all_texts[batch*batch_size:(batch+1)*batch_size]

            collection.add([str(x) for x in range(batch*batch_size, batch*batch_size + len(texts))], documents=[t[1] for t in texts], metadatas=[{"lang": t[0]} for t in texts])
    
    elif args.mode == "impostors":
        from random import seed, shuffle, choice
        tapaco_unsorted: defaultdict[str, defaultdict[int, set[str]]] = load_tapaco()
        tapaco_sorted = {}
        for tapaco_lang in sorted(tapaco_unsorted.keys()):
            subdict = {}
            for key in sorted(tapaco_unsorted[tapaco_lang].keys()):
                subdict[key] = sorted(tapaco_unsorted[tapaco_lang][key])
            tapaco_sorted[tapaco_lang] = subdict
        for lang in langs:
            rows = []
            reverse_map = {}
            for key, paraset in tapaco_sorted[lang].items():
                for text in paraset:
                    reverse_map[text] = key
            random.seed(23453454356435)
            subsample = list(tapaco_sorted[lang].items())
            shuffle(subsample)
            subsample = subsample[:1000]
            for key, paraset in tqdm(subsample):
                # Pick a random sentence from paraset
                x = sorted(paraset)
                shuffle(x)
                left = x[0]
                gold = x[1]
                # Fish out impostors
                row = [left, gold]
                print("---")
                print(left)
                print(gold)
                for result in collection.query(query_texts=left, n_results=len(paraset)+1, where={"lang": lang})["documents"][0]:
                    if reverse_map[result] != reverse_map[left]:
                        row.append(result)
                        break
                if len(row) != 3:
                    continue

                # Flip last two and mark that they were flipped

                if choice([True, False]):
                    row = [row[0], row[2], row[1]]
                    row.append("Flipped")
                else:
                    row.append("Unflipped")
                
                rows.append(row)
            
            with open("testdata/" + lang + ".tsv", "w") as f:
                for row in rows:
                    f.write("\t".join(row) + "\n")

    else:
        while True:
            query = input("-> ")
            print(collection.query(query_texts=query, n_results=3))