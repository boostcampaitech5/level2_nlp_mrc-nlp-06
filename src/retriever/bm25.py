import json
import os
import pickle
import time
import math
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from konlpy.tag import Mecab

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class BM25:
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer
        self.corpus = corpus

        if tokenizer:
            self.corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus):
        tokenized_corpus = list(map(self.tokenizer, corpus))
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, n=5):
        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [self.corpus[i] for i in top_n]
    
    def get_top_n_idx(self, query, n=5):
        scores = self.get_scores(query)
        return np.argsort(scores)[::-1][:n]


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score.tolist()


class BM25Retrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        """

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.context2id = {item : i for i, item in enumerate(self.contexts)}

        self.tokenizer = tokenize_fn
        with timer('make BM25 object'):
            self.bm25 = BM25Okapi(self.contexts, self.tokenizer)

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
        total = []
        for idx, example in enumerate(
            tqdm(query_or_dataset, desc="BM25 retrieval: ")
        ):
            tmp = {
                # Query와 해당 id를 반환합니다.
                "question": example["question"],
                "id": example["id"],
                # Retrieve한 context를 반환합니다.
                "context": ' '.join([' '.join(i) for i in self.bm25.get_top_n(example['question'], topk)]),
                "doc_idx" : ' '.join([str(i) for i in self.bm25.get_top_n_idx(example['question'], topk)])
            }
            if "context" in example.keys() and "answers" in example.keys():
                # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                tmp["original_context"] = example["context"]
                tmp["true_doc"] = self.context2id[example['context']]
                # Retrieve한 context id를 반환합니다.
                tmp["answers"] = example["answers"]
                int_list = [int(x) for x in tmp['doc_idx'].split()] # top-k document index list
                try:
                    ans_idx = int_list.index(tmp['true_doc']) # index of gold document in int_list
                except ValueError:
                    ans_idx = -1 # no gold document retrieved in top-k
                tmp['RR'] = (1 / (ans_idx + 1) if ans_idx + 1 > 0 else 0)

            total.append(tmp)

        cqas = pd.DataFrame(total)
        return cqas


if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", metavar="../../data/train_dataset", type=str, default="../../data/train_dataset", help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        metavar="bert-base-multilingual-cased",
        default ="bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", metavar="./data", type=str, default="../../data", help="")
    
    parser.add_argument(
        "--context_path", metavar="wikipedia_documents", type=str, default="wikipedia_documents.json", help=""
    )
    parser.add_argument("--use_faiss", metavar=False, type=bool, default=False, help="")
    parser.add_argument("--num_clusters", metavar=False, type=int, default=64, help="")

    args = parser.parse_args()

    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    full_ds = full_ds.select(range(10))
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)
    full_ds = full_ds.select(range(10))

    from transformers import AutoTokenizer

    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False,).tokenize
    tokenizer = lambda x : x.split(' ')
    # tokenizer = Mecab().morphs

    retriever = BM25Retrieval(
        tokenize_fn=tokenizer,
        data_path=args.data_path,
        context_path=args.context_path,
    )

    if args.use_faiss:
        with timer("bulk query by faiss"):
            df = retriever.retrieve_faiss(full_ds, 5)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds, 5)
            df["correct"] = df["original_context"] == df["context"]
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )

    # evaluate retrieval
    tmp = df.loc[df['RR'] > 0]['RR']
    MRR = tmp.sum() / len(df)
    Acc = len(tmp) / len(df)
    print(f'MRR : {MRR}')
    print(f'Acc : {Acc}')
