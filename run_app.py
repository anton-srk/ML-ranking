import json
import string
from typing import Dict, List, Union, Callable

import os
import faiss
import numpy as np
import torch
from flask import Flask, request, jsonify
import langdetect
from nltk import word_tokenize


os.environ['EMB_PATH_KNRM'] = 'model_embeds.bin'
os.environ['MLP_PATH'] = 'knrm_mlp.bin'
os.environ['VOCAB_PATH'] = 'vocab.json'


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(-0.5 * torch.pow(x - self.mu, 2) / np.power(self.sigma, 2))


class KNRM(torch.nn.Module):
    def __init__(
            self, embedding_matrix: np.ndarray, freeze_embeddings: bool, kernel_num: int = 21,
            sigma: float = 0.1, exact_sigma: float = 0.001,
            out_layers: List[int] = [10, 5]
    ):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        def kernal_mus(n_kernels):
            l_mu = [1.]
            if n_kernels == 1:
                return l_mu
            bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
            l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
            for j in range(1, n_kernels - 1):
                l_mu.append(l_mu[j] - bin_size)
            return sorted(l_mu)

        def kernel_sigmas(n_kernels, sigma, exact_sigma):
            exact_sigma = [exact_sigma]
            if n_kernels == 1:
                return exact_sigma
            exact_sigma += [sigma] * (n_kernels - 1)
            return sorted(exact_sigma)[::-1]

        mus = kernal_mus(self.kernel_num)
        sigmas = kernel_sigmas(self.kernel_num, self.exact_sigma, self.sigma)
        kernels = []

        for i in range(self.kernel_num):
            kernels.append(GaussianKernel(mus[i], sigmas[i]))

        return torch.nn.ModuleList(kernels)

    def _get_mlp(self) -> torch.nn.Sequential:
        if not self.out_layers:
            return torch.nn.Sequential(
                torch.nn.Linear(self.kernel_num, 1)
            )
        else:
            out_layers = []
            in_dim = self.kernel_num
            for layer in self.out_layers + [1]:
                out_layers.append(torch.nn.ReLU())
                out_layers.append(torch.nn.Linear(in_dim, layer))
                in_dim = layer

            return torch.nn.Sequential(*out_layers)

    def forward(
            self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]
    ) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        eps = 1e-8
        q_n = query.norm(dim=-1).unsqueeze(-1).repeat(1, 1, query.shape[-1])
        query_norm = query / torch.max(q_n, eps * torch.ones_like(q_n))
        d_n = doc.norm(dim=-1).unsqueeze(-1).repeat(1, 1, doc.shape[-1])
        doc_norm = doc / torch.max(d_n, eps * torch.ones_like(d_n))
        res = torch.bmm(query_norm, doc_norm.transpose(2, 1))

        return res

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)
        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = [Batch, Left], [Batch, Right]
        query = torch.clamp(inputs['query'], min=0, max=self.embeddings.num_embeddings - 1)
        doc = torch.clamp(inputs['document'], min=0, max=self.embeddings.num_embeddings - 1)

        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(self.embeddings(query), self.embeddings(doc))
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out


def _hadle_punctuation(inp_str: str) -> str:
    punct = string.punctuation
    return inp_str.translate(str.maketrans(punct, ' ' * len(punct)))


def _simple_preproc(inp_str: str) -> List[str]:
    return word_tokenize(_hadle_punctuation(inp_str.lower()))


class RankingDataset(torch.utils.data.Dataset):
    def __init__(
            self, index_pairs_or_triplets: List[List[Union[str, float]]],
            input_tokens: List[str],
            idx_to_text_mapping: Dict[str, str], vocab: Dict[str, int], oov_val: int,
            preproc_func: Callable, max_len: int = 30
    ):
        self.index_pairs_or_triplets = index_pairs_or_triplets
        self.idx_to_text_mapping = idx_to_text_mapping
        self.vocab = vocab
        self.input_tokens = input_tokens
        self.oov_val = oov_val
        self.preproc_func = preproc_func
        self.max_len = max_len

    def __len__(self):
        return len(self.index_pairs_or_triplets)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        return [
            self.vocab[word] if word in self.vocab.keys()
            else self.oov_val for word in tokenized_text
        ]

    def _convert_text_idx_to_token_idxs(self, idx: int) -> List[int]:
        tokens = self.preproc_func(self.idx_to_text_mapping[idx])
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        return self._tokenized_text_to_index(tokens)

    def __getitem__(self, idx):
        idx1 = self.index_pairs_or_triplets[idx]
        tokens1 = self._tokenized_text_to_index(self.input_tokens)
        tokens2 = self._convert_text_idx_to_token_idxs(idx1)
        return {'query': tokens1, 'document': tokens2}, idx


def collate_fn(batch_objs: List[Union[Dict[str, torch.Tensor], torch.FloatTensor]]):
    max_len_q1 = -1
    max_len_d1 = -1
    max_len_q2 = -1
    max_len_d2 = -1

    is_triplets = False
    for elem in batch_objs:
        if len(elem) == 3:
            left_elem, right_elem, label = elem
            is_triplets = True
        else:
            left_elem, label = elem

        max_len_q1 = max(len(left_elem['query']), max_len_q1)
        max_len_d1 = max(len(left_elem['document']), max_len_d1)
        if len(elem) == 3:
            max_len_q2 = max(len(right_elem['query']), max_len_q2)
            max_len_d2 = max(len(right_elem['document']), max_len_d2)

    q1s = []
    d1s = []
    q2s = []
    d2s = []
    labels = []

    for elem in batch_objs:
        if is_triplets:
            left_elem, right_elem, label = elem
        else:
            left_elem, label = elem

        pad_len1 = max_len_q1 - len(left_elem['query'])
        pad_len2 = max_len_d1 - len(left_elem['document'])
        if is_triplets:
            pad_len3 = max_len_q2 - len(right_elem['query'])
            pad_len4 = max_len_d2 - len(right_elem['document'])

        q1s.append(left_elem['query'] + [0] * pad_len1)
        d1s.append(left_elem['document'] + [0] * pad_len2)
        if is_triplets:
            q2s.append(right_elem['query'] + [0] * pad_len3)
            d2s.append(right_elem['document'] + [0] * pad_len4)
        labels.append([label])
    q1s = torch.LongTensor(q1s)
    d1s = torch.LongTensor(d1s)
    if is_triplets:
        q2s = torch.LongTensor(q2s)
        d2s = torch.LongTensor(d2s)
    labels = torch.FloatTensor(labels)

    ret_left = {'query': q1s, 'document': d1s}
    if is_triplets:
        ret_right = {'query': q2s, 'document': d2s}
        return ret_left, ret_right, labels
    else:
        return ret_left, labels


class DataStorage:
    def __init__(self):
        self.globalData = ""


def create_app():
    # define the app
    app = Flask(__name__)
    ds = DataStorage()

    @app.route('/ping')
    def ping():
        if not hasattr(ds, 'pinged'):
            ds.pinged = True
            ds.embeds = torch.load(os.environ.get('EMB_PATH_KNRM'))['weight'].numpy()
            with open(os.environ.get('VOCAB_PATH'), 'r') as handle:
                ds.vocab = json.load(handle)
            model = KNRM(embedding_matrix=ds.embeds, freeze_embeddings=False)
            model.mlp.load_state_dict(torch.load(os.environ.get('MLP_PATH')))
            model.eval()
            ds.model = model

            return jsonify({'status': 'ok'})
        else:
            return jsonify({'status': 'not ok'})

    @app.route('/query', methods=['GET', 'POST'])
    def query():
        example_input = json.loads(request.json)
        if not hasattr(ds, 'index'):
            return jsonify({'status': 'FAISS is not initialized!'})
        else:
            example_input = example_input.get('queries', None)
            lang_check = {}
            for i, sent in enumerate(example_input):
                if langdetect.detect(sent) == 'en':
                    lang_check[i] = True
                else:
                    lang_check[i] = False

            eng_idxs = [idx for idx in lang_check.keys() if lang_check[idx]]
            if len(eng_idxs) > 0:
                idx_dict = {eng_idxs[i]: i for i in range(len(eng_idxs))}

                example_processed = [
                    _simple_preproc(sent) for i, sent in enumerate(example_input)
                    if lang_check[i]
                ]

                example_embeds = [
                    np.array([ds.embeds[ds.vocab[word]] for word in sent if word in ds.vocab.keys()]).mean(axis=0)
                    for sent in example_processed
                ]

                if len(example_embeds) > 1:
                    example_embeds = np.stack(example_embeds)
                else:
                    example_embeds = np.array(example_embeds)

                num_neighbors = ds.nq
                filtered_embeds = ds.index.search(example_embeds.astype('float32'), num_neighbors)[1]
                filtered_embeds = [[ds.embeddings_vocab[f] for f in emb] for emb in filtered_embeds]

                suggest_list = []
                for i, emb in enumerate(filtered_embeds):
                    filtered_idx2str = {
                        w: ds.idx2text[w] for w in ds.idx2text.keys() if w in emb
                    }
                    dataset_input = example_processed[i]
                    dataset = RankingDataset(
                        list(filtered_idx2str.keys()), dataset_input, filtered_idx2str, ds.vocab, 1, _simple_preproc
                    )
                    dataloader = torch.utils.data.DataLoader(
                        dataset, batch_size=ds.nq, num_workers=0,
                        collate_fn=collate_fn, shuffle=False
                    )

                    idx_list = list(filtered_idx2str.keys())

                    batch, idx = next(iter(dataloader))
                    model = ds.model
                    preds = model.predict(batch)

                    num_max = 10 if ds.nq >= 10 else ds.nq
                    max_idxs = [int(idx[i]) for i in torch.topk(preds.squeeze(), num_max)[1].numpy()]
                    max_idxs_real = [idx_list[idx] for idx in max_idxs]
                    max_str = [ds.idx2text[idx_list[max_idx]] for max_idx in max_idxs]
                    suggest_list += [[tuple(pair) for pair in zip(max_idxs_real, max_str)]]

                out_dict = {
                    'lang_check': list(lang_check.values()),
                    'suggestions': [suggest_list[idx_dict[idx]] if idx in eng_idxs else None for idx in
                                    lang_check.keys()]
                }
            else:
                out_dict = {
                    'lang_check': list(lang_check.values()),
                    'suggestions': [None for idx in lang_check.keys()]
                }

            return jsonify(out_dict)

    @app.route('/update_index', methods=['GET', 'POST'])
    def index_update():
        ds.idx2text = json.loads(request.json)['documents']

        ds.embedded_docs = {w: _simple_preproc(ds.idx2text[w]) for w in ds.idx2text.keys()}
        ds.embedded_docs = {
            w: np.array([ds.embeds[ds.vocab[word]] for word in ds.embedded_docs[w] if word in ds.vocab.keys()]).mean(axis=0)
            for w in ds.embedded_docs.keys()
        }

        docs_embeddings = np.stack([
            np.array(ds.embedded_docs[w]) if not np.isnan(ds.embedded_docs[w]).any() else np.zeros(50)
            for w in ds.embedded_docs.keys()
        ])

        emb_keys = list(ds.embedded_docs.keys())
        ds.embeddings_vocab = {
            i: emb_keys[i] for i in range(docs_embeddings.shape[0])
        }

        dim = docs_embeddings.shape[1]
        ds.nq = 10

        index = faiss.IndexFlatL2(dim)
        index.add(docs_embeddings.astype('float32'))
        ds.index = index

        return jsonify({'status': 'ok', 'index_size': index.ntotal})

    return app


if __name__ == '__main__':
    # This is used when running locally.
    create_app().run(host='0.0.0.0', debug=True, port=11000)
