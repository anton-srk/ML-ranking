import string
from collections import Counter
from typing import Dict, List, Tuple, Union, Callable

import numpy as np
import pandas as pd
import torch
from nltk import word_tokenize
from tqdm.auto import tqdm

glue_qqp_dir = '/data/QQP/'
glove_path = '/data/glove.6B.50d.txt'


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(-0.5 * torch.pow(x - self.mu, 2) / np.power(self.sigma, 2))


class KNRM(torch.nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, freeze_embeddings: bool, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
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
            return torch.nn.Sequential(torch.nn.Linear(self.kernel_num, 1))
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
        query = torch.clamp(inputs['query'], min=0, max=self.embeddings.num_embeddings-1)
        doc = torch.clamp(inputs['document'], min=0, max=self.embeddings.num_embeddings-1)

        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(self.embeddings(query), self.embeddings(doc))
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out


class RankingDataset(torch.utils.data.Dataset):
    def __init__(
            self, index_pairs_or_triplets: List[List[Union[str, float]]],
            idx_to_text_mapping: Dict[str, str], vocab: Dict[str, int], oov_val: int,
            preproc_func: Callable, max_len: int = 3
    ):
        self.index_pairs_or_triplets = index_pairs_or_triplets
        self.idx_to_text_mapping = idx_to_text_mapping
        self.vocab = vocab
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

    def __getitem__(self, idx: int):
        pass


class TrainTripletsDataset(RankingDataset):
    def __getitem__(self, idx):
        query, doc_1, doc_2, rel = self.index_pairs_or_triplets[idx]
        tokens_q = self._convert_text_idx_to_token_idxs(query)
        tokens_d1 = self._convert_text_idx_to_token_idxs(doc_1)
        tokens_d2 = self._convert_text_idx_to_token_idxs(doc_2)
        return (
            {'query': tokens_q, 'document': tokens_d1},
            {'query': tokens_q, 'document': tokens_d2}, float(rel)
        )


class ValPairsDataset(RankingDataset):
    def __getitem__(self, idx):
        idx1, idx2, rel = self.index_pairs_or_triplets[idx]
        tokens1 = self._convert_text_idx_to_token_idxs(idx1)
        tokens2 = self._convert_text_idx_to_token_idxs(idx2)
        return {'query': tokens1, 'document': tokens2}, float(rel)


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


class Solution:
    def __init__(
            self, glue_qqp_dir: str = 'data/QQP/',
            glove_vectors_path: str = 'data/glove.6B.50d.txt',
            min_token_occurancies: int = 1,
            random_seed: int = 0,
            emb_rand_uni_bound: float = 0.2,
            freeze_knrm_embeddings: bool = True,
            knrm_kernel_num: int = 21,
            knrm_out_mlp: List[int] = [10, 5],
            dataloader_bs: int = 1024,
            train_lr: float = 0.001,
            change_train_loader_ep: int = 1
    ):
        self.glue_qqp_dir = glue_qqp_dir
        self.glove_vectors_path = glove_vectors_path
        self.glue_train_df = self.get_glue_df('train')
        self.glue_dev_df = self.get_glue_df('dev')
        self.dev_pairs_for_ndcg = self.create_val_pairs(self.glue_dev_df)
        self.all_triple = pd.DataFrame()
        self.min_token_occurancies = min_token_occurancies
        self.all_tokens = self.get_all_tokens(
            [self.glue_train_df, self.glue_dev_df], self.min_token_occurancies
        )
        self.random_seed = random_seed
        self.emb_rand_uni_bound = emb_rand_uni_bound
        self.freeze_knrm_embeddings = freeze_knrm_embeddings
        self.knrm_kernel_num = knrm_kernel_num
        self.knrm_out_mlp = knrm_out_mlp
        self.dataloader_bs = dataloader_bs
        self.train_lr = train_lr
        self.change_train_loader_ep = change_train_loader_ep

        self.model, self.vocab, self.unk_words = self.build_knrm_model()
        self.idx_to_text_mapping_train = self.get_idx_to_text_mapping(self.glue_train_df)
        self.idx_to_text_mapping_dev = self.get_idx_to_text_mapping(self.glue_dev_df)

        self.val_dataset = ValPairsDataset(
            self.dev_pairs_for_ndcg, self.idx_to_text_mapping_dev, vocab=self.vocab, oov_val=self.vocab['OOV'],
            preproc_func=self.simple_preproc
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.dataloader_bs, num_workers=0, collate_fn=collate_fn, shuffle=False
        )

    def get_glue_df(self, partition_type: str) -> pd.DataFrame:
        assert partition_type in ['dev', 'train']
        glue_df = pd.read_csv(
            self.glue_qqp_dir + f'/{partition_type}.tsv', sep='\t', error_bad_lines=False, dtype=object)
        glue_df = glue_df.dropna(axis=0, how='any').reset_index(drop=True)
        glue_df_fin = pd.DataFrame({
            'id_left': glue_df['qid1'],
            'id_right': glue_df['qid2'],
            'text_left': glue_df['question1'],
            'text_right': glue_df['question2'],
            'label': glue_df['is_duplicate'].astype(int)
        })
        return glue_df_fin

    def hadle_punctuation(self, inp_str: str) -> str:
        punct = string.punctuation
        return inp_str.translate(str.maketrans(punct, ' '*len(punct)))

    def simple_preproc(self, inp_str: str) -> List[str]:
        return word_tokenize(self.hadle_punctuation(inp_str.lower()))

    def _filter_rare_words(self, vocab: Dict[str, int], min_occurancies: int) -> Dict[str, int]:
        return {w: vocab[w] for w in vocab.keys() if vocab[w] >= min_occurancies}

    def get_all_tokens(self, list_of_df: List[pd.DataFrame], min_occurancies: int) -> List[str]:
        data = pd.concat(
            [df['text_left'] for df in list_of_df]
            + [df['text_right'] for df in list_of_df]
        )
        data = data.drop_duplicates()
        data = self.simple_preproc(' '.join(data.tolist()))
        cnt = Counter(data)
        cnt = self._filter_rare_words(cnt, min_occurancies)

        return list(cnt.keys())

    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        all_dict = {}
        with open(file_path) as f:
            for line in f:
                lst = line.split(' ')
                inner_list = [float(elt.strip()) for elt in lst[1:]]
                all_dict[lst[0]] = inner_list
        return all_dict

    def create_glove_emb_from_file(
            self, file_path: str, inner_keys: List[str],
            random_seed: int, rand_uni_bound: float
    ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:

        embs = self._read_glove_embeddings(file_path)
        emb_size = len(embs[list(embs.keys())[0]])
        np.random.seed(random_seed)
        unk_words = []
        output_dict = {}
        output_mat = []
        idx = 0
        for word in ['PAD', 'OOV']:
            output_dict[word] = idx
            idx += 1
            unk_words += [word]
            if word == 'OOV':
                vect = np.random.random(size=emb_size)
                vect -= 0.5
                vect *= 2 * rand_uni_bound
            else:
                vect = np.zeros(emb_size)
            output_mat += [vect]

        for word in inner_keys:
            if word in embs.keys():
                if word not in output_dict.keys():
                    output_dict[word] = idx
                    idx += 1
                    output_mat += [embs[word]]
            else:
                output_dict[word] = 1
                unk_words += [word]

        return np.array(output_mat), output_dict, unk_words

    def build_knrm_model(self) -> Tuple[torch.nn.Module, Dict[str, int], List[str]]:
        emb_matrix, vocab, unk_words = self.create_glove_emb_from_file(
            self.glove_vectors_path, self.all_tokens, self.random_seed, self.emb_rand_uni_bound
        )
        torch.manual_seed(self.random_seed)
        knrm = KNRM(emb_matrix, freeze_embeddings=self.freeze_knrm_embeddings,
                    out_layers=self.knrm_out_mlp, kernel_num=self.knrm_kernel_num)
        return knrm, vocab, unk_words

    def sample_data_for_train_iter(self, inp_df: pd.DataFrame, seed: int) -> List[List[Union[str, float]]]:
        if self.all_triple.empty:
            temp_agg_1 = inp_df.groupby('id_left')['label'].nunique().reset_index()
            id_left = set(temp_agg_1.loc[temp_agg_1['label'] > 1, 'id_left'])
            temp_agg_2 = inp_df.groupby('id_right')['label'].nunique().reset_index()
            id_right = set(temp_agg_2.loc[temp_agg_2['label'] > 1, 'id_right'])

            temp_1 = inp_df[inp_df['id_left'].isin(id_left)]
            temp_2 = inp_df[inp_df['id_right'].isin(id_right)]
            temp_1_ = temp_1.copy()
            temp_2_ = temp_2.copy()
            temp_1_['label'] = 1 - temp_1_['label']
            temp_2_['label'] = 1 - temp_2_['label']

            merged = pd.merge(
                temp_1, temp_1_,
                on=['id_left', 'label'], how='left'
            )
            triple_1 = merged[['id_left', 'id_right_x', 'id_right_y', 'label']]

            merged = pd.merge(
                temp_2, temp_2_,
                on=['id_right', 'label'], how='left'
            )
            triple_2 = merged[['id_right', 'id_left_x', 'id_left_y', 'label']]
            triple_1.columns = ['query', 'doc_1', 'doc_2', 'rel']
            triple_2.columns = ['query', 'doc_1', 'doc_2', 'rel']
            self.all_triple = pd.concat([triple_1, triple_2], axis=0).drop_duplicates()
            self.all_triple['rel'] = self.all_triple['rel'].astype('float')

        return self.all_triple.sample(frac=0.1, random_state=seed).values.tolist()

    def create_val_pairs(
            self, inp_df: pd.DataFrame, fill_top_to: int = 15, min_group_size: int = 2, seed: int = 0
    ) -> List[List[Union[str, float]]]:
        inp_df_select = inp_df[['id_left', 'id_right', 'label']]
        inf_df_group_sizes = inp_df_select.groupby('id_left').size()
        glue_dev_leftids_to_use = list(
            inf_df_group_sizes[inf_df_group_sizes >= min_group_size].index)
        groups = inp_df_select[inp_df_select.id_left.isin(
            glue_dev_leftids_to_use)].groupby('id_left')

        all_ids = set(inp_df['id_left']).union(set(inp_df['id_right']))

        out_pairs = []

        np.random.seed(seed)

        for id_left, group in groups:
            ones_ids = group[group.label > 0].id_right.values
            zeroes_ids = group[group.label == 0].id_right.values
            sum_len = len(ones_ids) + len(zeroes_ids)
            num_pad_items = max(0, fill_top_to - sum_len)
            if num_pad_items > 0:
                cur_chosen = set(ones_ids).union(
                    set(zeroes_ids)).union({id_left})
                pad_sample = np.random.choice(
                    list(all_ids - cur_chosen), num_pad_items, replace=False).tolist()
            else:
                pad_sample = []
            for i in ones_ids:
                out_pairs.append([id_left, i, 2.])
            for i in zeroes_ids:
                out_pairs.append([id_left, i, 1.])
            for i in pad_sample:
                out_pairs.append([id_left, i, 0.])
        return out_pairs

    def get_idx_to_text_mapping(self, inp_df: pd.DataFrame) -> Dict[str, str]:
        left_dict = (
            inp_df
            [['id_left', 'text_left']]
                .drop_duplicates()
                .set_index('id_left')
            ['text_left']
                .to_dict()
        )
        right_dict = (
            inp_df
            [['id_right', 'text_right']]
                .drop_duplicates()
                .set_index('id_right')
            ['text_right']
                .to_dict()
        )
        left_dict.update(right_dict)
        return left_dict

    def _dcg_k(self, ys_true, ys_pred, k):
        argsort = np.argsort(-ys_pred)
        ys_true_sorted = np.take(ys_true, argsort)[:k]
        ret = 0.
        for i, l in enumerate(ys_true_sorted, 1):
            ret += (2 ** l - 1) / np.log2(1 + i)
        return ret

    def ndcg_k(self, ys_true: np.array, ys_pred: np.array, ndcg_top_k: int = 10) -> float:
        ideal_dcg = self._dcg_k(ys_true, ys_true, ndcg_top_k)
        pred_dcg = self._dcg_k(ys_true, ys_pred, ndcg_top_k)
        return pred_dcg / ideal_dcg

    def valid(self, model: torch.nn.Module, val_dataloader: torch.utils.data.DataLoader) -> float:
        labels_and_groups = val_dataloader.dataset.index_pairs_or_triplets
        labels_and_groups = pd.DataFrame(labels_and_groups, columns=['left_id', 'right_id', 'rel'])

        all_preds = []
        model.eval()
        for batch in tqdm(val_dataloader):
            inp_1, y = batch
            preds = model.predict(inp_1)
            preds_np = preds.detach().numpy()
            all_preds.append(preds_np)
        all_preds = np.concatenate(all_preds, axis=0)
        labels_and_groups['preds'] = all_preds

        ndcgs = []
        for cur_id in labels_and_groups.left_id.unique():
            cur_df = labels_and_groups[labels_and_groups.left_id == cur_id]
            ndcg = self.ndcg_k(cur_df.rel.values.reshape(-1), cur_df.preds.values.reshape(-1))

            if np.isnan(ndcg):
                ndcgs.append(0.)
            else:
                ndcgs.append(ndcg)
        return float(sum(ndcgs)/len(ndcgs))

    def train(self, n_epochs: int):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.train_lr)
        criterion = torch.nn.BCELoss()

        for i in tqdm(list(range(n_epochs))):
            self.model.train()
            if i % self.change_train_loader_ep == 0:
                triplets = self.sample_data_for_train_iter(self.glue_train_df, i)
                train_dataset = TrainTripletsDataset(triplets,
                                                     self.idx_to_text_mapping_train,
                                                     vocab=self.vocab, oov_val=self.vocab['OOV'],
                                                     preproc_func=self.simple_preproc)
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=self.dataloader_bs, num_workers=0,
                    collate_fn=collate_fn, shuffle=True)
            for batch in tqdm(train_dataloader):
                input_1, input_2, rel = batch
                preds = self.model(input_1, input_2)
                batch_loss = criterion(preds.float(), rel.float())
                batch_loss.backward()
                opt.step()
        return None
