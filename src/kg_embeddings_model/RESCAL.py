"based on the implementation of https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/RESCAL.py"
import torch
import torch.autograd
import torch.nn as nn
import tensorflow as tf
from utilities.constants import EMBEDDING_DIM, MARGIN_LOSS, NUM_ENTITIES, NUM_RELATIONS, NORMALIZATION_OF_ENTITIES, \
    RESCAL, PREFERRED_DEVICE, GPU, CPU

class RESCAL(nn.Module):
    def __init__(self,config):
        super(RESCAL, self).__init__()
        self.model_name = RESCAL
        # A simple lookup table that stores embeddings of a fixed dictionary and size
        num_entities = config[NUM_ENTITIES]
        num_relations = config[NUM_RELATIONS]
        self.embedding_dim = config[EMBEDDING_DIM]
        margin_loss = config[MARGIN_LOSS]
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() and config[PREFERRED_DEVICE] == GPU else CPU)
        self.l_p_norm = config[NORMALIZATION_OF_ENTITIES]

        self.entities_embeddings = nn.Embedding(num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, self.embedding_dim)

        self.margin_loss = margin_loss
        self.criterion = nn.MarginRankingLoss(margin=self.margin_loss, size_average=True)

        self._init()

    def _init(self):
            nn.init.xavier_uniform_(self.entities_embeddings.weight.data)
            nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

     # score function of RESCAL
    def _calc(self, h, t, r):
        return h * tf.matmul(r, t)
        #return score
    #Calculate the margin based loss

    def loss_func(self, positive_score, negative_score):

        y = torch.tensor([1], dtype=torch.float, device=self.device)
        pos_score = torch.tensor(positive_score, dtype=torch.float, device=self.device)

        neg_score = torch.tensor(negative_score, dtype=torch.float, device=self.device)
        loss = self.criterion(positive_score, negative_score, y)
        #pos_scores = torch.tensor(positive_score, dtype=torch.float, device=self.device)
        #neg_scores = torch.tensor(negative_score, dtype=torch.float, device=self.device)

        return loss

    #Define the forward function
    def forward(self,pos_exmpls, neg_exmpls):
        pos_heads = pos_exmpls[:, 0:1]
        pos_relations = pos_exmpls[:, 1:2]
        pos_tails = pos_exmpls[:, 2:3]
        neg_heads = neg_exmpls[:, 0:1]
        neg_relations = neg_exmpls[:, 1:2]
        neg_tails = neg_exmpls[:, 2:3]

        pos_h_embs = self.entities_embeddings(pos_heads)
        pos_r_embs = self.relation_embeddings(pos_relations).view(-1, self.embedding_dim)
        pos_t_embs = self.entities_embeddings(pos_tails)

        neg_h_embs = self.entities_embeddings(neg_heads)
        neg_r_embs = self.relation_embeddings(neg_relations).view(-1, self.embedding_dim)
        neg_t_embs = self.entities_embeddings(neg_tails)
        # L-P normalization of the vectors
        pos_h_embs = torch.nn.functional.normalize(pos_h_embs, p=self.l_p_norm, dim=1).view(-1, self.embedding_dim)
        pos_t_embs = torch.nn.functional.normalize(pos_t_embs, p=self.l_p_norm, dim=1).view(-1, self.embedding_dim)
        neg_h_embs = torch.nn.functional.normalize(neg_h_embs, p=self.l_p_norm, dim=1).view(-1, self.embedding_dim)
        neg_t_embs = torch.nn.functional.normalize(neg_t_embs, p=self.l_p_norm, dim=1).view(-1, self.embedding_dim)
        #_p_score = self._calc(pos_h_embs, pos_t_embs, pos_r_embs).view(-1, 1, self.config.hidden_size)
        #_n_score = self._calc(neg_h_embs, neg_t_embs, neg_r_embs).view(-1, 1, self.config.hidden_size)
        pos_score = self._calc(h_emb=pos_h_embs, r_emb=pos_r_embs, t_emb=pos_t_embs)
        neg_score = self._calc(h_emb=neg_h_embs, r_emb=neg_r_embs, t_emb=neg_t_embs)
        p_score = torch.sum(torch.mean(pos_score, 1, False), 1)
        n_score = torch.sum(torch.mean(neg_score, 1, False), 1)
        loss = self.loss_func(p_score, n_score)
        return loss

    def predict(self, triples):
        heads = triples[:, 0:1]
        relations = triples[:, 1:2]
        tails = triples[:, 2:3]

        head_embs = self.entities_embeddings(heads).view(-1, self.embedding_dim)
        relation_embs = self.relation_embeddings(relations).view(-1, self.embedding_dim)
        tail_embs = self.entities_embeddings(tails).view(-1, self.embedding_dim)

        scores = -torch.sum(self._calc(head_embs, relation_embs, tail_embs), 1)
        return scores.detach().cpu().numpy()
