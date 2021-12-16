# Author: Jacek Komorowski
# Warsaw University of Technology

import numpy as np
import torch
from pytorch_metric_learning import losses, miners, reducers
from pytorch_metric_learning.distances import LpDistance


def make_loss(params):
    if params.loss == 'BatchHardTripletMarginLoss':
        # BatchHard mining with triplet margin loss
        # Expects input: embeddings, positives_mask, negatives_mask
        loss_fn = BatchHardTripletLossWithMasks(params.margin, params.normalize_embeddings)
    elif params.loss == 'MultiBatchHardTripletMarginLoss':
        # BatchHard mining with triplet margin loss
        # Expects input: embeddings, positives_mask, negatives_mask
        loss_fn = MultiBatchHardTripletLossWithMasks(params.margin, params.normalize_embeddings, params.weights)
    else:
        print('Unknown loss: {}'.format(params.loss))
        raise NotImplementedError
    return loss_fn


class HardTripletMinerWithMasks:
    # Hard triplet miner
    def __init__(self, distance):
        self.distance = distance
        # Stats
        self.max_pos_pair_dist = None
        self.max_neg_pair_dist = None
        self.mean_pos_pair_dist = None
        self.mean_neg_pair_dist = None
        self.min_pos_pair_dist = None
        self.min_neg_pair_dist = None

    def __call__(self, embeddings, positives_mask, negatives_mask):
        assert embeddings.dim() == 2
        d_embeddings = embeddings.detach()
        with torch.no_grad():
            hard_triplets = self.mine(d_embeddings, positives_mask, negatives_mask)
        return hard_triplets

    def mine(self, embeddings, positives_mask, negatives_mask):
        # Based on pytorch-metric-learning implementation
        dist_mat = self.distance(embeddings)
        (hardest_positive_dist, hardest_positive_indices), a1p_keep = get_max_per_row(dist_mat, positives_mask)
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = get_min_per_row(dist_mat, negatives_mask)
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        a = torch.arange(dist_mat.size(0)).to(hardest_positive_indices.device)[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]
        self.max_pos_pair_dist = torch.max(hardest_positive_dist).item()
        self.max_neg_pair_dist = torch.max(hardest_negative_dist).item()
        self.mean_pos_pair_dist = torch.mean(hardest_positive_dist).item()
        self.mean_neg_pair_dist = torch.mean(hardest_negative_dist).item()
        self.min_pos_pair_dist = torch.min(hardest_positive_dist).item()
        self.min_neg_pair_dist = torch.min(hardest_negative_dist).item()
        return a, p, n


def get_max_per_row(mat, mask):
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = 0
    return torch.max(mat_masked, dim=1), non_zero_rows


def get_min_per_row(mat, mask):
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = float('inf')
    return torch.min(mat_masked, dim=1), non_inf_rows


class MultiBatchHardTripletLossWithMasks:
    def __init__(self, margin, normalize_embeddings, weights):
        assert len(weights) == 3
        self.weights = weights
        self.final_loss = BatchHardTripletLossWithMasksHelper(margin, normalize_embeddings)
        self.cloud_loss = BatchHardTripletLossWithMasksHelper(margin, normalize_embeddings)
        self.image_loss = BatchHardTripletLossWithMasksHelper(margin, normalize_embeddings)
        print('MultiBatchHardTripletLossWithMasks')
        print('Weights (final/cloud/image): {}'.format(weights))

    def __call__(self, x, positives_mask, negatives_mask):
        # Loss on the final global descriptor
        final_loss, final_stats, final_hard_triplets = self.final_loss(x['embedding'], positives_mask, negatives_mask)
        final_stats = {'final_{}'.format(e): final_stats[e] for e in final_stats}

        loss = 0.

        stats = final_stats
        if self.weights[0] > 0.:
            loss = self.weights[0] * final_loss + loss

        # Loss on the cloud-based descriptor
        if 'cloud_embedding' in x:
            cloud_loss, cloud_stats, _ = self.cloud_loss(x['cloud_embedding'], positives_mask, negatives_mask)
            cloud_stats = {'cloud_{}'.format(e): cloud_stats[e] for e in cloud_stats}
            stats.update(cloud_stats)
            if self.weights[1] > 0.:
                loss = self.weights[1] * cloud_loss + loss

        # Loss on the image-based descriptor
        if 'image_embedding' in x:
            image_loss, image_stats, _ = self.image_loss(x['image_embedding'], positives_mask, negatives_mask)
            image_stats = {'image_{}'.format(e): image_stats[e] for e in image_stats}
            stats.update(image_stats)
            if self.weights[2] > 0.:
                loss = self.weights[2] * image_loss + loss

        stats['loss'] = loss.item()
        return loss, stats, None


class BatchHardTripletLossWithMasks:
    def __init__(self, margin, normalize_embeddings):
        self.loss_fn = BatchHardTripletLossWithMasksHelper(margin, normalize_embeddings)

    def __call__(self, x, positives_mask, negatives_mask):
        embeddings = x['embedding']
        return self.loss_fn(embeddings, positives_mask, negatives_mask)


class BatchHardTripletLossWithMasksHelper:
    def __init__(self, margin, normalize_embeddings):
        self.margin = margin
        self.distance = LpDistance(normalize_embeddings=normalize_embeddings, collect_stats=True)
        # We use triplet loss with Euclidean distance
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        reducer_fn = reducers.AvgNonZeroReducer(collect_stats=True)
        self.loss_fn = losses.TripletMarginLoss(margin=self.margin, swap=True, distance=self.distance,
                                                reducer=reducer_fn, collect_stats=True)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)
        stats = {'loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_non_zero_triplets': self.loss_fn.reducer.triplets_past_filter,
                 'num_triplets': len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist,
                 'normalized_loss': loss.item() * self.loss_fn.reducer.triplets_past_filter,
                 # total loss per batch
                 'total_loss': self.loss_fn.reducer.loss * self.loss_fn.reducer.triplets_past_filter
                 }

        return loss, stats, hard_triplets
