import torch
import torch.nn as nn
import torch.nn.functional as F

import manifolds
from models.etm import ETM
from models.sawetm import SawETM


class HyperETM(ETM):
    """A variant of ETM that embeds words and topic into hyperbolic space to measure their distance
    """

    def __init__(self, args, device, word_embeddings):
        super(HyperETM, self).__init__(args, device, word_embeddings)
        self.manifold = getattr(manifolds, args.manifold)()
        if args.c is not None:
            self.curvature = torch.tensor([args.c])
            self.curvature = self.curvature.to(device)
        else:
            self.curvature = nn.Parameter(torch.Tensor([-1.]))

        # the effective radius used to clip Euclidean features
        self.clip_r = args.clip_r

    def feat_clip(self, x):
        """Use feature clipping technique to avoid the gradient vanishing problem"""
        x_norm = x.norm(p=2, dim=-1, keepdim=True)
        cond = x_norm > self.clip_r
        projected = x / x_norm * self.clip_r
        return torch.where(cond, projected, x)

    def get_phi(self):
        """Derives the topic-word matrix by the distance in hyperbolic space.
        """
        hyp_rho = self.manifold.proj(
            # self.manifold.expmap0(self.feat_clip(self.rho), self.curvature),
            self.manifold.expmap0(self.rho, self.curvature),
            self.curvature
        )
        hyp_alpha = self.manifold.proj(
            # self.manifold.expmap0(self.feat_clip(self.alpha), self.curvature),
            self.manifold.expmap0(self.alpha, self.curvature),
            self.curvature
        )
        return torch.softmax(-self.manifold.dist(
            hyp_rho.unsqueeze(1), hyp_alpha.unsqueeze(0), self.curvature
        ), dim=0)

    def forward(self, x, is_training=True):
        """Forward pass: compute the kl loss and data likelihood.
        """
        denorm = torch.where(x.sum(dim=1, keepdims=True) > 0, x.sum(dim=1, keepdims=True), torch.tensor([1.]).cuda())
        norm_x = x / denorm

        hidden_feats = self.h_encoder(norm_x)
        mu, logvar = torch.chunk(self.q_theta(hidden_feats), 2, dim=1)
        kl_loss = self.kl_normal_normal(mu, logvar)
        if is_training:
            theta = torch.softmax(self.reparameterize(mu, logvar), dim=1)
        else:
            theta = torch.softmax(mu, dim=1)

        # =================================================================================
        phi = self.get_phi()
        logit = torch.mm(theta, phi.t())
        almost_zeros = torch.full_like(logit, 1e-6)
        results_without_zeros = logit.add(almost_zeros)
        predictions = torch.log(results_without_zeros)
        recon_loss = -(predictions * x).sum(1).mean()

        nelbo = recon_loss + 0.5 * kl_loss
        return nelbo, recon_loss, kl_loss, theta


class HyperMiner(SawETM):
    """A variant of SawETM that embeds words and topic into hyperbolic space to measure their distance
    """

    def __init__(self, args, device, word_embeddings):
        super(HyperMiner, self).__init__(args, device, word_embeddings)
        self.manifold = getattr(manifolds, args.manifold)()
        if args.c is not None:
            self.curvature = torch.tensor([args.c])
            self.curvature = self.curvature.to(device)
        else:
            self.curvature = nn.Parameter(torch.Tensor([-1.]))

        self.clip_r = args.clip_r

    def feat_clip(self, x):
        x_norm = x.norm(p=2, dim=-1, keepdim=True)
        cond = x_norm > self.clip_r
        projected = x / x_norm * self.clip_r
        return torch.where(cond, projected, x)

    def get_phi(self):
        """Returns the factor loading matrix by utilizing sawtooth connection.
        """
        phis = []
        for n in range(self.num_layers):
            if n == 0:
                hyp_rho = self.manifold.proj(
                    self.manifold.expmap0(self.rho, self.curvature), self.curvature)
                hyp_alpha = self.manifold.proj(
                    self.manifold.expmap0(self.alpha[n], self.curvature), self.curvature)
                phi = torch.softmax(-self.manifold.dist(
                    hyp_rho.unsqueeze(1), hyp_alpha.unsqueeze(0), self.curvature), dim=0)
            else:
                hyp_alpha1 = self.manifold.proj(
                    self.manifold.expmap0(self.alpha[n - 1], self.curvature), self.curvature)
                hyp_alpha2 = self.manifold.proj(
                    self.manifold.expmap0(self.alpha[n], self.curvature), self.curvature)
                phi = torch.softmax(-self.manifold.dist(
                    hyp_alpha1.unsqueeze(1).detach(), hyp_alpha2.unsqueeze(0), self.curvature), dim=0)
            phis.append(phi)
        return phis

    def forward(self, x, is_training=True):
        """Forward pass: compute the kl loss and data likelihood.
        """
        hidden_feats = []
        for n in range(self.num_layers):
            if n == 0:
                hidden_feats.append(self.h_encoder[n](x))
            else:
                hidden_feats.append(self.h_encoder[n](hidden_feats[-1]))

        # =================================================================================
        phis = self.get_phi()

        ks = []
        lambs = []
        thetas = []
        phi_by_theta_list = []
        for n in range(self.num_layers - 1, -1, -1):
            if n == self.num_layers - 1:
                joint_feat = hidden_feats[n]
            else:
                joint_feat = torch.cat((hidden_feats[n], phi_by_theta_list[0]), dim=1)

            k, lamb = torch.chunk(F.softplus(self.q_theta[n](joint_feat)), 2, dim=1)
            k = torch.clamp(k, self.wei_shape_min.item(), self.wei_shape_max.item())
            lamb = torch.clamp(lamb, self.real_min.item())

            if is_training:
                lamb = lamb / torch.exp(torch.lgamma(1 + 1 / k))
                theta = self.reparameterize(k, lamb, sample_num=3) if n == 0 else self.reparameterize(k, lamb)
            else:
                theta = torch.min(lamb, self.theta_max)

            phi_by_theta = torch.mm(theta, phis[n].t())
            phi_by_theta_list.insert(0, phi_by_theta)
            thetas.insert(0, theta)
            lambs.insert(0, lamb)
            ks.insert(0, k)

        # =================================================================================
        nll = self.get_nll(x, phi_by_theta_list[0])

        kl_loss = []
        for n in range(self.num_layers):
            if n == self.num_layers - 1:
                kl_loss.append(self.kl_weibull_gamma(
                    ks[n], lambs[n], self.gam_prior, self.gam_prior))
            else:
                kl_loss.append(self.kl_weibull_gamma(
                    ks[n], lambs[n], phi_by_theta_list[n + 1], self.gam_prior))

        nelbo = nll + sum(kl_loss)
        return nelbo, nll, sum(kl_loss), thetas


class HyperMinerKG(HyperMiner):
    """An improved version of HyperMiner that injects external knowledge to guide the learning of topic taxonomy
    """

    def __init__(self, args, device, word_embeddings, adjacent_mat):
        super(HyperMinerKG, self).__init__(args, device, word_embeddings)
        self.manifold = getattr(manifolds, args.manifold)()
        if args.c is not None:
            self.curvature = torch.tensor([args.c])
            self.curvature = self.curvature.to(device)
        else:
            self.curvature = nn.Parameter(torch.Tensor([-1.]))

        self.clip_r = args.clip_r
        self.adj = adjacent_mat.to(device)
        self.split_sections = self.num_topics_list[:: -1] + [self.vocab_size]
        self.inp_embeddings = nn.Parameter(
            torch.empty(sum(self.split_sections), args.embed_size).normal_(std=0.02))

        del self.rho, self.alpha
        self.temp = 1.0

    def get_phi(self):
        """Returns the factor loading matrix according to hyperbolic distance.
        """
        hyp_embeddings = self.manifold.proj(
            self.manifold.expmap0(self.inp_embeddings, self.curvature), self.curvature)
        hyp_embeddings[0] = torch.tensor([0.35, 0.35]).to(self.device)
        hyp_embeddings[1] = torch.tensor([-0.35, -0.35]).to(self.device)

        N = sum(self.num_topics_list)
        dist_mat = self.manifold.dist(
            hyp_embeddings[: N].unsqueeze(1),
            hyp_embeddings.unsqueeze(0),
            self.curvature
        )

        phis = []
        for n in range(self.num_layers):
            x_start = sum(self.split_sections[: self.num_layers - n - 1])
            x_end = sum(self.split_sections[: self.num_layers - n])
            y_start = sum(self.split_sections[: self.num_layers - n])
            y_end = sum(self.split_sections[: self.num_layers - n + 1])
            phi = torch.softmax(
                -dist_mat[x_start: x_end, y_start: y_end].t(), dim=0)
            phis.append(phi)
        return phis, dist_mat

    def contrastive_loss(self, dist_mat, K=256):
        """Hyperbolic contrastive loss to maintain the semantic structure information.
        """
        N = sum(self.num_topics_list)
        adj_dense = self.adj.to_dense()
        neg_adj = torch.ones_like(adj_dense) - adj_dense
        pos_loss = torch.exp(-(adj_dense[: N] * dist_mat).max(1)[0] / self.temp)
        # pos_loss = torch.exp(-(adj_dense[: N] * dist_mat) / self.temp).sum(1)

        # 1. sampling K negatives from the set of non-first-order neighbors
        neg_dist = (neg_adj[: N] * dist_mat)
        neg_dist = torch.where(neg_dist > 1e-6, neg_dist, torch.tensor(1000, dtype=torch.float32).to(self.device))
        neg_loss = torch.exp(-neg_dist.topk(K, dim=1, largest=False)[0] / self.temp).sum(1)

        # 2. consider all non-first-order neighbors as negatives
        # neg_loss = (neg_adj[: N] * torch.exp(-dist_mat / self.temp)).sum(1)

        nce_loss = torch.log(pos_loss + neg_loss + self.real_min) - torch.log(pos_loss + self.real_min)
        return nce_loss.mean()

    def forward(self, x, is_training=True):
        """Forward pass: compute the kl loss and data likelihood.
        """
        hidden_feats = []
        for n in range(self.num_layers):
            if n == 0:
                hidden_feats.append(self.h_encoder[n](x))
            else:
                hidden_feats.append(self.h_encoder[n](hidden_feats[-1]))

        # =================================================================================
        phis, dist_mat = self.get_phi()
        contrast_loss = self.contrastive_loss(dist_mat)

        ks = []
        lambs = []
        thetas = []
        phi_by_theta_list = []
        for n in range(self.num_layers-1, -1, -1):
            if n == self.num_layers - 1:
                joint_feat = hidden_feats[n]
            else:
                joint_feat = torch.cat((hidden_feats[n], phi_by_theta_list[0]), dim=1)

            k, lamb = torch.chunk(F.softplus(self.q_theta[n](joint_feat)), 2, dim=1)
            k = torch.clamp(k, self.wei_shape_min.item(), self.wei_shape_max.item())
            lamb = torch.clamp(lamb, self.real_min.item())

            if is_training:
                lamb = lamb / torch.exp(torch.lgamma(1 + 1 / k))
                theta = self.reparameterize(k, lamb, sample_num=5) if n == 0 else self.reparameterize(k, lamb)
            else:
                theta = torch.min(lamb, self.theta_max)

            phi_by_theta = torch.mm(theta, phis[n].t())
            phi_by_theta_list.insert(0, phi_by_theta)
            thetas.insert(0, theta)
            lambs.insert(0, lamb)
            ks.insert(0, k)

        # =================================================================================
        nll = self.get_nll(x, phi_by_theta_list[0])

        kl_loss = []
        for n in range(self.num_layers):
            if n == self.num_layers - 1:
                kl_loss.append(self.kl_weibull_gamma(
                    ks[n], lambs[n], self.gam_prior, self.gam_prior))
            else:
                kl_loss.append(self.kl_weibull_gamma(
                    ks[n], lambs[n], phi_by_theta_list[n+1], self.gam_prior))

        nelbo = nll + 0.2 * sum(kl_loss) + 5 * contrast_loss
        return nelbo, nll, contrast_loss, thetas
