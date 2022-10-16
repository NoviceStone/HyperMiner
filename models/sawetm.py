import torch
import torch.nn as nn
import torch.nn.functional as F
from models.block import ResBlock


class SawETM(nn.Module):
    """Simple implementation of the <<Sawtooth Factorial Topic Embeddings Guided Gamma Belief Network>>

    Args
        args: the set of arguments used to characterize the hierarchical neural topic model.
        device: the physical hardware that the model is trained on.
        pretrained_embeddings: if not None, initialize each word embedding in the vocabulary with pretrained Glove embeddings.
    """

    def __init__(self, args, device, word_embeddings):
        super(SawETM, self).__init__()
        # constants
        self.device = device
        self.gam_prior = torch.tensor(1.0, dtype=torch.float, device=device)
        self.real_min = torch.tensor(1e-30, dtype=torch.float, device=device)
        self.theta_max = torch.tensor(1000.0, dtype=torch.float, device=device)
        self.wei_shape_min = torch.tensor(1e-1, dtype=torch.float, device=device)
        self.wei_shape_max = torch.tensor(100.0, dtype=torch.float, device=device)

        # hyper-parameters
        self.embed_size = args.embed_size
        self.vocab_size = args.vocab_size
        self.num_topics_list = args.num_topics_list
        self.num_hiddens_list = args.num_hiddens_list
        assert len(args.num_topics_list) == len(args.num_hiddens_list)
        self.num_layers = len(args.num_topics_list)

        # learnable word embeddings
        if word_embeddings is not None:
            self.rho = nn.Parameter(torch.from_numpy(word_embeddings).float())
        else:
            self.rho = nn.Parameter(
                torch.empty(args.vocab_size, args.embed_size).normal_(std=0.02))

        # topic embeddings for different latent layers
        self.alpha = nn.ParameterList([])
        for n in range(self.num_layers):
            self.alpha.append(nn.Parameter(
                torch.empty(args.num_topics_list[n], args.embed_size).normal_(std=0.02)))

        # deterministic mapping to obtain hierarchical features
        self.h_encoder = nn.ModuleList([])
        for n in range(self.num_layers):
            if n == 0:
                self.h_encoder.append(
                    ResBlock(args.vocab_size, args.num_hiddens_list[n]))
            else:
                self.h_encoder.append(
                    ResBlock(args.num_hiddens_list[n - 1], args.num_hiddens_list[n]))

        # variational encoder to obtain posterior parameters
        self.q_theta = nn.ModuleList([])
        for n in range(self.num_layers):
            if n == self.num_layers - 1:
                self.q_theta.append(
                    nn.Linear(args.num_hiddens_list[n], 2 * args.num_topics_list[n]))
            else:
                self.q_theta.append(nn.Linear(
                    args.num_hiddens_list[n] + args.num_topics_list[n], 2 * args.num_topics_list[n]))

    def log_max(self, x):
        return torch.log(torch.max(x, self.real_min))

    def reparameterize(self, shape, scale, sample_num=50):
        """Returns a sample from a Weibull distribution via reparameterization.
        """
        shape = shape.unsqueeze(0).repeat(sample_num, 1, 1)
        scale = scale.unsqueeze(0).repeat(sample_num, 1, 1)
        eps = torch.rand_like(shape, dtype=torch.float, device=self.device)
        samples = scale * torch.pow(- self.log_max(1 - eps), 1 / shape)
        return torch.clamp(samples.mean(0), self.real_min.item(), self.theta_max.item())

    def kl_weibull_gamma(self, wei_shape, wei_scale, gam_shape, gam_scale):
        """Returns the Kullback-Leibler divergence between a Weibull distribution and a Gamma distribution.
        """
        euler_mascheroni_c = torch.tensor(0.5772, dtype=torch.float, device=self.device)
        t1 = torch.log(wei_shape) + torch.lgamma(gam_shape)
        t2 = - gam_shape * torch.log(wei_scale * gam_scale)
        t3 = euler_mascheroni_c * (gam_shape / wei_shape - 1) - 1
        t4 = gam_scale * wei_scale * torch.exp(torch.lgamma(1 + 1 / wei_shape))
        return (t1 + t2 + t3 + t4).sum(1).mean()

    def get_nll(self, x, x_reconstruct):
        """Returns the negative Poisson likelihood of observational count data.
        """
        log_likelihood = self.log_max(x_reconstruct) * x - torch.lgamma(1.0 + x) - x_reconstruct
        neg_log_likelihood = - torch.sum(log_likelihood, dim=1, keepdim=False).mean()
        return neg_log_likelihood

    def get_phi(self):
        """Returns the factor loading matrix by utilizing sawtooth connection.
        """
        phis = []
        for n in range(self.num_layers):
            if n == 0:
                phi = torch.softmax(torch.mm(
                    self.rho, self.alpha[n].transpose(0, 1)), dim=0)
            else:
                phi = torch.softmax(torch.mm(
                    self.alpha[n - 1].detach(), self.alpha[n].transpose(0, 1)), dim=0)
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
