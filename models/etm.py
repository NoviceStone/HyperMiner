import torch
import torch.nn as nn
from models.block import ResBlock, _get_activation_fn


class ETM(nn.Module):
    """Simple implementation of the <<Topic Modeling in Embedding space>>

    Args
        args: the set of arguments used to characterize the hierarchical neural topic model.
        device: the physical hardware that the model is trained on.
        pretrained_embeddings: if not None, initialize each word embedding in the vocabulary with pretrained Glove embeddings.
    """

    def __init__(self, args, device, word_embeddings):
        super(ETM, self).__init__()
        self.device = device

        # hyper-parameters
        self.embed_size = args.embed_size
        self.vocab_size = args.vocab_size
        self.num_topics = args.num_topics_list[0]
        self.num_hiddens = args.num_hiddens_list[0]

        # learnable word embeddings
        if word_embeddings is not None:
            self.rho = nn.Parameter(torch.from_numpy(word_embeddings).float())
        else:
            self.rho = nn.Parameter(
                torch.empty(args.vocab_size, args.embed_size).normal_(std=0.02))

        # topic embeddings for different latent layers
        self.alpha = nn.Parameter(
                torch.empty(self.num_topics, args.embed_size).normal_(std=0.02))

        # deterministic mapping to obtain hidden features
        self.h_encoder = ResBlock(self.vocab_size, self.num_hiddens, args.act)

        # variational encoder to obtain posterior parameters
        self.q_theta = nn.Linear(self.num_hiddens, 2 * self.num_topics)

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul_(std).add_(mu)

    def kl_normal_normal(self, mu_theta, logsigma_theta):
        """Returns the Kullback-Leibler divergence between a normal distribution and a standard normal distribution.
        """
        kl_div = -0.5 * torch.sum(
            1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1
        )
        return kl_div.mean()

    def get_phi(self):
        """Derives the topic-word matrix by computing the inner product.
        """
        dist = torch.mm(self.rho, self.alpha.transpose(0, 1))
        phi = torch.softmax(dist, dim=0)
        return phi

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

        nelbo = recon_loss + kl_loss
        return nelbo, recon_loss, kl_loss, theta


class VanillaETM(nn.Module):

    def __init__(self, args, device, word_embeddings):
        super(VanillaETM, self).__init__()
        self.device = device

        # hyper-parameters
        self.embed_size = args.embed_size
        self.vocab_size = args.vocab_size
        self.num_topics = args.num_topics_list[0]
        self.num_hiddens = args.num_hiddens_list[0]

        # learnable word embeddings
        if word_embeddings is not None:
            self.rho = nn.Parameter(torch.from_numpy(word_embeddings).float())
        else:
            self.rho = nn.Linear(args.embed_size, args.vocab_size, bias=False)

        # topic embeddings for different latent layers
        self.alpha = nn.Linear(args.embed_size, self.num_topics, bias=False)

        # deterministic mapping to obtain hierarchical features
        self.activation = _get_activation_fn(args.act)
        self.h_encoder = nn.Sequential(
            nn.Linear(self.vocab_size, self.num_hiddens),
            self.activation,
            nn.Linear(self.num_hiddens, self.num_hiddens),
            self.activation,
            nn.Dropout(args.dropout)
        )

        # variational encoder to obtain posterior parameters
        self.q_theta = nn.Linear(self.num_hiddens, 2 * self.num_topics)

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul_(std).add_(mu)

    def kl_normal_normal(self, mu_theta, logsigma_theta):
        """Returns the Kullback-Leibler divergence between a normal distribution and a standard normal distribution.
        """
        kl_div = -0.5 * torch.sum(
            1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1
        )
        return kl_div.mean()

    def get_phi(self):
        """Derives the topic-word matrix by computing the inner product.
        """
        dist = self.alpha(self.rho.weight)
        phi = torch.softmax(dist, dim=0)
        return phi

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

        nelbo = recon_loss + kl_loss
        return nelbo, recon_loss, kl_loss, theta
