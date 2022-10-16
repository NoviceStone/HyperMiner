from __future__ import division
from __future__ import print_function

import datetime
import logging
import numpy as np
import os
import pickle
import time

import torch
from config import parser
from models.etm import ETM
from models.sawetm import SawETM
from models.hyperminer import HyperETM, HyperMiner, HyperMinerKG
from utils.data_util import get_data_loader
from utils.train_util import get_dir_name, convert_to_coo_adj, load_glove_embeddings, visualize_topics
from utils.eval_util import text_clustering


def main(args):
    global save_dir
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.cuda) >= 0 and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda:' + str(args.cuda))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    else:
        device = torch.device('cpu')

    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)

    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join('logs', args.dataset, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])
    logging.info('Clustering experiment')
    logging.info(f'Using {device}')
    logging.info(f'Using seed {args.seed}')

    train_loader, vocab = get_data_loader(args.dataset, args.data_path, 'train', args.batch_size)
    test_loader, _ = get_data_loader(args.dataset, args.data_path, 'test', args.batch_size, shuffle=False, drop_last=False)
    args.vocab_size = len(vocab)
    logging.info(f'Using dataset {args.dataset}')
    logging.info(f'{len(vocab)} words as vocabulary')
    logging.info(f'{len(train_loader.dataset)} training docs')
    logging.info(f'{len(test_loader.dataset)} test docs')

    if args.pretrained_embeddings:
        logging.info('Using pretrained glove embeddings')
        initial_embeddings = load_glove_embeddings(args.embed_size, vocab)
    else:
        initial_embeddings = None

    if args.add_knowledge:
        with open(args.file_path, 'rb') as f:
            adj, num_topics_list, concept_names = pickle.load(f)
        num_layers = len(num_topics_list)
        args.num_topics_list = num_topics_list
        args.num_hiddens_list = args.num_hiddens_list[: num_layers]

        sparse_adj = convert_to_coo_adj(adj)
        model = HyperMinerKG(args, device, initial_embeddings, sparse_adj)
    else:
        if args.manifold == 'Euclidean':
            model = SawETM(args, device, initial_embeddings)
            # model = ETM(args, device, initial_embeddings)
        else:
            model = HyperMiner(args, device, initial_embeddings)
            # model = HyperETM(args, device, initial_embeddings)
    model = model.to(device)
    logging.info(str(model))
    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters:{total_params}")

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    logging.info(f'Using {args.optimizer} optimizer')
    logging.info(f'Initial learning rate {args.lr}')

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs
    else:
        logging.info(f'Decay rate {args.gamma}')
        logging.info(f'Step size {args.lr_reduce_freq}')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )

    best_purity_epoch = 0
    best_purity = 0
    best_nmi_epoch = 0
    best_nmi = 0

    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        model.train()
        t = time.time()

        lowerbound = []
        likelihood = []
        kl_div = []
        for idx, (batch_data, _) in enumerate(train_loader):
            batch_data = batch_data.float().to(device)
            nelbo, nll, kl_loss, _ = model(batch_data)
            lowerbound.append(-nelbo.item())
            likelihood.append(-nll.item())
            kl_div.append(kl_loss.item())

            flag = 0
            for p in model.parameters():
                flag += torch.sum(torch.isnan(p))
            if flag == 0:
                optimizer.zero_grad()
                nelbo.backward()
                if args.grad_clip is not None:
                    max_norm = float(args.grad_clip)
                    all_params = list(model.parameters())
                    for param in all_params:
                        torch.nn.utils.clip_grad_norm_(param, max_norm)
                optimizer.step()

            if (idx + 1) % 10 == 0:
                print('Epoch: [{}/{}]\t elbo: {}\t likelihood: {}\t kl: {}'.format(
                    idx + 1, epoch + 1, np.mean(lowerbound), np.mean(likelihood), np.mean(kl_div)))

        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   '\tlr: {}'.format(lr_scheduler.get_last_lr()[0]),
                                   '\telbo: {:.8f}'.format(np.mean(lowerbound)),
                                   '\tlikelihood: {:.8f}'.format(np.mean(likelihood)),
                                   '\ttime: {:.4f}s'.format(time.time() - t)
                                   ]))

        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            test_feats = []
            test_labels = []
            for idx, (batch_data, batch_labels) in enumerate(test_loader):
                batch_data = batch_data.float().to(device)
                with torch.no_grad():
                    _, _, _, thetas = model(batch_data, is_training=False)
                    # test_feats.append(thetas.cpu().numpy())
                    test_feats.append(thetas[0].cpu().numpy())
                    test_labels.append(batch_labels.numpy())
            test_feats = np.concatenate(test_feats, axis=0)
            test_labels = np.concatenate(test_labels)

            purity, nmi = text_clustering(test_feats, test_labels)
            logging.info("Epoch: {:04d}\t Purity: {:.6f}\t NMI: {:.6f}".format(epoch + 1, purity, nmi))

            if purity > best_purity:
                best_purity = purity
                best_purity_epoch = epoch + 1
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, 'ckpt_best_purity.pth')
                )
            if nmi > best_nmi:
                best_nmi = nmi
                best_nmi_epoch = epoch + 1
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, 'ckpt_best_nmi.pth')
                )

        lr_scheduler.step()

    logging.info("Optimization Finished!")
    # logging.info("Best epoch: {}".format(best_epoch))
    logging.info("Best clustering purity: {:.6f} at epoch {}".format(best_purity, best_purity_epoch))
    logging.info("Best clustering nmi: {:.6f} at epoch {}".format(best_nmi, best_nmi_epoch))
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # model.load_state_dict(torch.load(
    #     os.path.join(save_dir, 'ckpt_best.pth'),
    #     map_location=device
    # ))
    # model.eval()
    # with torch.no_grad():
    #     phis = model.get_phi()
    #     visualize_topics(phis, save_dir, vocab)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
