import os
import argparse
import torch
import logging
from dataset import make_loaders
from model import GCLModel_Morph
from loss import spatial_contrastive_loss, reconstruction_loss
from metrics import compute_clustering_metrics
from sklearn.cluster import MiniBatchKMeans


def setup_logger(log_dir):
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    logger = logging.getLogger('train_root')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    return logger


def main(args):
    logger = setup_logger(args.save_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Device: %s' % device)

    full_dataset, train_loader, val_loader = make_loaders(args.data_dir, batch_size=args.batch_size)
    logger.info(f'Loaded dataset with {len(full_dataset)} samples')

    # detect in_dim
    sample = torch.load(full_dataset.file_list[0])
    in_dim = sample.x.shape[1]
    model = GCLModel_Morph(in_channels=in_dim, hidden_channels=args.hidden_dim, out_channels=args.out_dim, n_clusters=args.n_clusters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for b_orig, b_corr in train_loader:
            b_orig, b_corr = b_orig.to(device), b_corr.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                z1, _, rec1, _ = model(b_orig.x, b_orig.edge_index, b_orig.edge_attr)
                z2, _, _, _ = model(b_corr.x, b_corr.edge_index, b_corr.edge_attr)
                loss_cl = spatial_contrastive_loss(z1, z2, b_orig.edge_index, b_orig.x, b_orig.batch if hasattr(b_orig,'batch') else torch.zeros(b_orig.x.size(0), dtype=torch.long, device=b_orig.x.device))
                loss_rec = reconstruction_loss(rec1, b_orig.x)
                loss = loss_cl + args.lambda_rec * loss_rec
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item()

        avg = total_loss / len(train_loader)
        logger.info(f'Epoch {epoch} train loss {avg:.4f}')

        # save checkpoint
        if epoch % args.save_every == 0:
            torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch}, os.path.join(args.save_dir, f'ckpt_{epoch}.pth'))

    # final save
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model.pth'))
    logger.info('Training complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='DATA_DIRECTORY/kz_data/Graph_pt')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=32)
    parser.add_argument('--n_clusters', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_rec', type=float, default=1.0)
    parser.add_argument('--save_every', type=int, default=5)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
