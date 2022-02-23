import os
import pdb
import argparse
import torch
import torch.nn as nn

from tqdm import tqdm
from os.path import join as pjoin
from torch.utils.data import DataLoader
from transformers import AutoModel, AdamW

from wiki_loader import CrossSegWiki727KDataset, CrossSegWikiSectionDataset

class CrossSegmentBert(nn.Module):
    def __init__(self, args):
        super(CrossSegmentBert, self).__init__()
        self.encoder = AutoModel.from_pretrained(args.encoder)

        classifier_input_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, args.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1, bias=True),
        )
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.init_classifier()
    
    def init_classifier(self):
        for p in self.classifier.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.constant_(p, 0)

    def forward(self, input_ids, attention_mask, token_type_ids, targets):
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        hidden_states = encoder_output[0]
        cls_embeddings = hidden_states[:, 0, :]

        logits = self.classifier(cls_embeddings).squeeze(1)
        cls_loss = self.criterion(logits, targets)
        return (logits, cls_loss)


def train_collate_fn(batch):
    results = {}
    results['input_ids'] = torch.tensor([example[0] for example in batch])
    results['token_type_ids'] = torch.tensor([example[1] for example in batch])
    results['attention_mask'] = torch.tensor([example[2] for example in batch])
    results['targets'] = torch.tensor([example[3] for example in batch]).float()
    return results

def eval(model, eval_loader):
    model.eval()
    cuda = next(model.parameters()).is_cuda

    with torch.no_grad():

        tp, fp, fn = 0, 0, 0
        total_loss = 0.
        for step, batch in tqdm(enumerate(eval_loader), total=len(eval_loader)):
            if cuda:
                batch = {k: v.to('cuda') for k, v in batch.items()}

            
            logits, loss = model(**batch)

            targets = batch['targets']
            pred = logits.round()
            tp += torch.logical_and(targets == 1, pred == 1).sum().item()
            fp += torch.logical_and(targets == 0, pred == 1).sum().item()
            fn += torch.logical_and(targets == 1, pred == 0).sum().item()
            total_loss += loss.item()

            if step > 100:
                break

        precision = round(tp / (tp + fp), 4)
        recall = round(tp / (tp + fn), 4)
        f_score = round(tp / (tp + 0.5 * (fp + fn)), 4)
        total_loss = round(total_loss, 4)


    return precision, recall, f_score, total_loss



def train(args):
    model = CrossSegmentBert(args)
    model.train()

    if args.cuda:
        model = model.to('cuda')

    # Initialize datasets
    if args.dataset == 'wiki_727K':
        train_set = CrossSegWiki727KDataset(args, 'train')
        dev_set = CrossSegWiki727KDataset(args, 'dev')
    elif args.dataset == 'wiki_section':
        train_set = CrossSegWikiSectionDataset(args, 'train')
        dev_set = CrossSegWikiSectionDataset(args, 'validation')
        test_set = CrossSegWikiSectionDataset(args, 'test')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=train_collate_fn, num_workers=args.num_workers)
    dev_loader = DataLoader(dev_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=train_collate_fn, num_workers=args.num_workers)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    total_accum_steps = 0
    for epoch in range(args.num_epochs):
        with tqdm(desc='Training', total=len(train_loader)) as pbar:
            for step, batch in enumerate(train_loader):
                    
                if args.cuda:
                    batch = {k: v.to('cuda') for k, v in batch.items()}

                logits, loss = model(**batch)
                loss.backward()

                if ((step + 1) % args.grad_accum_steps == 0) or (step + 1 == len(train_loader)):
                    optimizer.step()
                    model.zero_grad()
                    total_accum_steps += 1

                if (total_accum_steps + 1) % args.eval_steps == 0:
                    pbar.set_description('Evaluating on Dev Set')
                    precision, recall, f_score, total_loss = eval(model, dev_loader)
                    pbar.set_description(f'(Dev: f1: {f_score}, loss={total_loss}')
                    model.train()

                pbar.update(1)


def test(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', help='Use cuda', action='store_true')
    parser.add_argument('-encoder', help='Pretrained encoder for the cross-segment attention model', default='bert-base-uncased', type=str)
    parser.add_argument('-mode', help='Train or test the model', default='train', type=str, choices=['train', 'test'])
    # parser.add_argument('-preprocess', help='Whether to preprocess the data to the format of the pretained encoder', action='store_true')
    parser.add_argument('-high_granularity', help='Use high granularity for wikipedia dataset segmentation', action='store_true')

    parser.add_argument('-dataset', help='Name of the dataset', default='wiki_section', type=str, choices=['wiki_727K', 'wiki_section'])
    parser.add_argument('-data_dir', help='Path to directory containing the data files', default=pjoin('data', 'wiki_section'), type=str)
    parser.add_argument('-context_len', help='Token length of the left and right context (input_len = 2 x context_len + 2)', default=128, type=int)
    parser.add_argument('-pad_context', help='Pad left and right context with [PAD]', action='store_true')

    # Effective batch size = batch_size * grad_accum_steps
    parser.add_argument('-batch_size', help='Batch size during training', type=int, default=8)
    parser.add_argument('-grad_accum_steps', help='Number of steps for gradient accumulation (Effective batch size = batch_size x grad_accum_steps)', type=int, default=192)
    parser.add_argument('-num_workers', help='Number of workers for the dataloaders', type=int, default=0)
    parser.add_argument('-num_epochs', help='Max number of epochs to train', type=int, default=1)
    parser.add_argument('-lr', help='Learning rate', type=float, default=1e-5)
    parser.add_argument('-eval_steps', help='Number of accumulated steps before each dev set evaluation', type=int, default=1000)
    parser.add_argument('-eval_batch_size', help='Batch size during evaluation', type=int, default=16)

    parser.add_argument('-hidden_size', help='Hidden size of the output classifier', type=int, default=128)
    

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        test(args)

    # train_set = CrossSegWikiDataset(args, 'train')
    # dev_set = CrossSegWikiDataset(args, 'dev')
    # test_set = CrossSegWikiDataset(args, 'test')

    # for i, data in tqdm(enumerate(train_set), total=len(train_set)):
    #     assert (len(data[1]) == args.context_len and len(data[0]) == args.context_len)
    #     continue