import os
import pdb
import argparse
import torch
import torch.nn as nn

from tqdm import tqdm
from os.path import join as pjoin
from torch.utils.data import DataLoader
from transformers import AutoModel

from wiki_loader import CrossSegWikiDataset

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
        return cls_loss


def train_collate_fn(batch):
    results = {}
    results['input_ids'] = torch.tensor([example[0] for example in batch])
    results['token_type_ids'] = torch.tensor([example[1] for example in batch])
    results['attention_mask'] = torch.tensor([example[2] for example in batch])
    results['targets'] = torch.tensor([example[3] for example in batch]).float()
    return results


def train(args):
    model = CrossSegmentBert(args)

    if args.cuda:
        model = model.to('cuda')

    train_set = CrossSegWikiDataset(args, 'train')
    dev_set = CrossSegWikiDataset(args, 'dev')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=train_collate_fn)
    optimizer = build_optim(args, model)

    for epoch in range(args.num_epochs):
        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                
            if args.cuda:
                batch = {k: v.to('cuda') for k, v in batch.items()}

            loss = model(**batch)
            loss.backward()

            if ((idx + 1) % args.grad_accum_steps == 0) or (idx + 1 == len(train_loader)):
                optimizer.step()
                model.zero_grad()

def test(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', help='Use cuda', action='store_true')
    parser.add_argument('-encoder', help='Pretrained encoder for the cross-segment attention model', default='bert-base-uncased', type=str)
    parser.add_argument('-mode', help='Train or test the model', default='train', type=str, choices=['train', 'test'])
    # parser.add_argument('-preprocess', help='Whether to preprocess the data to the format of the pretained encoder', action='store_true')
    parser.add_argument('-high_granularity', help='Use high granularity for wikipedia dataset segmentation', action='store_true')
    parser.add_argument('-data_dir', help='Path to directory containing the data files', default=pjoin('data', 'wiki_727'), type=str)
    parser.add_argument('-context_len', help='Token length of the left and right context (input_len = 2 x context_len + 2)', default=128, type=int)
    parser.add_argument('-pad_context', help='', action='store_true')

    # Effective batch size = batch_size * grad_accum_steps
    parser.add_argument('-batch_size', help='Batch size during training', type=int, default=8)
    parser.add_argument('-grad_accum_steps', help='Number of steps for gradient accumulation (Effective batch size = batch_size x grad_accum_steps)', type=int, default=192)
    parser.add_argument('-num_workers', help='Number of workers for the dataloaders', type=int, default=0)
    parser.add_argument('-num_epochs', help='Max number of epochs to train', type=int, default=1)

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