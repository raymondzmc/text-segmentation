import os
import pdb
import math
import logging
import argparse
from os.path import join as pjoin

import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup

from wiki_loader import CrossSegWiki727KDataset, CrossSegWikiSectionDataset


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger

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
        if args.encoder == 'roberta-base':
            self.use_token_type_ids = False
        else:
            self.use_token_type_ids = True
    
    def init_classifier(self):
        for p in self.classifier.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.constant_(p, 0)

    def forward(self, input_ids, attention_mask, token_type_ids, targets, output_hidden=False, output_attentions=False):
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if self.use_token_type_ids else None,
            output_attentions=output_attentions,
        )

        hidden_states = encoder_output[0]
        cls_embeddings = hidden_states[:, 0, :]

        logits = self.classifier(cls_embeddings).squeeze(1)
        cls_loss = self.criterion(logits, targets)

        output = {
            'logits': logits,
            'loss': cls_loss,
        }

        if output_hidden:
            output['hidden'] = cls_embeddings
        if output_attentions:
            output['attentions'] = encoder_output[-1]

        return output


def collate_fn(batch):
    results = {}
    results['input_ids'] = torch.tensor([example[0] for example in batch])
    results['token_type_ids'] = torch.tensor([example[1] for example in batch])
    results['attention_mask'] = torch.tensor([example[2] for example in batch])
    results['targets'] = torch.tensor([example[3] for example in batch]).float()
    return results

def eval(model, eval_loader):
    model.eval()
    device = next(model.parameters()).device

    tp, fp, fn = 0, 0, 0
    total_loss = 0.
    for step, batch in tqdm(enumerate(eval_loader), total=len(eval_loader)):

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            output = model(**batch)

        logits, loss = output['logits'], output['loss']

        targets = batch['targets']
        pred = torch.sigmoid(logits).round()
        tp += torch.logical_and(targets == 1, pred == 1).sum().item()
        fp += torch.logical_and(targets == 0, pred == 1).sum().item()
        fn += torch.logical_and(targets == 1, pred == 0).sum().item()
        total_loss += loss.item()

    try:
        precision = round(tp / (tp + fp), 4)
    except:
        precision = 0.

    try:
        recall = round(tp / (tp + fn), 4)
    except:
        recall = 0.

    f_score = round(tp / (tp + 0.5 * (fp + fn)), 4)
    avg_loss = round(total_loss / len(eval_loader), 4)

    return precision, recall, f_score, avg_loss


class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []
        self.sum = 0

    def process(self, value):
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        return float(self.sum) / len(self.values)


def train(args):
    model = CrossSegmentBert(args)
    model.train()
    model = model.to(args.device)

    # Initialize datasets
    if args.dataset == 'wiki_727K':
        train_set = CrossSegWiki727KDataset(args, 'train')
        dev_set = CrossSegWiki727KDataset(args, 'dev')
    elif args.dataset == 'wiki_section':
        train_set = CrossSegWikiSectionDataset(args, 'train')
        dev_set = CrossSegWikiSectionDataset(args, 'validation')
        test_set = CrossSegWikiSectionDataset(args, 'test')

    # Data loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    dev_loader = DataLoader(dev_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    # Optimization
    optimizer = AdamW(model.parameters(), lr=args.lr)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    num_training_steps = (steps_per_epoch * args.num_epochs) if args.num_training_steps == None else args.num_training_steps
    num_warmup_steps = num_training_steps // 10 if args.num_training_steps == None else args.num_training_steps
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    total_accum_steps = 0

    # Tracking moving average (loss)
    window_size = args.grad_accum_steps * args.report_steps
    moving_loss = MovingAverage(window_size=100)
    best_f_score = 0.


    for epoch in range(args.num_epochs):
        epoch_loss = 0.

        with tqdm(desc='Training', total=len(train_loader)) as pbar:
            for step, batch in enumerate(train_loader):
                    
                batch = {k: v.to(args.device) for k, v in batch.items()}

                output = model(**batch)
                loss = output['loss']
                epoch_loss += loss.item()
                avg_loss = moving_loss.process(loss.item())
                loss.backward()

                if ((step + 1) % args.grad_accum_steps == 0) or (step + 1 == len(train_loader)):
                    optimizer.step()
                    lr_scheduler.step()
                    model.zero_grad()
                    total_accum_steps += 1

                    if total_accum_steps % args.report_steps == 0:
                        lr = optimizer.param_groups[0]['lr']
                        args.logger.info(f'(Step {total_accum_steps}) LR: {lr}, Moving Loss: {avg_loss}')

                    if total_accum_steps % args.eval_steps == 0:
                        pbar.set_description('Evaluating on Dev Set')
                        precision, recall, f_score, total_loss = eval(model, dev_loader)
                        dev_results = f'(Dev) F1 Score: {f_score}, Loss: {total_loss}'
                        pbar.set_description(dev_results)
                        args.logger.info(dev_results)
                        model.train()

                        if f_score > best_f_score:
                            save_name = f'{args.encoder}_{args.dataset}_{total_accum_steps}_{round(f_score * 100)}.pth'
                            torch.save(model.state_dict(), pjoin(args.results_dir, save_name))
                            best_f_score = f_score
                            args.logger.info(f"Saved Checkpoint at \"{save_name}\"")

                pbar.update(1)


def test(args):
    model = CrossSegmentBert(args)
    model.eval()
    model = model.to(args.device)

    checkpoints = [f for f in os.listdir(args.results_dir) if f.split('.')[-1] in ['pt', 'pth']]

    # Initialize datasets
    if args.dataset == 'wiki_727K':
        test_set = CrossSegWiki727KDataset(args, 'test')
    elif args.dataset == 'wiki_section':
        test_set = CrossSegWikiSectionDataset(args, 'test')

    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    for ckpt in checkpoints:
        args.logger.info(f"\nResults for Checkpoint Loaded at \"{ckpt}\":")

        ckpt_path = pjoin(args.results_dir, ckpt)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))

        precision, recall, f_score, total_loss = eval(model, test_loader)
        test_results = f'(Test) F1 Score: {f_score}, Precision: {precision}, Recall: {recall} Loss: {total_loss}'
        args.logger.info(test_results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', help='GPU index', default=0)
    parser.add_argument('-encoder', help='Pretrained encoder for the cross-segment attention model', default='bert-base-uncased', type=str)
    parser.add_argument('-mode', help='Train or test the model', default='train', type=str, choices=['train', 'test'])
    # parser.add_argument('-preprocess', help='Whether to preprocess the data to the format of the pretained encoder', action='store_true')
    parser.add_argument('-high_granularity', help='Use high granularity for wikipedia dataset segmentation', action='store_true')

    parser.add_argument('-dataset', help='Name of the dataset', default='wiki_section', type=str, choices=['wiki_727K', 'wiki_section'])
    parser.add_argument('-data_dir', help='Path to directory containing the data files', default=pjoin('data', 'wiki_section'), type=str)
    parser.add_argument('-context_len', help='Token length of the left and right context (input_len = 2 x context_len + 2)', default=128, type=int)
    parser.add_argument('-pad_context', help='Pad left and right context with [PAD]', action='store_true', default=True) # Based on implementation described in paper

    # Effective batch size = batch_size * grad_accum_steps (e.g. 8 x 8 = 64)
    parser.add_argument('-batch_size', help='Batch size during training', type=int, default=8)
    parser.add_argument('-grad_accum_steps', help='Number of steps for gradient accumulation (Effective batch size = batch_size x grad_accum_steps)', type=int, default=8)
    parser.add_argument('-num_warmup_steps', help='Number of warm-up steps (for lr scheduler)', type=int, default=None)
    parser.add_argument('-num_training_steps', help='Total number of training steps (for lr scheduler)', type=int, default=None)
    parser.add_argument('-num_epochs', help='Max number of epochs to train', type=int, default=4)
    parser.add_argument('-lr', help='Learning rate', type=float, default=1e-5)

    parser.add_argument('-num_workers', help='Number of workers for the dataloaders', type=int, default=0)
    parser.add_argument('-eval_steps', help='Number of accumulated steps before each dev set evaluation', type=int, default=50)
    parser.add_argument('-report_steps', help='Number of accumulated steps before printing the training loss', type=int, default=1)
    parser.add_argument('-eval_batch_size', help='Batch size during evaluation', type=int, default=16)

    parser.add_argument('-hidden_size', help='Hidden size of the output classifier', type=int, default=128)
    parser.add_argument('-results_dir', help='Directory for storing the results', type=str, default='results')

    args = parser.parse_args()

    if torch.cuda.is_available():
        if type(args.device) == int:
            args.device = torch.device(f'cuda:{args.device}')
        else:
            args.device = torch.device(args.device)
    else:
        args.device = torch.device('cpu')

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)

    date = datetime.now().strftime("%Y%m%d")
    log_file = f"{args.encoder}_{args.dataset}_{date}.log"
    args.logger = init_logger(pjoin(args.results_dir, log_file))
    args.logger.info(args)

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