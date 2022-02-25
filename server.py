import os
import csv
import math
import json
import argparse
import numpy as np
from os.path import join as pjoin


#### only needed for cross-origin requests:
# from flask_cors import CORS
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.manifold import TSNE
from run_cross_segment import CrossSegmentBert
from wiki_loader import CrossSegWiki727KDataset, CrossSegWikiSectionDataset
from flask import Flask, send_from_directory, request, Response, redirect, url_for, jsonify

import pdb



app = Flask(__name__)
#### only needed for cross-origin requests:
# CORS(app)

# Variables
loaded_model = None
loaded_dataset = None
loaded_tokenizer = None


def load_globals(args):
    """
    Initialize global variables to be used by later function calls
    """
    global loaded_dataset, loaded_model, loaded_tokenizer

    model = CrossSegmentBert(args)
    model = model.to(args.device)
    model.load_state_dict(torch.load(args.load_checkpoint, map_location=args.device), strict=False)

    loaded_model = model
    loaded_tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    # Initialize datasets
    if args.dataset == 'wiki_727K':
        loaded_dataset = CrossSegWiki727KDataset(args, 'dev')
    elif args.dataset == 'wiki_section':
        loaded_dataset = CrossSegWikiSectionDataset(args, 'validation')



####################################### Helper Functions #######################################
def collate_fn(batch):
    results = {}
    results['input_ids'] = torch.tensor([example[0] for example in batch])
    results['token_type_ids'] = torch.tensor([example[1] for example in batch])
    results['attention_mask'] = torch.tensor([example[2] for example in batch])
    results['targets'] = torch.tensor([example[3] for example in batch]).float()
    return results

def pairwise_cosine_similarity(a, b, eps=1e-8):
    """
    Compute pairwise cosine similarity between two sets of vectors
    (Added eps for numerical stability)
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def normalize(matrix, axis=None):
    """
    Normalize NumPy matrix, across all dimensions by default
    """
    normalized = (matrix - matrix.min(axis=axis)) /\
                 (matrix.max(axis=axis) - matrix.min(axis=axis))
    return normalized

def save_hidden_state_projection(model, dataloader, projection_file, n_projections=5000):
    total_examples = len(dataloader.dataset)
    hidden_states = torch.zeros(total_examples, 768)
    confidence = np.zeros((total_examples, 1))
    tsne_model = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=500, metric='precomputed', random_state=0)

    model.eval()
    device = next(model.parameters()).device
    sample_idx = 0
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

        batch_size = len(batch['input_ids'])
        batch = {k: v.to(device) for k, v in batch.items()}
        batch['output_hidden'] = True

        with torch.no_grad():   
            output = model(**batch)
            cls_hidden_states = output['hidden']
            predictions = torch.sigmoid(output['logits']).to('cpu').numpy()


        for idx in range(batch_size):
            hidden_states[sample_idx] = cls_hidden_states[idx].cpu()

            label = int(batch['targets'][idx].item())
            confidence[sample_idx][0] = predictions[idx] if (label == 1) else (1 - predictions[idx])

            sample_idx += 1

    # Create TSNE projections using cosine similarity
    dist = pairwise_cosine_similarity(hidden_states, hidden_states).numpy()
    dist = (dist - dist.min()) / (dist.max() - dist.min())
    tsne_vectors = tsne_model.fit_transform(dist).round(decimals=5)

    # Save formatted output to projection file
    header = ','.join(['ID', 'tsne_1', 'tsne_2', 'confidence'])
    example_ids = np.expand_dims(np.arange(n_projections), axis=1).astype(int)
    save_data = np.hstack((example_ids, tsne_vectors[:n_projections], confidence[:n_projections]))
    np.savetxt(projection_file, save_data, delimiter=',', header=header, comments='', fmt='%s')
    print(f"Successfully saved projection data at \"{projection_file}\"")

    return save_data

def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    """
    List, int, int, set -> Tuple[set, "torch.LongTensor"]
    """

    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0

    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[~mask].long()
    return heads, index

def estimate_importance(model, dataloader, measures=['taylor']):

    assert set(measures).issubset(set(['oracle', 'taylor', 'sensitivity', 'lrp']))

    encoder = model.bert if hasattr(model, 'bert') else model.encoder

    n_layers = encoder.config.num_hidden_layers
    n_heads = encoder.config.num_attention_heads
    head_size = int(encoder.config.hidden_size / n_heads)

    importance_scores = {}
    for measure in measures:
        importance_scores[measure] = np.zeros((n_layers, n_heads))

    device = next(model.parameters()).device
    model.train()

    total_loss = 0.

    if 'taylor' in measures or 'sensitivity' in measures or 'lrp' in measures:

        for i, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            logits, loss = output['logits'], output['loss']
            loss.backward(retain_graph=True)

            total_loss += loss.item()

            if 'taylor' in measures :
                for i in range(n_layers):
                    attention = encoder.encoder.layer[i].attention
                    num_attention_heads = attention.self.num_attention_heads

                    pruned_heads = attention.pruned_heads
                    leftover_heads = set(list(range(n_heads))) - pruned_heads

                    for head_idx in leftover_heads:
                        heads, index = find_pruneable_heads_and_indices([head_idx], num_attention_heads, head_size, pruned_heads)
                        index = index.to(device)

                        query_b_grad = (attention.self.query.bias.grad[index] *\
                                        attention.self.query.bias[index]) ** 2
                        query_W_grad = (attention.self.query.weight.grad.index_select(0, index) *\
                                        attention.self.query.weight.index_select(0, index)) ** 2

                        key_b_grad = (attention.self.key.bias.grad[index] *\
                                      attention.self.key.bias[index]) ** 2
                        key_W_grad = (attention.self.key.weight.grad.index_select(0, index) *\
                                      attention.self.key.weight.index_select(0, index)) ** 2

                        value_b_grad = (attention.self.value.bias.grad[index] *\
                                        attention.self.value.bias[index]) ** 2
                        value_W_grad = (attention.self.value.weight.grad.index_select(0, index) *\
                                        attention.self.value.weight.index_select(0, index)) ** 2

                        abs_grad_magnitude = query_b_grad.sum() + query_W_grad.sum() + key_b_grad.sum() + \
                            key_W_grad.sum() + value_b_grad.sum() + value_W_grad.sum() 

                        score = abs_grad_magnitude.item()
                        importance_scores['taylor'][i, head_idx] += score

    return importance_scores

################################################################################################



# redirect requests from root to index.html
@app.route('/')
def index():
    return redirect('client/index.html')


@app.route('/api/projections', methods=['POST'])
def encoder_embedding():
    """
    Load files containing processed data for hidden states projection and 
    attention head importance scores before returning the data to front-end  
    """
    projection_file = pjoin('client', 'resources', 'projections', 'bert_cross_seg_projections.csv')
    head_importance_file = pjoin('client', 'resources', 'head_importance', 'bert_cross_seg_importance.pth')
    dataloader = DataLoader(loaded_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)

    if not os.path.isfile(projection_file):
        save_hidden_state_projection(loaded_model, dataloader, projection_file)

    with open(projection_file, 'r') as f:
        csv_reader = csv.DictReader(f)
        projection_data = [row for row in csv_reader]
    
    if not os.path.isfile(head_importance_file):
        importance_scores = estimate_importance(loaded_model, dataloader, measures=['taylor'])
        torch.save(importance_scores, head_importance_file)
    else:
        importance_scores = torch.load(head_importance_file)
    importance = normalize(np.array(importance_scores['taylor'])).tolist()

    results = {}
    results['projection'] = projection_data
    results['importance'] = importance
    return jsonify(results)


@app.route('/api/attn', methods=['POST'])
def get_attention_map():
    example = loaded_dataset[int(request.json['ID'])]
    device = next(loaded_model.parameters()).device

    tokens = loaded_tokenizer.convert_ids_to_tokens(example[0])

    model_input = {
        'input_ids': torch.tensor([example[0]]).to(device),
        'token_type_ids': torch.tensor([example[1]]).to(device),
        'attention_mask': torch.tensor([example[2]]).to(device),
        'targets': torch.tensor([example[3]]).float().to(device),
        'output_attentions': True,
    }

    output = loaded_model(**model_input)

    attentions = []
    attn = (torch.cat([l for l in output['attentions']], dim=0) * 100).round().byte().to('cpu')
    for layer in range(attn.size(0)):
        attentions.append([])
        for head in range(attn.size(1)):
            attentions[layer].append(attn[layer][head].tolist())

    results = {}
    results['tokens'] = tokens
    results['attentions'] = attentions
    return jsonify(results)


# just a simple example for GET request
@app.route('/api/data/')
def get_data():
    options = request.args
    name = str(options.get("name", ''))
    y = int(options.get("y", 0))

    res = {
        'name': name,
        'y': [10 * y]
    }
    json_res = json.dumps(res)
    return Response(json_res, mimetype='application/json')


# send everything from client as static content
@app.route('/client/<path:path>')
def send_static(path):
    """ serves all files from ./client/ to ``/client/<path:path>``
    :param path: path from api call
    """
    return send_from_directory('client/', path)


if __name__ == '__main__':
    CWD = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-port', type=int, default='8888')
    parser.add_argument('-host', default=None)
    parser.add_argument('-device', help='GPU index', default=0, type=int)

    parser.add_argument('-encoder', help='Pretrained encoder for the cross-segment attention model', default='bert-base-uncased', type=str)
    parser.add_argument('-hidden_size', help='Hidden size of the output classifier', type=int, default=128)

    parser.add_argument('-high_granularity', help='Use high granularity for wikipedia dataset segmentation', action='store_true')
    parser.add_argument('-dataset', help='Name of the dataset', default='wiki_section', type=str, choices=['wiki_727K', 'wiki_section'])
    parser.add_argument('-context_len', help='Token length of the left and right context (input_len = 2 x context_len + 2)', default=128, type=int)
    parser.add_argument('-pad_context', help='Pad left and right context with [PAD]', action='store_true', default=True) # Based on implementation described in paper

    parser.add_argument('-data_dir', help='Path to directory containing the data files', default=pjoin('data', 'wiki_section'), type=str)
    parser.add_argument("-resource_dir", help='Resource directory used by the client (front-end)', default=pjoin('client', 'resources'), type=str)
    parser.add_argument("-load_checkpoint", default=pjoin('results', 'bert-base-uncased_wiki_section_350_64.pth'), type=str)

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.device}')
    else:
        args.device = torch.device('cpu')

    load_globals(args)
    app.run(host=args.host, port=int(args.port), debug=args.debug)