import re
import os
import torch
import pdb
import itertools

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from nltk.tokenize import RegexpTokenizer
from pathlib2 import Path
from tqdm import tqdm
from os.path import join as pjoin

import utils


logger = utils.setup_logger(__name__, 'train.log')

section_delimiter = "========"

def get_seperator_foramt(levels = None):
    segment_seperator = "========"
    level_format = '\d' if levels == None else '['+ str(levels[0]) + '-' + str(levels[1]) + ']'
    seperator_fromat = segment_seperator + ',' + level_format + ",.*?\."
    return seperator_fromat


def get_list_token():
    return "***LIST***"

def get_formula_token():
    return "***formula***"

def get_codesnipet_token():
    return "***codice***"

def get_special_tokens():
    special_tokens = []
    special_tokens.append(get_list_token())
    special_tokens.append(get_formula_token())
    special_tokens.append(get_codesnipet_token())
    return special_tokens

def get_words_tokenizer():
    global words_tokenizer

    if words_tokenizer:
        return words_tokenizer

    words_tokenizer = RegexpTokenizer(r'\w+')
    return words_tokenizer


def extract_sentence_words(sentence, remove_missing_emb_words = False,remove_special_tokens = False):
    if (remove_special_tokens):
        for token in get_special_tokens():
            # Can't do on sentence words because tokenizer delete '***' of tokens.
            sentence = sentence.replace(token, "")
    tokenizer = get_words_tokenizer()
    sentence_words = tokenizer.tokenize(sentence)
    if remove_missing_emb_words:
        sentence_words = [w for w in sentence_words if w not in missing_stop_words]

    return sentence_words

def word_model(word, model):
    if model is None:
        return np.random.randn(1, 300)
    else:
        if word in model:
            return model[word].reshape(1, 300)
        else:
            #print ('Word missing w2v: ' + word)
            return model['UNK'].reshape(1, 300)



def get_files(path):
    all_objects = Path(path).glob('**/*')
    files = [str(p) for p in all_objects if p.is_file()]
    return files


def get_cache_path(wiki_folder):
    cache_file_path = wiki_folder / 'paths_cache'
    return cache_file_path


def cache_wiki_filenames(wiki_folder):
    files = Path(wiki_folder).glob('*/*/*/*')
    cache_file_path = get_cache_path(wiki_folder)
    with cache_file_path.open('w+') as f:
        for file in files:
            f.write(str(file) + '\n')


def clean_section(section):
    cleaned_section = section.strip('\n')
    return cleaned_section


def get_scections_from_text(txt, high_granularity=True):
    sections_to_keep_pattern = get_seperator_foramt() if high_granularity else get_seperator_foramt(
        (1, 2))
    if not high_granularity:

        # if low granularity required we should flatten segments within segemnt level 2
        pattern_to_ommit = get_seperator_foramt((3, 999))
        txt = re.sub(pattern_to_ommit, "", txt)

        #delete empty lines after re.sub()
        sentences = [s for s in txt.strip().split("\n") if len(s) > 0 and s != "\n"]
        txt = '\n'.join(sentences).strip('\n')


    all_sections = re.split(sections_to_keep_pattern, txt)
    non_empty_sections = [s for s in all_sections if len(s) > 0]

    return non_empty_sections


def get_sections(path, high_granularity=True):
    file = open(str(path), "r")
    raw_content = file.read()
    file.close()

    clean_txt = raw_content.strip()

    sections = [clean_section(s) for s in get_scections_from_text(clean_txt, high_granularity)]

    return sections


def read_wiki_file(path, word2vec, remove_preface_segment=True, ignore_list=False, remove_special_tokens=False,
                   return_as_sentences=True, high_granularity=True, only_letters = False):
    data = []
    targets = []

    # Get different sections based on patterns specified by data
    all_sections = get_sections(path, high_granularity)

    # Remove preface and empty sections
    required_sections = all_sections[1:] if remove_preface_segment and len(all_sections) > 0 else all_sections
    required_non_empty_sections = [section for section in required_sections if len(section) > 0 and section != "\n"]


    for section in required_non_empty_sections:
        sentences = section.split('\n')
        if sentences:
            for sentence in sentences:
                is_list_sentence = get_list_token() + "." == sentence
                if ignore_list and is_list_sentence:
                    continue
                if not return_as_sentences:
                    sentence_words = extract_sentence_words(sentence, remove_special_tokens=remove_special_tokens)
                    if 1 <= len(sentence_words):
                        data.append([word_model(word, word2vec) for word in sentence_words])
                    else:
                        #raise ValueError('Sentence in wikipedia file is empty')
                        logger.info('Sentence in wikipedia file is empty')
                else:  # for the annotation. keep sentence as is.
                    if (only_letters):
                        sentence = re.sub('[^a-zA-Z0-9 ]+', '', sentence)
                        data.append(sentence)
                    else:
                        data.append(sentence)
            if data:
                # Index of the last sentence in the current section
                targets.append(len(data) - 1)
    return data, targets, path


class WikipediaDataSet(Dataset):
    def __init__(self, root, word2vec, train=True, manifesto=False, folder=False, high_granularity=False):

        if (manifesto):
            self.textfiles = list(Path(root).glob('*'))
        else:
            if (folder):
                self.textfiles = get_files(root)
            else:
                root_path = Path(root)
                cache_path = get_cache_path(root_path)
                if not cache_path.exists():
                    cache_wiki_filenames(root_path)

                # Stores the path to all files in the dataset
                self.textfiles = cache_path.read_text().splitlines()

        if len(self.textfiles) == 0:
            raise RuntimeError('Found 0 images in subfolders of: {}'.format(root))
        self.train = train
        self.root = root
        self.word2vec = word2vec
        self.high_granularity = high_granularity

    def __getitem__(self, index):
        path = self.textfiles[index]

        return read_wiki_file(Path(path), self.word2vec, ignore_list=True, remove_special_tokens=True,
                              high_granularity=self.high_granularity)

    def __len__(self):
        return len(self.textfiles)


class CrossSegWikiDataset(Dataset):
    """
    Wikipedia topic segmentation dataset formatted for the cross-segment attention model
    """
    def __init__(self, args, split_name):
        assert split_name in os.listdir(args.data_dir)
        self.data_dir = pjoin(args.data_dir, split_name)

        # Load cache file containing absolute path to each data file
        cached_file_path = get_cache_path(Path(self.data_dir))
        if not cached_file_path.exists():
            cache_wiki_filenames(self.data_dir)
            print(f"Created cache file containing paths to all data files at: \"{cache_file_path}\"")

         # Stores the path to all files in the dataset
        self.data_files = cached_file_path.read_text().splitlines()
        self.high_granularity = args.high_granularity
        self.context_len = args.context_len
        self.pad_context = args.pad_context
        self.tokenizer = AutoTokenizer.from_pretrained(args.encoder)
        # self.wiki_dataset = WikipediaDataSet(self.data_dir, word2vec=None, train=False, high_granularity=self.high_granularity)

        # data_splits = os.listdir(args.data_dir)

        # for dir_name in data_splits:
        self.file_suffix = 'preprocessed'
        cached_data_mapping = pjoin(self.data_dir, f"cached_data_mapping")
        if not os.path.isfile(cached_data_mapping):
            data_mapping = self.preprocess_for_encoder()
            torch.save(data_mapping, cached_data_mapping)

        else:
            data_mapping = torch.load(cached_data_mapping)
            self.index2data = data_mapping['index2data']
            self.num_boundaries = data_mapping['total']

    def preprocess_for_encoder(self):

        # Number of sentence boundaries
        self.num_boundaries = 0
        
        # Map index to the corresponding sentence boundary in the data file
        self.index2data = {}
        for file_idx, file_path in tqdm(enumerate(self.data_files), total=len(self.data_files)):
            (data, seg_idx, path) = read_wiki_file(Path(file_path), None, ignore_list=True, remove_special_tokens=True, high_granularity=self.high_granularity)
            if len(data) < 2:
                continue

            save_path = f"{str(path)}_{self.file_suffix}"
            input_ids = self.tokenizer(data, add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False)['input_ids']

            sent_boundaries = list(itertools.accumulate([len(sent) for sent in input_ids]))
            input_ids = [token_id for sent in input_ids for token_id in sent]
            targets = [0 for _ in range(len(sent_boundaries))]
            for idx in seg_idx:
                targets[idx] = 1

            # targets = [0 for idx in range(len(sent_boundaries)) if idx in seg_idx else 0]

            torch.save({'input_ids': input_ids, 'sent_boundaries': sent_boundaries, 'targets': targets}, save_path)

            for boundary_idx in range(len(data) - 1):
                dataset_idx = self.num_boundaries + boundary_idx
                self.index2data[dataset_idx] = (file_idx, boundary_idx)
            
            self.num_boundaries += len(data) - 1

        cached_data_mapping = {
            'index2data': self.index2data,
            'total': self.num_boundaries,
        }

        return cached_data_mapping

    def __len__(self):
        return self.num_boundaries

    def __getitem__(self, index):
        file_idx, boundary_idx = self.index2data[index]

        file_path = f"{self.data_files[file_idx]}_{self.file_suffix}"

        data = torch.load(file_path)
        token_idx = data['sent_boundaries'][boundary_idx]

        left_context = data['input_ids'][max(0, token_idx - self.context_len):token_idx]
        right_context = data['input_ids'][token_idx:token_idx + self.context_len]

        # Pad to the left or right of context
        if self.pad_context:
            left_context = [self.tokenizer.pad_token_id for _ in range(self.context_len - len(left_context))] + left_context
            right_context = right_context + [self.tokenizer.pad_token_id for _ in range(self.context_len - len(right_context))]

        input_ids = [self.tokenizer.cls_token_id] + left_context + [self.tokenizer.sep_token_id] + right_context
        token_type_ids = [0 for _ in range(len(left_context) + 2)] + [1 for _ in range(len(right_context))]
        attention_mask = [0 if x == 0 else 1 for x in input_ids]
        return (input_ids, token_type_ids, attention_mask, data['targets'][boundary_idx]) 



    


