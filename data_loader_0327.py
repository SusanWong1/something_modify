import gc
import glob
import random
import argparse
import pickle
# import keras

import numpy as np
from sklearn.externals import joblib
import torch

from pytorch_pretrained_bert import BertTokenizer, BertModel


# from others.logging import logger

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data if len(d) > 0)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data if len(d) > 0]
        return rtn_data

    def __init__(self, data=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        self.vocab_size = 30522
        self.negnum = 32
        self.device = device
        self.words = 0
        self.raw_data = data
        if data is not None and len(data) > 0:
            self.real = True
            self.batch_size = len(data)
            for x in data:
                self.words += len(x[0])
            
            pre_src = [x[0] for x in data]
            pre_labels = [x[1] for x in data]
            self.labels = pre_labels
            pre_segs = [x[2] for x in data]
            pre_clss = [x[3] for x in data]
            pre_topics = [x[4] for x in data]
            src = pre_src
            topics = pre_topics

            labels = torch.tensor(self._pad(pre_labels, 0))
            segs = torch.tensor(self._pad(pre_segs, 0))
            # mask = src == 0
            # mask = mask.logical_not()

            clss = torch.tensor(self._pad(pre_clss, -1))
            # mask_cls = clss == -1
            # mask_cls = mask_cls.logical_not()
            clss[clss == -1] = 0

            setattr(self, 'clss', clss.to(device))
            # setattr(self, 'mask_cls', mask_cls.to(device))
            setattr(self, 'src', src)
            setattr(self, 'topics', topics)
            # setattr(self, 'src', src.to(device))
            setattr(self, 'labels', labels.to(device))
            setattr(self, 'segs', segs.to(device))
            # setattr(self, 'mask', mask.to(device))
            if is_test:
                src_str = [x[-2] for x in data]
                setattr(self,'src_txt',src_str)
                tgt_str = [x[-1] for x in data]
                setattr(self, 'tgt_txt', tgt_str)
        else:
            self.real = False

    def parse(self, tokenizer):
        whole_src = self.src
        topics = self.topics
        # whole_src = whole_src.numpy()
        paras = []
        para_sentences = []

        self.parts = len(whole_src)
        for i in range(len(whole_src)):
            ids = whole_src[i]
            topic = topics[i][0]
            topic_ = []
            for phrase in topic:
                topic_ = topic_ + tokenizer.tokenize(phrase)
            topic = topic_
            words = tokenizer.convert_ids_to_tokens(ids)
            sentences = " ".join(words)
            sentences = sentences.split(" [SEP] [CLS] ")
            sentences = [sentence.replace(" [SEP]", "").replace(" [PAD]", "").replace("[CLS]", "").strip() + " [SEP]"
                         for sentence in sentences]
            sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
            print("Before add topic, the length of  Sentences is {}".format(len(sentences)))
            #sentences = [tokenizer.convert_tokens_to_ids(sentence) for sentence in sentences]
            sentences = [tokenizer.convert_tokens_to_ids(topic)] + [tokenizer.convert_tokens_to_ids(sentence) for
                                                                    sentence in sentences]
            para_len = len(sentences)
            print("After add topic, the length of Sentences is {}".format(para_len))
            src, tar1, tar2, labels = [], [], [], []

            # src = [torch.LongTensor(sent) for sent in sentences[0: para_len-1]]
            for i in range(para_len - 1):  # para
                sentence = sentences[i + 1]
                sent_len = len(sentence)
                if sent_len <= 2:
                    continue
                src.append(torch.LongTensor(sentences[i]))
                tar1.append(torch.LongTensor(sentence[0:sent_len - 1]).to(self.device))
                tar2_sent = []
                label = []
                for j in range(sent_len - 1):  # sentence
                    word_sam = [sentence[j + 1]]
                    while len(word_sam) < self.negnum:
                        new_id = np.random.randint(self.vocab_size)
                        while new_id in word_sam:
                            new_id = np.random.randint(self.vocab_size)
                        word_sam.append(new_id)
                    index = np.random.randint(self.negnum)
                    v_ = word_sam[0]
                    word_sam[0] = word_sam[index]
                    word_sam[index] = v_

                    label.append(index)
                    tar2_sent.append(word_sam)
                tar2_sent = torch.LongTensor(tar2_sent).to(self.device)
                label = torch.LongTensor(label).to(self.device)
                tar2.append(tar2_sent)
                labels.append(label)
            src = [torch.LongTensor(sent).to(self.device) for sent in src]
            print("data_loader.py class Batch func parse")
            print("src len is: {}".format(len(src)))
            print("tar1 len is: {}".format(len(tar1)))
            print("labels len is: {}".format(len(self.labels[0])))
            paras.append((src, tar1, tar2, labels))
            para_sentences.append([torch.LongTensor(sentence) for sentence in sentences])
        self.trains = paras
        self.para_sentences = para_sentences
        # sentences = sentences.split(' [SEP] ')

    def __len__(self):
        return self.batch_size


def batchIt(data, batch_size):
    """Yield elements from data in chunks of batch_size."""
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = simple_batch_size_fn(ex, len(minibatch))
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
    if minibatch and len(minibatch) > 0:
        yield minibatch


def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like dayin,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        # logger.info('Loading %s dataset from %s, number of examples: %d' %
        #             (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.bert_data_path + '.' + corpus_type + '.[0-9]*.bert.pt'))
    print("TraingSet list ************")
    print(pts)
    if pts:
        # if (shuffle):
        #     random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.bert_data_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def simple_batch_size_fn(new, count):
    src, labels = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets, tokenizer, batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test

        self.tokenizer = tokenizer

        self.cur_iter = self._next_dataset_iterator(datasets)

        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                del self.cur_dataset
                gc.collect()
                self.cur_dataset = None
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args=self.args,
                            dataset=self.cur_dataset, tokenizer=self.tokenizer, batch_size=self.batch_size,
                            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset, tokenizer, batch_size, device=None, is_test=False,
                 shuffle=False):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle
        self.tokenizer = tokenizer

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0

    def data(self):
        # if self.shuffle:
        #     random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        src = ex['src']
        if ('labels' in ex):
            labels = ex['labels']
        else:
            labels = ex['src_sent_labels']
        segs = ex['segs']
        if (not self.args.use_interval):
            segs = [0] * len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']
        
        if is_test:
            return src,labels,segs,clss,src_txt,tgt_txt
        else:
            return src, labels, segs, clss

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if (len(ex['src']) == 0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if (ex is None):
                continue
            minibatch.append(ex)
            size_so_far = simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 50):
            p_batch = sorted(buffer, key=lambda x: len(x[3]))
            p_batch = buffer
            p_batch = batchIt(p_batch, self.batch_size)
            p_batch = list(p_batch)
            #if (self.shuffle):
            #     random.shuffle(p_batch)
            for b in p_batch:
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)
                if batch.real:
                    batch.parse(self.tokenizer)
                    yield batch
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bert_data_path", default='bertsum_data/cnndm')
    parser.add_argument("-batch_size", default=1000, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?', const=True, default=True)
    device = "cpu"
    args = parser.parse_args()

    dataset = load_dataset(args, 'train', shuffle=False)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    vocab_size = len(tokenizer.vocab)

    train_iter = Dataloader(args, dataset, tokenizer, args.batch_size, device,
                            shuffle=False, is_test=True)
    for i, batch in enumerate(train_iter):
        #pass
        print("iteration: ", i+1)
        print("src len: ", batch.words)
        # print(batch.parts)

        # setattr(self, 'clss', clss.to(device))
        # setattr(self, 'mask_cls', mask_cls.to(device))
        # setattr(self, 'src', src.to(device))
        # setattr(self, 'labels', labels.to(device))
        # setattr(self, 'segs', segs.to(device))
        # setattr(self, 'mask', mask.to(device))

