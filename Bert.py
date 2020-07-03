import csv
import os
import codecs
import json
import random
import logging
import argparse

from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange

from sklearn import metrics
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s-%(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def load_data():
    data = []
    sample_num = 25000

    with open('./data/positive_process.data', 'r', encoding='utf-8') as f:
        sentences = f.readlines()
        for sentence in sentences[:sample_num]:
            words = [x for x in sentence.strip().split('\t')]
            words = ''.join(words)
            data.append([words, 0])

    with open('./data/negative_process.data', 'r', encoding='utf-8') as f:
        sentences = f.readlines()
        for sentence in sentences[:sample_num]:
            words = [x for x in sentence.strip().split('\t')]
            words = ''.join(words)
            data.append([words, 1])

    random.shuffle(data)
    return data


class MyPro:
    '''自定义数据读取方法
    Returns:
        examples: 数据集，包含index、中文文本、类别三个部分
    '''
    train_dev_data, test_data = train_test_split(load_data(), test_size=0.1)
    train_data, dev_data = train_test_split(train_dev_data, test_size=0.1)
    def get_train_examples(self, data_dir):
        return self._create_examples(self.train_data, 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(self.dev_data, 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(self.test_data, 'test')

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, lists, set_type):
        examples = []
        for (i, infor) in enumerate(lists):
            guid = "%s-%s" % (set_type, i)
            text_a = infor[0]
            label = infor[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples




def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, show_exp=True):
    '''Loads a data file into a list of `InputBatch`s.
    Args:
        examples      : [List] 输入样本，包括question, label, index
        label_list    : [List] 所有可能的类别，可以是int、str等，如['book', 'city', ...]
        max_seq_length: [int] 文本最大长度
        tokenizer     : [Method] 分词方法
    Returns:
        features:
            input_ids  : [ListOf] token的id，在chinese模式中就是每个分词的id，对应一个word vector
            input_mask : [ListOfInt] 真实字符对应1，补全字符对应0
            segment_ids: [ListOfInt] 句子标识符，第一句全为0，第二句全为1
            label_id   : [ListOfInt] 将Label_list转化为相应的id表示
    '''
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5 and show_exp:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def val(model, processor, args, label_list, tokenizer, device):
    '''模型验证

    Args:
        model: 模型
	processor: 数据读取方法
	args: 参数表
	label_list: 所有可能类别
	tokenizer: 分词方法
	device

    Returns:
        f1: F1值
    '''
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, show_exp=False)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    val_acc_sum, n = 0.0, 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        n += label_ids.shape[0]
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            pred = logits.max(1)[1]
            val_acc_sum += (pred == label_ids).sum().cpu().item()
            predict = np.hstack((predict, pred.cpu().numpy()))
            gt = np.hstack((gt, label_ids.cpu().numpy()))

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()

    print(len(gt))
    f1 = np.mean(metrics.f1_score(predict, gt, average=None))
    acc = val_acc_sum / n
    print('val acc = {}'.format(acc))

    return f1


def test(model, processor, args, label_list, tokenizer, device):
    '''模型测试

    Args:
        model: 模型
	processor: 数据读取方法
	args: 参数表
	label_list: 所有可能类别
	tokenizer: 分词方法
	device

    Returns:
        f1: F1值
    '''
    test_examples = processor.get_test_examples(args.data_dir)
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    model.eval()
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    test_acc_sum, n = 0.0, 0
    for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        n += label_ids.shape[0]
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            pred = logits.max(1)[1]
            test_acc_sum += (pred == label_ids).sum().cpu().item()
            predict = np.hstack((predict, pred.cpu().numpy()))
            gt = np.hstack((gt, label_ids.cpu().numpy()))

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()

    f1 = np.mean(metrics.f1_score(predict, gt, average=None))
    acc = test_acc_sum / n
    print('test acc = {}'.format(acc))
    print('F1 score in test set is {}'.format(f1))

    return f1


def main():
    # ArgumentParser对象保存了所有必要的信息，用以将命令行参数解析为相应的python数据类型
    parser = argparse.ArgumentParser()

    # required parameters
    # 调用add_argument()向ArgumentParser对象添加命令行参数信息，这些信息告诉ArgumentParser对象如何处理命令行参数
    parser.add_argument("--data_dir",
                        default='/data/users/zfsun3/text_classification/data/',
                        type=str,
                        # required = True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model",
                        default='bert-base-chinese',
                        type=str,
                        # required = True,
                        help="choose [bert-base-chinese] mode.")
    parser.add_argument("--task_name",
                        default='MyPro',
                        type=str,
                        # required = True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='checkpoints/',
                        type=str,
                        # required = True,
                        help="The output directory where the model checkpoints will be written")
    parser.add_argument("--model_save_pth",
                        default='checkpoints/bert_classification.pth',
                        type=str,
                        # required = True,
                        help="The output directory where the model checkpoints will be written")

    # other parameters
    parser.add_argument("--max_seq_length",
                        default=52,
                        type=int,
                        help="字符串最大长度")
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="训练模式")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="验证模式")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="英文字符的大小写转换，对于中文来说没啥用")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="训练时batch大小")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="验证时batch大小")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="Adam初始学习步长")
    parser.add_argument("--num_train_epochs",
                        default=5.0,
                        type=float,
                        help="训练的epochs次数")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for."
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="用不用CUDA")
    parser.add_argument("--local_rank",
                        default=-1,
                        type=int,
                        help="local_rank for distributed training on gpus.")
    parser.add_argument("--seed",
                        default=777,
                        type=int,
                        help="初始化时的随机数种子")
    parser.add_argument("--gradient_accumulation_steps",
                        default=1,
                        type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--optimize_on_cpu",
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU.")
    parser.add_argument("--fp16",
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit.")
    parser.add_argument("--loss_scale",
                        default=128,
                        type=float,
                        help="Loss scaling, positive power of 2 values can improve fp16 convergence.")

    args = parser.parse_args()

    # 对模型输入进行处理的processor，git上可能都是针对英文的processor
    processors = {'mypro': MyPro}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                              args.local_rank),
                                                          num_labels=2)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                           for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                           for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()

    global_step = 0
    if args.do_train:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, show_exp=False)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        best_score = 0
        flags = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()

            f1 = val(model, processor, args, label_list, tokenizer, device)
            if f1 > best_score:
                best_score = f1
                print('*f1 score = {}'.format(f1))
                flags = 0
                checkpoint = {
                    'state_dict': model.state_dict()
                }
                torch.save(checkpoint, args.model_save_pth)
            else:
                print('f1 score = {}'.format(f1))
                flags += 1
                if flags >= 6:
                    break

    model.load_state_dict(torch.load(args.model_save_pth)['state_dict'])
    test(model, processor, args, label_list, tokenizer, device)


if __name__ == '__main__':
    main()