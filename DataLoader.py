import json
import copy
import logging
import collections
import os
import torch
from torch.utils.data import TensorDataset
from transformers.tokenization_bert import whitespace_tokenize

logger = logging.getLogger(__name__)

class SquadExample(object):
    """A single training/test example for simple sequence classification.
       For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def Get_Date_From_DataSet(input_file):
    examples = []
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    for entry in input_data: #entry 是由title和paragraph构成的dict
        for paragraph in entry["paragraphs"]:   #遍历paragraph这个dict
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:        #处理多个空格的情况
                if is_whitespace(c):        #如果c是空格 将前面空格的flag设为true
                    prev_is_whitespace = True
                else:                       #如果不是空格，前面是空格的话另起一个
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:                   #如果不是空格，加到前面的token后面
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1) #记录每一个char对应的词汇index

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                is_impossible = qa["is_impossible"]
                if not is_impossible:
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]      # 答案开始的char的位置
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset] # 获得answer开始位置词的索引
                    end_position = char_to_word_offset[answer_offset + answer_length - 1] # 获得answer结束位置词的索引
                    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])   # 按照start位置找到的answer
                    cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))   # 数据集中给的answer
                    if actual_text.find(cleaned_answer_text) == -1: # 将tokenize函数的提取和手动提取作比较，如果不能包含数据集中的数据，则warn
                        logger.warning("Could not find answer: '%s' vs. '%s'",
                                       actual_text, cleaned_answer_text)
                        continue
                else:
                    start_position = -1
                    end_position = -1
                    orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    return examples

def convert_examples_to_features(examples, tokenizer, is_training, doc_stride = 128,
                                 max_seq_length = 384, max_query_length = 64,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """Loads a data file into a list of `InputBatch`s. 接入bert的有input_id, input_mask(attention_mask), segment_id, p_mask(标记是否是paragraph_token)"""
    unique_id = 1000000000
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token = tokenizer.pad_token_id
    features = []
    for (example_index, example) in enumerate(examples):    # 对每一个example

        # if example_index % 100 == 0:
        #     logger.info('Converting %s/%s pos %s neg %s', example_index, len(examples), cnt_pos, cnt_neg)

        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))   # origin_doc中的每一个的起始index
            sub_tokens = tokenizer.tokenize(token)
            if not sub_tokens:
                sub_tokens = [unk_token]
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)                 # 最后的分词的token_list对应的origin_doc的index
                all_doc_tokens.append(sub_token)            # 构造分词list

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:  # doc的end_pos对应的是分词之后的尾词的第一个分量
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1  # 这对应尾词后的一个词的倒数第二个分量.
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(  # 主要是过滤掉尾词后面的多余的词
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3     # bert 可以接受的净doc长度

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans): # 对每一个paragraph地 doc_span处理
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = []

            # CLS token at the beginning
            # 添加CLS的token信息
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            p_mask.append(0)
            cls_index = 0

            # Query
            for token in query_tokens:  #添加query信息，seg_id = 1, mask_id = 1
                tokens.append(token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                # 当前文本span内的index向原始文本的偏移量的映射
                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context  # 标记词汇是否在最大语境
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(0)    #标记可以作为答案

            paragraph_len = doc_span.length

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length. 将剩余不够的加入pad
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            if is_training and not span_is_impossible:  # 如果有答案
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    # 为什么out_of_span是True, 就说明span内无答案?
                    # 因为如果答案横跨两个span, |120【200|120】, 则答案至少有200个token, 这是不现实的.
                    # 所以答案一定是在一个span中, 如果out_of_span的情况下, 如果有部分答案，答案一定出现在其他的span
                    # 实际上在answer很短的情况下, 部分out_of_span的情况很少出现.
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    doc_offset = len(query_tokens) + 2                              # query + CLS + SEP
                    start_position = tok_start_position - doc_start + doc_offset    # start_position是在bert中的pos
                    end_position = tok_end_position - doc_start + doc_offset        # end_position是在bert中的pos

            if is_training and span_is_impossible:
                start_position = cls_index
                end_position = cls_index

            if example_index < 20:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % unique_id)
                logger.info("example_index: %s" % example_index)
                logger.info("doc_span_index: %s" % doc_span_index)
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and span_is_impossible:
                    logger.info("impossible example")
                if is_training and not span_is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % start_position)
                    logger.info("end_position: %d" % end_position)
                    logger.info(
                        "answer: %s" % answer_text)

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    cls_index = cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible))
            unique_id += 1
    return features

def load_and_cache_examples(args, tokenizer, mode):
    # Load data features from cache or dataset file
    # 首先判断路径下是否存在cache
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}'.format(
            mode,                   #mode = train/dev
            args.doc_stride,        #120
            args.max_query_length,  #64
            args.max_seq_length     #384
        )
    )
    #如果cache存在，从cache中load
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        if mode == "train":
            examples = Get_Date_From_DataSet("train-v2.0.json")
            logger.info("Creating features from dataset file train-v2.0.json")
        elif mode == "dev":
            examples = Get_Date_From_DataSet("dev-v2.0.json")
            logger.info("Creating features from dataset file dev-v2.0.json")
        else:
            raise Exception("For mode, Only train, dev is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        features = convert_examples_to_features(examples, tokenizer, is_training = True)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_p_flag = torch.tensor([f.p_mask for f in features], dtype = torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids,
                            all_start_positions, all_end_positions,
                            all_cls_index, all_p_flag)
    return dataset
