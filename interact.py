# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
print(sys.path)
#export PYTHONPATH=$PYTHONPATH: /
import json
import logging
import random
import tqdm
from tqdm import tqdm
import pickle
from argparse import ArgumentParser
from pprint import pformat
import torch
import torch.nn.functional as F
#import train

from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
#from train import SPECIAL_TOKENS
#from train import build_para_only_input_from_segments, build_qa_only_input_from_segments
#from dataloader import get_positional_dataset_from_file

class_map = {
    'causal': 'general',
    'judgemental': 'general',
    'instrumental': 'general',
    'general': 'general',
    'general_concept_completion': 'general',
    'specific': 'specific',
    'specific_concept_completion': 'specific'
}

SPECIAL_TOKENS = [
    "<bos>", "<eos>", "<paragraph>", "<answer-general>", "<answer-specific>",
    "<question-general>", "<question-specific>", "<pad>"
]

def get_position(para_ids, ans_ids, ans_prefix_ids):
    diff_index = -1
    # Find the first token where the paragraph and answer prefix differ
    for i, (pid, apid) in enumerate(zip(para_ids, ans_prefix_ids)):
        if pid != apid:
            diff_index = i
            break
    if diff_index == -1:
        diff_index = min(len(ans_prefix_ids), len(para_ids))
    # Starting from this token, we take a conservative overlap
    return (diff_index, min(diff_index + len(ans_ids), len(para_ids)))

def build_para_only_input_from_segments(data_point, tokenizer):
    """A paragraph-only version of build_input_from_segments()."""
    bos, eos, paragraph, answer_general, answer_specific, question_general, question_specific = \
        tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])

    curr_para = data_point['paragraph']
    ans_start = data_point['answer_position_tokenized'][0]
    ans_end = data_point['answer_position_tokenized'][1]

    sequence = [bos] + curr_para
    if data_point['class'] == 'general':
        # This segmentation will encode positional information
        token_types = [
            answer_general if ((i - 1) >= ans_start and (i - 1) < ans_end) else paragraph
            for i in range(len(curr_para) + 1)
        ]
    elif data_point['class'] == 'specific':
        # This segmentation will encode positional information
        token_types = [
            answer_specific if ((i - 1) >= ans_start and (i - 1) < ans_end) else paragraph
            for i in range(len(curr_para) + 1)
        ]
    lm_labels = [-1 for _ in range(len(curr_para) + 1)]

    assert len(sequence) == len(token_types)
    assert len(token_types) == len(lm_labels)

    instance = {
        "input_ids": sequence,
        "token_type_ids": token_types,
        "lm_labels": lm_labels
    }
    return instance, sequence


def build_qa_only_input_from_segments(data_point, tokenizer, with_eos=True):
    """A QA-only version of build_input_from_segments()."""
    bos, eos, paragraph, answer_general, answer_specific, question_general, question_specific = \
        tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])

    curr_ans = data_point['answer']
    curr_ques = data_point['question']

    sequence = []
    token_types = []
    lm_labels = []

    if data_point['class'] == 'general':
        sequence.extend([answer_general] + curr_ans)
        token_types.extend([answer_general for _ in range(len(curr_ans) + 1)])
        lm_labels.extend([-1 for _ in range(len(curr_ans) + 1)])

        if with_eos is True:
            sequence.extend([question_general] + curr_ques + [eos])
            token_types.extend([question_general for _ in range(len(curr_ques) + 2)])
            lm_labels.extend([-1] + curr_ques + [eos])
        else:
            sequence.extend([question_general] + curr_ques)
            token_types.extend([question_general for _ in range(len(curr_ques) + 1)])
            lm_labels.extend([-1] + curr_ques)

    elif data_point['class'] == 'specific':
        sequence.extend([answer_specific] + curr_ans)
        token_types.extend([answer_specific for _ in range(len(curr_ans) + 1)])
        lm_labels.extend([-1 for _ in range(len(curr_ans) + 1)])

        if with_eos is True:
            sequence.extend([question_specific] + curr_ques + [eos])
            token_types.extend([question_specific for _ in range(len(curr_ques) + 2)])
            lm_labels.extend([-1] + curr_ques + [eos])
        else:
            sequence.extend([question_specific] + curr_ques)
            token_types.extend([question_specific for _ in range(len(curr_ques) + 1)])
            lm_labels.extend([-1] + curr_ques)

    assert len(sequence) == len(token_types)
    assert len(token_types) == len(lm_labels)

    instance = {
        "input_ids": sequence,
        "token_type_ids": token_types,
        "lm_labels": lm_labels
    }
    return instance, sequence


def get_positional_dataset_from_file(tokenizer, file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    truncated_sequences = 0
    for inst in tqdm(data):
        tokenized_para = tokenizer.tokenize(inst['paragraph'])
        tokenized_question = tokenizer.tokenize(inst['question'])
        tokenized_answer = tokenizer.tokenize(inst['answer'])
        tokenized_ans_prefix = tokenizer.tokenize(inst['paragraph'][:inst['answer_position']])

        total_seq_len = len(tokenized_para) + len(tokenized_answer) + len(tokenized_question) + 4

        if total_seq_len > tokenizer.max_len:
            # Heuristic to chop off extra tokens in paragraphs
            tokenized_para = tokenized_para[:-1 * (total_seq_len - tokenizer.max_len + 1)]
            truncated_sequences += 1
            assert len(tokenized_para) + len(tokenized_answer) + len(tokenized_question) + 4 < tokenizer.max_len

        inst['paragraph'] = tokenizer.convert_tokens_to_ids(tokenized_para)
        inst['question'] = tokenizer.convert_tokens_to_ids(tokenized_question)
        inst['answer'] = tokenizer.convert_tokens_to_ids(tokenized_answer)
        ans_prefix_ids = tokenizer.convert_tokens_to_ids(tokenized_ans_prefix)
        inst['class'] = class_map[inst['class']]
        inst['answer_position_tokenized'] = get_position(inst['paragraph'], inst['answer'], ans_prefix_ids)
        pass
    logger = logging.getLogger(__file__)
    logger.info("%d / %d sequences truncated due to positional embedding restriction" % (truncated_sequences, len(data)))

    return data

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(inst, tokenizer, model, args, para_cache):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    inst['original_question'] = inst['question']
    inst['question'] = []

    # Ignore the paragraph while building the input instance and token type ids
    instance, _ = build_qa_only_input_from_segments(inst, tokenizer, with_eos=False)
    input_ids = torch.tensor(instance['input_ids'], device=args.device).unsqueeze(0)
    token_type_ids = torch.tensor(instance['token_type_ids'], device=args.device).unsqueeze(0)

    # Initialize the past using the paragraph cache hidden representations
    past = para_cache["hidden_states"]

    prev = None
    # This will be either <question-general> or <question-specific>, used to create subsequent inputs
    token_type = instance['token_type_ids'][-1]

    for i in range(args.max_length):
        if i != 0:
            # In the first step of decoding, we want to look at the entire answer
            # In the subsequent steps, we can just cache the hidden representations from previous steps
            input_ids = prev.unsqueeze(0)
            token_type_ids = torch.tensor([token_type]).unsqueeze(0).to(args.device)

        logits, past = model(input_ids, token_type_ids=token_type_ids, past=past)

        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break

        inst['question'].append(prev.item())

    return inst


def run():
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gpt", help="gpt or gpt2")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--filename", type=str, default="data/instances_dev.pkl", help="File to use for decoding")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")

    # While using SQUASH in the pipeline mode, prefer using the --key flag
    parser.add_argument("--key", type=str, default=None,
                        help="Override the default settings if the key is set, used in pipeline mode")
    args = parser.parse_args()

    if args.key is not None:
        # Override some the filename and top_p default settings if args.key is set
        # This is done when the question generation module is being used in the SQUASH pipeline mode
        args.filename = "temp/%s/input.pkl" % args.key
        print(args.key)

        with open("temp/%s/metadata.json" % args.key, "r") as f:
            metadata = json.loads(f.read())
        args.top_p = metadata["settings"]["top_p"]

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")

    if args.model_type == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
        model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)
    else:
        tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_checkpoint)
        model = OpenAIGPTLMHeadModel.from_pretrained(args.model_checkpoint)

    model.to(args.device)
    model.eval()
    print(args.filename)
    data = get_positional_dataset_from_file(tokenizer, args.filename)
    final_output_dict = {
        "version": "squash-2.0",
        "data": [{
            "paragraphs": []
        }]
    }
    question_number = 0

    para_cache = {
        "index": None,
        "hidden_states": None
    }

    for inst in tqdm(data):
        with torch.no_grad():
            para_index = inst["para_index"]
            # Questions from the same paragraph all appear together
            # We can re-use the paragraph hidden representations for different questions in the same paragraph
            if para_index != para_cache["index"]:
                # Since we have moved to a new paragraph, generate its cache
                para_cache["hidden_states"] = None
                # Ignore the answer and question while building the input
                instance, _ = build_para_only_input_from_segments(inst, tokenizer)
                input_ids = torch.tensor(instance['input_ids'], device=args.device).unsqueeze(0)
                token_type_ids = torch.tensor(instance['token_type_ids'], device=args.device).unsqueeze(0)

                # Run a forward pass to generate the para caches
                _, para_cache["hidden_states"] = model(input_ids, token_type_ids=token_type_ids)

            # Sample a question using the paragraph cache
            output = sample_sequence(inst, tokenizer, model, args, para_cache)

        original_paragraph = tokenizer.decode(output['paragraph'])
        generated_question = tokenizer.decode(output['question'])
        original_answer = tokenizer.decode(output['answer'])
        para_index = inst['para_index']
        para_cache["index"] = inst['para_index']

        # verify whether the answer position is correct, since this will be utilized for filtering
        original_ans_position = output["answer_position"]
        if original_paragraph[output["answer_position"]:output["answer_position"] + len(original_answer)] != original_answer:
            # This should never be executed, only used as a last resort
            logger.info("Answer mismatch!")
            original_ans_position = original_paragraph.index(original_answer)

        # Output in a SQUAD-like format with questions clumped together under their parent paragraph
        if len(final_output_dict["data"][0]["paragraphs"]) > para_index:
            # verify whether the paragraph text is identical
            assert original_paragraph == final_output_dict["data"][0]["paragraphs"][para_index]['context']
            # append the question answer pair
            final_output_dict["data"][0]["paragraphs"][para_index]['qas'].append({
                'id': 'question_%d' % question_number,
                'question': generated_question,
                'answers': [{
                    'text': original_answer,
                    'answer_start': original_ans_position,
                }],
                'class': output['class'],
                'algorithm': output['algorithm'],
                'is_impossible': False
            })
        else:
            # add a new question to the list of QA pairs
            final_output_dict['data'][0]['paragraphs'].append({
                'context': original_paragraph,
                'qas': [{
                    'id': 'question_%d' % question_number,
                    'question': generated_question,
                    'answers': [{
                        'text': original_answer,
                        'answer_start': original_ans_position,
                    }],
                    'class': output['class'],
                    'algorithm': output['algorithm'],
                    'is_impossible': False
                }]
            })

        question_number += 1

    with open("temp/%s/generated_questions.json" % args.key, "w") as f:
        f.write(json.dumps(final_output_dict))


if __name__ == "__main__":
    run()