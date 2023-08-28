from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import sys
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader, SequentialSampler
import time
import math
import numpy as np

def collate_fn_(batch, device=None, dtype=None):
    for key, value in batch.items():
        if device:
            batch[key] = value.to(device, non_blocking=True)
        if isinstance(value, torch.FloatTensor) or isinstance(value, torch.cuda.FloatTensor):
            batch[key] = value.to(dtype)
    return batch

def save_jit_model(model, inputs, dtype, device):
    jit_model = None
    print("create trace model")
    in_1 = torch.unsqueeze(inputs["input_ids"][0].clone(), 0)
    in_2 = torch.unsqueeze(inputs["token_type_ids"][0].clone(), 0)
    in_3 = torch.unsqueeze(inputs["attention_mask"][0].clone(), 0)
    jit_model = torch.jit.trace(model,
        (in_1.to(device, non_blocking=True),
        in_2.to(device, non_blocking=True),
        in_3.to(device, non_blocking=True)),
        strict = False)
    jit_model=torch.jit.freeze(jit_model)

    return jit_model

def inference(model, device, eval_loader, num_perf_steps, num_warmup_steps, test_epoches, amp_dtype, total_steps, total_time):
    i = 0
    time_list = []
    for epoch in range(test_epoches + 1):
        for it, batch in enumerate(eval_loader):
            if i + 1 > total_steps:
                break
            if args.data_dir:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
            else:
                inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "token_type_ids": batch["token_type_ids"],
                }
            if i == 0:
                if args.jit:
                    model = save_jit_model(model, inputs, amp_dtype, device)
                if args.device == "hpu" and args.use_hpu_graph:
                    model = ht.hpu.wrap_in_hpu_graph(model)
            # Start benchmarking
            time_start = time.time()
            collate_fn_(inputs, device, amp_dtype)
            outputs = model(**inputs)
            if args.device == 'xpu':
                torch.xpu.synchronize()
            if args.device == 'cuda':
                torch.cuda.synchronize()
            if args.device == 'hpu' and not args.use_hpu_graph:
                htcore.mark_step()
            if args.device != "cpu":
                for k, v in outputs.items():
                    v = v.to(torch.float32).to("cpu")
            time_end = time.time()

            print("Iter {}, Time {}".format(i, time_end - time_start))
            if i + 1 >= num_warmup_steps and i + 1 < total_steps:
                if i + 1 == num_warmup_steps:
                    print("Start calculate time")
                total_time += (time_end - time_start)
                time_list.append((time_end - time_start))
            i+=1

    return total_time, time_list

# args
parser = argparse.ArgumentParser('BERT-Large squad benchmark script', add_help=False)
parser.add_argument("--model-name-or-path", default="bert-large-uncased-whole-word-masking-finetuned-squad", type=str, help="bert-large model name or path")
parser.add_argument("--data-dir", default=None,  type=str, help="Dataset dir")
parser.add_argument("--device", default="cpu", type=str, help="device")
parser.add_argument("--dtype",
        type=str,
        choices=["float32", "bfloat16", "float16"],
        help="bfloat16 or float32 or float16",
        default="bfloat16",
    )
parser.add_argument('--jit', action='store_true')
parser.add_argument('--ipex', action='store_true')
parser.add_argument('--use-hpu-graph', action='store_true')
parser.add_argument("--num-iter", default=100, type=int, help="num iter")
parser.add_argument("--num-warmup", default=10, type=int, help="num warmup")
parser.add_argument("--batch-size", default=32, type=int, help="batch size")
parser.add_argument("--max-length", default=384, type=int, help="The maximum length of a feature (question and context)")
parser.add_argument("--doc-stride", default=128, type=int, help="The authorized overlap between two part of the context when splitting it is needed.")
parser.add_argument("--threads", default=1, type=int, help="threads number to process dataset, better to use cores per instance")
args = parser.parse_args()
print(args)

# dtype
if args.dtype == 'bfloat16':
    amp_enabled = True
    amp_dtype = torch.bfloat16
elif args.dtype == 'float16':
    amp_enabled = True
    amp_dtype = torch.float16
else:
    amp_enabled = False
    amp_dtype = torch.float32

# device
device = torch.device(args.device)
if args.device == 'xpu':
    import intel_extension_for_pytorch as ipex
    autocast = torch.xpu.amp.autocast(enabled=amp_enabled, dtype=amp_dtype if amp_enabled else None)
elif args.device == 'cuda':
    autocast = torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype if amp_enabled else None)
elif args.device == 'cpu':
    if args.ipex:
        import intel_extension_for_pytorch as ipex
    autocast = torch.cpu.amp.autocast(enabled=amp_enabled, dtype=amp_dtype if amp_enabled else None)
elif args.device == 'hpu':
    import habana_frameworks.torch.hpu as hthpu
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch as ht
    autocast = torch.autocast(enabled=amp_enabled, dtype=amp_dtype if amp_enabled else None, device_type="hpu")
else:
    print("Not support this device so far. Please choose:'xpu', 'cuda', 'xpu' or 'hpu'.")
    sys.exit()

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True, cache_dir=None, use_fast=False)
model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path, torch_dtype=amp_dtype)

# wrap model
if args.device == "cpu":
    model = model.to(memory_format=torch.channels_last)
model.eval()
if args.device == "hpu":
    model = model.to(device, non_blocking=True)
else:
    model = model.to(device)
# ipex optimization
if args.ipex:
    if args.device == "cpu":
        model = ipex.optimize(model, dtype=amp_dtype, inplace=True)
    if args.device == "xpu":
        model = torch.xpu.optimize(model, dtype=amp_dtype, level="O1")

# load data & tokenize
if args.data_dir is None:
    dataset = load_dataset("squad", split="validation")
    def tokenize_function(data):
        return tokenizer(data["question"], data["context"], padding='max_length', truncation="only_second", max_length=args.max_length, stride=args.doc_stride, return_tensors="pt")
    eval_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['id', 'title', 'context', 'question', 'answers']).with_format("torch")
else:
    from transformers.data.processors.squad import SquadV1Processor
    from transformers import squad_convert_examples_to_features
    processor = SquadV1Processor()
    examples = processor.get_dev_examples(args.data_dir, filename=None)
    features, eval_dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_length,
            doc_stride=args.doc_stride,
            max_query_length=64,
            is_training=False,
            return_dataset="pt",
            threads=args.threads,
        )

eval_sampler = SequentialSampler(eval_dataset)  # or default value: None. might impact perf
if args.device != "cpu":
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, pin_memory=True, sampler=eval_sampler, drop_last=True)
else:
    eval_loader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

# Inference
steps_per_epoch = len(eval_loader)
total_steps = args.num_iter + args.num_warmup
test_epoches = math.ceil(total_steps / steps_per_epoch)
total_time = 0
with autocast, torch.no_grad():
    total_time, time_list = inference(model, device, eval_loader, args.num_iter, args.num_warmup, test_epoches, amp_dtype, total_steps, total_time)

average_latency = np.mean(time_list)
min_latency = np.min(time_list)
max_latency = np.max(time_list)
p50_latency = np.percentile(time_list, 50)
p90_latency = np.percentile(time_list, 90)
p95_latency = np.percentile(time_list, 95)
p99_latency = np.percentile(time_list, 99)
p999_latency = np.percentile(time_list, 99.9)

print("\n", "-"*10, "Summary:", "-"*10)
print("Throughput (sentences/sec): ", args.batch_size/average_latency)
print("Average latency (ms): ", average_latency * 1000)
print("Min latency (ms): ", min_latency * 1000)
print("Max latency (ms): ", max_latency * 1000)
print("P50 latency (ms): ", p50_latency * 1000)
print("P90 latency (ms): ", p90_latency * 1000)
print("P95 latency (ms): ", p95_latency * 1000)
print("P99 latency (ms): ", p99_latency * 1000)
print("P99.9 latency (ms): ", p999_latency * 1000)