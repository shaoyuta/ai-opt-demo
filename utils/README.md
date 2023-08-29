- Sample code -1
```python
import torch
import intel_extension_for_pytorch as ipex
import torchvision
from torchinfo import summary
from torch import nn
import inspect
import sys
import importlib
customer_path="/home/taosy/repo/shaoyu/ai-opt-demo/utils"
if customer_path not in sys.path:
    sys.path.append('/home/taosy/repo/shaoyu/ai-opt-demo/utils')
import objinsp

import importlib
customer_path="ai-opt-demo/utils"
if customer_path not in sys.path:
    sys.path.append('ai-opt-demo/utils')
import objinsp
# importlib.reload(objinsp)

m=torch.nn.Linear(2,2)
tm=torch.jit.trace(m,torch.randn(2,2))
fm=torch.jit.freeze(tm.eval())
sm=torch.fx.symbolic_trace(m)
om=ipex.optimize(m.eval())

diff_a= objinsp.comp_ret(m,tm)
diff_b= objinsp.comp_ret(m,fm)
diff_c= objinsp.comp_ret(m,sm)
diff_d= objinsp.comp_ret(m,om)

print(diff_a.added_filter)
print(diff_a.both_filter_diff)
```

- Sample code -2
```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
from datasets import load_dataset
from torch.utils.data import DataLoader, SequentialSampler
dataset = load_dataset("squad", split="validation")
def tokenize_function(data):
    return tokenizer(data["question"], data["context"], padding='max_length', truncation="only_second", max_length=384, stride=128, return_tensors="pt")
eval_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['id', 'title', 'context', 'question', 'answers']).with_format("torch")

eval_sampler = SequentialSampler(eval_dataset)  # or default value: None. might impact perf
eval_loader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=32)
p=next(iter(eval_loader))
in_1 = torch.unsqueeze(p["input_ids"][0].clone(), 0)
in_2 = torch.unsqueeze(p["token_type_ids"][0].clone(), 0)
in_3 = torch.unsqueeze(p["attention_mask"][0].clone(), 0)

tmodel = torch.jit.trace(model, 
       (in_1.to("cpu", non_blocking=True),
        in_2.to("cpu", non_blocking=True),
        in_3.to("cpu", non_blocking=True)),strict = False)
fmodel=torch.jit.freeze(tmodel.eval())
#smodel=torch.fx.symbolic_trace(model)
omodel=ipex.optimize(model.eval())

a= objinsp.comp_ret(model,tmodel)
b= objinsp.comp_ret(model,fmodel)
#c= objinsp.comp_ret(model,smodel)
d= objinsp.comp_ret(model,omodel)
```