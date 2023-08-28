- Sample code -1
```python
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
from datasets import load_dataset
from torch.utils.data import DataLoader, SequentialSampler
dataset = load_dataset("squad", split="validation")
def tokenize_function(data):
    return tokenizer(data["question"], data["context"], padding='max_length', truncation="only_second", max_length=384, stride=128, return_tensors="pt")
eval_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['id', 'title', 'context', 'question', 'answers']).with_format("torch")

eval_sampler = SequentialSampler(eval_dataset)  
eval_loader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=32)
#fmodel=torch.jit.freeze(tmodel.eval())
#smodel=torch.fx.symbolic_trace(model)
omodel=ipex.optimize(model.eval())

#b= objinsp.comp_ret(model,fmodel)
#c= objinsp.comp_ret(model,smodel)
d= objinsp.comp_ret(model,omodel)
```