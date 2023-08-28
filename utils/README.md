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
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
#fmodel=torch.jit.freeze(tmodel.eval())
#smodel=torch.fx.symbolic_trace(model)
omodel=ipex.optimize(model.eval())

#b= objinsp.comp_ret(model,fmodel)
#c= objinsp.comp_ret(model,smodel)
d= objinsp.comp_ret(model,omodel)
```