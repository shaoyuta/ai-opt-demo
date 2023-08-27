- Sample code
```python
import importlib
customer_path="ai-opt-demo/utils"
if customer_path not in sys.path:
    sys.path.append('ai-opt-demo/utils')
import objinsp
# importlib.reload(objinsp)

m=torchvision.models.resnet50()
tm=torch.jit.trace(m,torch.randn(5, 3, 224, 224))
fm=torch.jit.freeze(tm.eval())

diff=objinsp.comp_ret(m,tm)
diff.filter_out()
print(diff.added_filter)
```