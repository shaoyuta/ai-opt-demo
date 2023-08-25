- nlp(itrex): https://github.com/intel-innersource/frameworks.ai.nlp-toolkit.intel-nlp-toolkit 
- ipex(llm): https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu/tree/llm_feature_branch/examples/cpu/inference/python/llm

|            | WW31/32  |  WW33   |
|------------|----------|---------|
|Script      |nlp(itrex)|ipex(llm)|
|transformers|sandbox-transformers | sandbox-transformers / pip transformers |
|ipex        |innersource-ipex(llm)| innersource-ipex(llm) |
|optimize    |  opt-plan-1 | opt-plan-2 |

- opt-plan-1:
    - optimize in transformers
        - hardcode in "greedy_search()", "beem_search()" (utils.py)
        - change behavia in GPTJAttention::forward()
    - new ipex op "rotary_position_embedding"
- opt-plan-2:
    - replace function in ipex transformer
    - jit and optimize in ipex transformer

----------------------
conda env:
- pt-39-2
    - pip transformers : 4.28.1
    - innersource-ipex ( 174a1874adf ）: 2.1.0+git174a187

- pt-39-4
    - sandbox-transformers ( 1720855 ) : 4.28.1
    - innersource-ipex ( 174a1874adf ）: 2.1.0+git174a187
    