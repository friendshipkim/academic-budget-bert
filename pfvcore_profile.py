import torch
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from transformers import BertConfig, BertForMaskedLM

# bert base config
config = BertConfig()
model = BertForMaskedLM(config)

bsz = 1
seq_len = 32

inputs = torch.randint(0, 30522, (bsz, seq_len), dtype=torch.long)
flops = FlopCountAnalysis(model, inputs)
total_flops_1 = flops.total()

print("FLOPS of bert_base w/ hidden 768: ", f'{flops.total() / 1e9}G')
print(flop_count_table(flops))

# quarter base
config = BertConfig(hidden_size=768//2, intermediate_size=3072//2, num_attention_heads=12//2)
model = BertForMaskedLM(config)
inputs = torch.randint(0, 30522, (bsz, seq_len), dtype=torch.long)
flops = FlopCountAnalysis(model, inputs)
total_flops_2 = flops.total()

print("FLOPS of bert_base w/ hidden 384: ", f'{flops.total() / 1e9}G')
print(f"ratio: {total_flops_2/total_flops_1}")
print(flop_count_table(flops))
