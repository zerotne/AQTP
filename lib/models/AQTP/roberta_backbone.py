import torch

# 原始的 input_ids 和 attention_mask
input_ids = torch.tensor([[0,   438, 18375,  1630,  1848,  2828,    15,     5,  1255,     2],
                           [0,   627,  7401,     9,     5, 31985, 11642,  1437,     2,     1],
                           [0,   438, 18375,  1630,  1848,  2828,    15,     5,  1255,     2],
                           [0,   627,  7401,     9,     5, 31985, 11642,  1437,     2,     1],
                           [0,   438, 18375,  1630,  1848,  2828,    15,     5,  1255,     2],
                           [0,   627,  7401,     9,     5, 31985, 11642,  1437,     2,     1],
                           [0,   438, 18375,  1630,  1848,  2828,    15,     5,  1255,     2],
                           [0,   627,  7401,     9,     5, 31985, 11642,  1437,     2,     1]], device='cuda:0')

attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]], device='cuda:0')

# 要加入的 cls_id
cls_id = 3

# 在每个 input_ids 子列表的最前面加入 cls_id=3
new_input_ids = torch.cat((torch.full((input_ids.size(0), 1), cls_id, dtype=torch.long, device=input_ids.device), input_ids), dim=1)

# 在每个 attention_mask 子列表的最前面添加 1
new_attention_mask = torch.cat((torch.ones((attention_mask.size(0), 1), dtype=torch.long, device=attention_mask.device), attention_mask), dim=1)

# 打印更新后的 new_input_ids 和 new_attention_mask
print("Updated input_ids:")
print(new_input_ids)

print("\nUpdated attention_mask:")
print(new_attention_mask)