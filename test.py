# def _read_descriptions(file):
#         # id2rel, rel2id = {}, {}
#         rel2des = {}
#         id2des = {}
#         with open(file) as f:
#             for index, line in enumerate(f):
#                 rel = line.strip()
#                 x = rel.split('\t')
#                 rel2des[x[1]] = x[2]
#                 id2des[int(x[0])] = x[2]
#         return rel2des, id2des 

# _read_descriptions('/media/data/thanhnb/Bi/abc_cpl/sadfasdfasdf/data/CFRLFewRel/relation_description.txt')


# Example of a dictionary of dictionaries
# import torch

# # Assume rep_des is your tensor of shape (368, 4096)
# rep_des = torch.randn(368, 4096)  # Replace with your actual tensor

# # Compute the mean embedding across the first dimension (tokens)
# mean_embedding = rep_des.mean(dim=0)

# print(mean_embedding.shape)  # Output should be: torch.Size([4096])
import random
random.seed(42)
random_list_i = random.sample(range(0, 16), 4)
print(random_list_i)
remaining_elements = list(set(range(0, 16)) - set(random_list_i))
print(remaining_elements)
random_list_j = random.sample(remaining_elements, 4)
print(random_list_j)