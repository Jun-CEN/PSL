
import torch

NUM_LABELS = 1000

num_samples_per_label = 30
num_files_to_generate = 3


samples = []
labels = []
with open("./tools/utils/imagenet_val.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        sample, label = line.strip().split(' ')
        samples.append(sample)
        labels.append(int(label))

sample_index = torch.linspace(0, len(samples)-1, len(samples)).long()

selected_label = []
for file_idx in range(num_files_to_generate):

    file_name = f"imagenet_val_s{num_samples_per_label}_{file_idx}.txt"

    selected_sample_indexes = []
    for l in range(NUM_LABELS):
        samples_per_label = sample_index[torch.tensor(labels)==l]
        n = samples_per_label.shape[0]
        selected_sample_indexes_per_label = samples_per_label[torch.randperm(n)[:num_samples_per_label]]
        selected_sample_indexes.append(selected_sample_indexes_per_label)
    selected_sample_indexes = torch.stack(selected_sample_indexes).reshape(-1).tolist()

    lines = ""
    for selected_sample_index in selected_sample_indexes:
        lines += f"{samples[selected_sample_index]} {labels[selected_sample_index]}\n"
    lines = lines[:-1]
    with open(f"./tools/utils/{file_name}", "w") as f:
        f.writelines(lines)

