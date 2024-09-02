import numpy as np

# Accuracy lists
his_acc1 = [0.9415, 0.8565, 0.7869, 0.7510, 0.7035, 0.6828, 0.6672, 0.6756]
his_acc2 = [0.9524, 0.8275, 0.8068, 0.7977, 0.7676, 0.7447, 0.6993, 0.6803]
his_acc3 = [0.9504, 0.8170, 0.7583, 0.7325, 0.7151, 0.6978, 0.7070, 0.6943]
his_acc4 = [0.9494, 0.8700, 0.7846, 0.7548, 0.7662, 0.7300, 0.7089, 0.6911]
his_acc5 = [0.9534, 0.8555, 0.8052, 0.7498, 0.7338, 0.7183, 0.6968, 0.6721]
his_acc6 = [0.9484, 0.8820, 0.8019, 0.7695, 0.7600, 0.7362, 0.7110, 0.6811]

# Combine lists into a 2D array
data = np.array([his_acc1, his_acc2, his_acc3, his_acc4, his_acc5, his_acc6])

# Calculate the mean for each column
means = np.mean(data, axis=0)

# Print the means
print("Mean values for each column:")
for i, mean in enumerate(means):
    print(f"Column {i+1}: {mean:.4f}")
