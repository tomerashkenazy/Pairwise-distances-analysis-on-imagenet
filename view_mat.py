import torch

x = torch.load("results_imagenet_stats/val_dist_matrix_l2.pt")

print(x)
print(x.shape)
print(x.mean())
print(x.std())
print(x.min())
print(x.max())

# Check if the diagonal is very close to zero
diag = torch.diag(x)
tolerance = 1e-6  # Define a small tolerance
is_diag_close_to_zero = torch.all(torch.abs(diag) < tolerance)
print("Is the diagonal very close to zero?", is_diag_close_to_zero.item())

# # ------------ superclass ----------------
# import numpy as np
# y = np.load("/home/tomer_a/Documents/epsilon_bounded_contstim/utils/adjacency_matrix.npy")

# class_num = 19
# axis = 0
# print(y)
# print(y.shape)
# print(y[class_num, :])
# print(y[class_num, :].sum())
# print(np.all(y == y.T))
# unique_rows = np.unique(y, axis=axis)
# print(f"Number of unique rows in y: {unique_rows.shape[axis]}")

