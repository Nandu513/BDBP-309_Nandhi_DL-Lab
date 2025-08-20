import numpy as np

def quantile_normalize(data):
    sorted_idx = np.argsort(data, axis=0)
    sorted_data = np.sort(data, axis=0)
    mean_values = np.mean(sorted_data, axis=1)
    normalized_data = np.zeros_like(data)
    for col in range(data.shape[1]):
        normalized_data[sorted_idx[:, col], col] = mean_values
    return normalized_data

# Load
data = np.load("1000G_float64.npy")

# Landmark genes (first 943 rows)
landmark = data[:943, :]
landmark_norm = quantile_normalize(landmark)

# Target genes (remaining rows)
target = data[943:, :]
target_norm = quantile_normalize(target)

# Combine back
data_reqnorm = np.vstack((landmark_norm, target_norm))

print(f"Final normalized matrix shape: {data_reqnorm.shape}")

# Save
np.save("1000G_reqnorm_float64.npy", data_reqnorm)
