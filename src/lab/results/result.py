import json
import numpy as np
import matplotlib.pyplot as plt

# Load JSON data
file_paths = [
    './basic_lstm_wd10.json',
    './basic_lstm_wd60.json',
    './WeatherModel_Mix_wd10.json',
    './WeatherModel_Mix_wd60.json',
    './basic_ltsf_linear_wd10.json',
    './basic_ltsf_linear_wd60.json'
]

data = {}

for file_path in file_paths:
    with open(file_path, 'r') as file:
        content = json.load(file)
        model_name = content['model_name']
        data[model_name] = {
            'train_loss': content['train_loss'],
            'test_loss': content['test_loss']
        }

# Plotting the loss curves
plt.figure(figsize=(15, 10))

for model_name, losses in data.items():
    plt.plot(losses['train_loss'], label=f'{model_name} Train Loss')
    plt.plot(losses['test_loss'], label=f'{model_name} Test Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Loss Curves for Different Models')
plt.legend()
plt.grid(True)
plt.savefig('./loss_curves.png')

# Calculate the error range for all models
error_ranges = {}

for model_name, losses in data.items():
    mse = np.min(losses['test_loss'])
    mae = np.sqrt(mse)
    error_ranges[model_name] = {
        'MSE': mse,
        'MAE': mae
    }

# Prepare data for plotting
labels = list(error_ranges.keys())
mse_values = [error_ranges[model]['MSE'] for model in labels]
mae_values = [error_ranges[model]['MAE'] for model in labels]

# Plotting the error ranges
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(15, 10))
rects1 = ax.bar(x - width/2, mse_values, width, label='MSE')
rects2 = ax.bar(x + width/2, mae_values, width, label='± MAE')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Models')
ax.set_ylabel('Error')
ax.set_title('Error Ranges for Different Models')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()

fig.tight_layout()
plt.grid(True)
plt.savefig('./error_ranges.png')

# save the error ranges to a JSON file

with open('./error_ranges.json', 'w') as file:
    json.dump(error_ranges, file, indent=4)