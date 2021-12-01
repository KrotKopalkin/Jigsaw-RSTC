from robertajrsotcv1_config import *
from robertajrsotcv1_func import *

"""# Load/prepare data"""

# drop unnecessary columns and rename the remaining ones
data = pd.read_csv('./data/balanced_train.csv')

train_data, test_data = train_test_split(data, test_size=0.1)
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)
"""# Prepare dataset


"""  ## prepare"""

train_tokenized_data = [tokenizer.encode_plus(
    text,
    None,
    add_special_tokens=True,
    max_length=MAX_LEN,
    padding="max_length",
    return_token_type_ids=True,
    truncation=True
)
    for text in train_data['comment_text']]
test_tokenized_data = [tokenizer.encode_plus(
    text,
    None,
    add_special_tokens=True,
    max_length=MAX_LEN,
    padding="max_length",
    return_token_type_ids=True,
    truncation=True
)
    for text in test_data['comment_text']]

train_dataset = SentimentData(train_data, train_tokenized_data)
test_dataset = SentimentData(test_data, test_tokenized_data)

train_loader = DataLoader(train_dataset, **params)
test_loader = DataLoader(test_dataset, **params)

"""# Load RoBERTa"""

# importing libraries for neural network


model = RobertaClass()
model.to(device)

"""# Train function"""

train_loss = []
test_loss = []

train_accuracy = []
test_accuracy = []

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


def train_loop(epochs):
    for epoch in range(epochs):
        for phase in ['Train', 'Test']:
            if (phase == 'Train'):
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = test_loader
            epoch_loss = 0
            epoch_acc = 0
            count = 0
            for steps, data in tqdm(enumerate(loader, 0)):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)

                outputs = model.forward(ids, mask, token_type_ids)

                loss = loss_function(outputs.squeeze(0), targets)

                epoch_loss += loss.detach().item()
                _, max_indices = torch.max(outputs.data, dim=1)
                batch_acc = (abs(max_indices - targets) < 0.05).sum().item()
                epoch_acc += batch_acc

                count += targets.size(0)
                if (phase == 'Train'):
                    train_loss.append(loss.detach().item())
                    train_accuracy.append(batch_acc)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    test_loss.append(loss.detach())
                    test_accuracy.append(batch_acc)

            print(f"{phase} Loss: {epoch_loss / steps}")
            print(f"{phase} Accuracy: {epoch_acc / count}")


"""# Fine-tuning model"""

train_loop(EPOCHS)

"""# Visualizing"""
make_plot(train_loss, "Train Loss", "blue")
make_plot(test_loss, "Test Loss", 'orange')
make_plot(train_accuracy, "Train Accuracy", "blue")
make_plot(test_accuracy, "Test Accuracy", 'orange')
"""<a id='section07'></a>
### Save model
"""

save_path = "/content/drive/MyDrive/Colab Notebooks/BroutonLab/Kaggle/"
torch.save(model, save_path + 'kaggle_roberta.pt')
print('All files saved')
print('Congratulations, you complete this tutorial')
