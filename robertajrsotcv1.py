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

train_accuracy = [[],[],[]]
test_accuracy = []

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


def train_loop(epochs=15):
    for epoch in range(epochs):
        for phase in ['Train', 'Test']:
            if (phase == 'Train'):
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = test_loader
            epoch_loss = 0
            epoch_acc5, epoch_acc10, epoch_acc15 = 0, 0, 0
            count = 0
            for steps, data in tqdm(enumerate(loader, 0)):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)

                outputs = model.forward(ids, mask, token_type_ids)

                loss = loss_function(outputs.squeeze(0), targets)

                epoch_loss += loss.detach().item()
                batch_ansv5=abs(outputs - targets) < 0.05
                batch_ansv10 = abs(outputs - targets) < 0.1
                batch_ansv15 = abs(outputs - targets) < 0.15
                batch_acc5 = (batch_ansv5).sum().item()
                batch_acc10 = (batch_ansv10).sum().item()
                batch_acc15 = (batch_ansv15).sum().item()
                epoch_acc5 += batch_acc5
                epoch_acc10 += batch_acc10
                epoch_acc15 += batch_acc15
                count += targets.size(0)
                if (phase == 'Train'):
                    train_loss.append(loss.detach().item())
                    train_accuracy[0].append(batch_acc5)
                    train_accuracy[1].append(batch_acc10)
                    train_accuracy[2].append(batch_acc15)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    test_loss.append(loss.detach())
                    test_accuracy[0].append(batch_acc5)
                    test_accuracy[1].append(batch_acc10)
                    test_accuracy[2].append(batch_acc15)

            print(f"{phase} Loss: {epoch_loss / len(loader)}")
            print(f"{phase} Accuracy: {epoch_acc5 / count}")
            print(f"{phase} Accuracy: {epoch_acc10 / count}")
            print(f"{phase} Accuracy: {epoch_acc15 / count}")


"""# Fine-tuning model"""
EPOCHS = 15
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
