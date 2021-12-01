from robertajrsotcv1_config import *


class SentimentData(Dataset):
    def __init__(self, data, inputs_tokenized):
        self.inputs = inputs_tokenized
        self.text = data['comment_text']
        self.targets = data['target']

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        input = self.inputs[index]
        ids = input['input_ids']
        mask = input['attention_mask']
        token_type_ids = input['token_type_ids']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_predicter = torch.nn.Linear(768, 768)
        self.predicter = torch.nn.Linear(768, 1)
        self.act = torch.nn.ReLU()

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_predicter(pooler)
        output = self.predicter(pooler)
        pooler = self.act(pooler)
        return output


def make_plot(data, title, color="blue", x_label="Batch", y_label="Loss"):
    plt.plot(data, color=color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
