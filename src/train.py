import transformers
from sklearn import model_selection
from torch.utils.data import DataLoader
import torch.optim as optim

from src.dataset import *
from src.engine import *
from src.model import *

class run():
    pd_data = pd.read_csv(config.TRAIN_FILES).dropna()
    train_data, valid_data = model_selection.train_test_split(pd_data, random_state=42, test_size=0.1)
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)

    train_dataset = TweetDataset(
        tweet=train_data.text.values,
        selected_text=train_data.selected_text.values,
        sentiment=train_data.sentiment.values,
    )
    train_data_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.TRAIN_BATCH_SIZE
    )

    valid_dataset = TweetDataset(
        tweet=valid_data.text.values,
        selected_text=valid_data.selected_text.values,
        sentiment=valid_data.sentiment.values
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BertBaseUncased().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)

    for epoch in range(config.EPOCHS):
        train_fn(model, device, train_data_loader, optimizer, scheduler)

    torch.save(model, "../model/model.pkl")

    # test_data = pd.read_csv("../input/test.csv").dropna().reset_index(drop=True)
    #
    # test_dataset = TweetDataset(
    #     tweet=test_data.text.values,
    #     sentiment=test_data.sentiment.values
    # )
    # test_data_loader = DataLoader(
    #     test_dataset,
    #     batch_size=config.VALID_BATCH_SIZE,
    #     shuffle=False
    # )
    # eval_fn(model, device, valid_dataset)





if __name__ == '__main__':
    run()