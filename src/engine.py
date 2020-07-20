import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.dataset import *
from src.utils import *

def loss_fn(o1, o2, t1, t2):
    l1 = nn.BCEWithLogitsLoss()(o1, t1)
    l2 = nn.BCEWithLogitsLoss()(o2, t2)
    return l1 + l2

def train_fn(model, device, data_loader, optimizer, scheduler=None):
    model.train()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    for batch_i, data in enumerate(tk0):
        ids = data["ids"]
        mask = data["mask"]
        token_type_ids = data["token_type_ids"]
        targets_start = data["targets_start"]
        targets_end = data["targets_end"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)


        optimizer.zero_grad()

        o1, o2 = model(ids, token_type_ids=token_type_ids, attention_mask=mask)
        if batch_i == 0:
            with open("../outputs/output.txt", "w", encoding="utf-8") as f:
                np.savetxt(f, o1.cpu().detach().numpy())

        loss = loss_fn(o1, o2, targets_start, targets_end)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.update(loss.item(), n=ids.size()[0])
        tk0.set_postfix(loss=loss.item())

def eval_fn(model, device, data_loader):
    fin_start = []
    fin_end = []
    fin_padding = []
    fin_tweet_token_ids = []

    model.eval()
    with torch.no_grad():
        for batch_id, data in enumerate(data_loader):
            ids = data["ids"]
            mask = data["mask"]
            token_type_ids = data["token_type_ids"]


            padding_len = data["padding_len"]


            ids = ids.to(device, dtype=torch.long)
            mask = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)

            output_start, output_end = model(
                ids=ids,
                token_type_ids=token_type_ids,
                attention_mask=mask
            )

            fin_start.append(torch.sigmoid(output_start).cpu().detach().numpy())
            fin_end.append((torch.sigmoid(output_end).cpu().detach().numpy()))

            fin_padding.extend(padding_len.cpu().detach().numpy().tolist())
            fin_tweet_token_ids.append(ids.cpu().detach().numpy().tolist())


    fin_start = np.vstack(fin_start)
    fin_end = np.vstack(fin_end)

    fin_tweet_token_ids = np.vstack(fin_tweet_token_ids)

    threshold = 0.8
    all_outputs = []
    for i, padding in enumerate(fin_padding):
        start, end = -1, -1
        fin_start_i = fin_start[i]
        fin_end_i = fin_end[i]
        if padding > 0:
            start = fin_start[i, 1: -1][:-padding] >= threshold
            end = fin_end[i, 1: -1][:-padding] >= threshold
        else:
            start = fin_start[i, 1: -1] >= threshold
            end = fin_end[i, 1: -1] >= threshold

        start_non_zero = np.nonzero(start)[0]
        end_non_zero = np.nonzero(end)[0]

        mask = [0] * len(start)
        if len(start_non_zero) > 0:
            idx_start = mask[start_non_zero[0]]
            if len(end_non_zero) > 0:
                idx_end = mask[end_non_zero[0]]
            else:
                idx_end = idx_start
        else:
            idx_start, idx_end = 0, 0

        for idx in range(idx_start, idx_end+1):
            mask[idx] = 1

        tweet_token_ids = fin_tweet_token_ids[i]
        output_token_ids = [d for j, d in enumerate(tweet_token_ids[1:]) if j < len(mask)-21 and mask[j-1] == 1]
        output_token = config.TOKENIZER.decode(output_token_ids)
        all_outputs.append(output_token)
    sample = pd.read_csv("../input/test.csv").dropna().reset_index(drop=True)
    print(len(all_outputs))
    print(len(sample.textID.values))
    sample.loc[:, 'selected_text'] = all_outputs
    print(all_outputs[192])
    just_select_test_sample = sample.loc[:, "selected_text"]
    print(type(just_select_test_sample))
    print(type(sample))

    sample.to_csv("../outputs/submission.csv", index=False)
    just_select_test_sample.to_csv("../outputs/just_select_test_sample.csv", index=False)



if __name__ == '__main__':
    model = torch.load("../model/model.pkl")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_data = pd.read_csv("../input/test.csv").dropna().reset_index(drop=True)

    test_dataset = EvalDataset(
        ids=test_data.text.values
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False
    )
    eval_fn(model, device, test_data_loader)