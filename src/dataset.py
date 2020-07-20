import torch
import numpy as np
import pandas as pd


import src.config as config

class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        tweet = " ".join(str(self.tweet[item])).split()
        selected_text = " ".join(str(self.selected_text[item])).split()
        len_sel_text = len(selected_text)
        idx0, idx1 = -1, -1

        for start in (i for i, e in enumerate(tweet) if e == selected_text[0]):
            if tweet[start: start + len_sel_text] == selected_text:
                idx0, idx1 = start, start + len_sel_text - 1
                break

        char_target = [0] * len(tweet)
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if idx0 != -1 and idx1 != -1:
            for i in range(idx0, idx1+1):
                if tweet[i] != " ":
                    char_target[i] = 1
        # [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0]

        tok_tweet = self.tokenizer.encode(self.tweet[item])
        tok_tweet_ids = tok_tweet.ids
        tok_tweet_offsets = tok_tweet.offsets[1: -1]
        tok_tweet_tokens = tok_tweet.tokens

        targets = [0] * (len(tok_tweet_ids) - 2)

        for i, (offset1, offset2) in enumerate(tok_tweet_offsets):
            if sum(char_target[offset1: offset2]) > 0:
                targets[i] = 1
        targets = [0] + targets + [0]
        targets_start = [0] * len(targets)
        targets_end = [0] * len(targets)
        non_zero = np.nonzero(targets)[0]
        if len(non_zero) > 0:
            targets_start[non_zero[0]] = 1
            targets_end[non_zero[-1]] = 1

        attention_mask = [1] * len(tok_tweet_ids)
        token_type_ids = [0] * len(tok_tweet_ids)

        padding_len = self.max_len - len(tok_tweet_ids)
        tok_tweet_ids = tok_tweet_ids + [0] * padding_len
        attention_mask = attention_mask + [0] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len
        targets = targets + [0] * padding_len
        targets_start = targets_start + [0] * padding_len
        targets_end = targets_end + [0] * padding_len

        sentiment = [1, 0, 0]
        if self.sentiment[item] == "positive":
            sentiment = [0, 0, 1]
        if self.sentiment[item] == "negative":
            sentiment = [0, 1, 0]

        return {
            "ids": torch.tensor(tok_tweet_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "targets_end": torch.tensor(targets_end, dtype=torch.long),
            "targets_start": torch.tensor(targets_start, dtype=torch.long),
            "sentiment": torch.tensor(sentiment, dtype=torch.long),
            "padding_len": torch.tensor(padding_len, dtype=torch.long),
            "tok_tweet_token": tok_tweet_tokens,
            "orig_tweet": " ".join(str(self.tweet[item])),
            "orig_selected_text": " ".join(str(self.selected_text[item])),
            "orig_sentiment": " ".join(str(self.sentiment[item])),
        }


class EvalDataset:
    def __init__(self, ids):
        self.ids = ids
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        ids = self.ids[item]
        tweet_token = self.tokenizer.encode(ids)
        tweet_token_ids = tweet_token.ids
        tweet_token_mask = tweet_token.attention_mask
        tweet_token_type_ids = [0] * len(tweet_token_ids)

        if len(tweet_token_ids) < self.max_len:
            padding_len = self.max_len - len(tweet_token_ids)
            tweet_token_ids = tweet_token_ids + [0] * padding_len
            tweet_token_mask = tweet_token_mask + [0] * padding_len
            tweet_token_type_ids = tweet_token_type_ids + [0] * padding_len

        return {
            "ids": torch.tensor(tweet_token_ids, dtype=torch.long),
            "mask": torch.tensor(tweet_token_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(tweet_token_type_ids, dtype=torch.long),
            "padding_len": torch.tensor(padding_len, dtype=torch.long)
        }

if __name__ == '__main__':
    train_file = pd.read_csv(config.TRAIN_FILES).dropna().reset_index(drop=True)
    tds = TweetDataset(
        tweet=train_file.text.values,
        sentiment=train_file.sentiment.values,
        selected_text=train_file.selected_text.values
    )
    print(tds[0])