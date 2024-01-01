import json

import torch

from torch.utils.data import Dataset

from openreviewer.common import model_without_positional_ids


class InstructionTuningDataset(Dataset):
    def __init__(self, args, path_or_data, tokenizer, process_func=None, max_samples=None):
        self.model_type = args.model_type
        # self.path = path
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        if isinstance(path_or_data, list):
            self.data = path_or_data
        else:
            with open(path_or_data, 'r', encoding='utf-8') as f:
                self.data = [json.loads(line) for line in f]
        if max_samples is not None:
            self.data = self.data[:max_samples]
        self.tokenizer = tokenizer
        # self.prompt_key = prompt_key
        # self.response_key = response_key
        self.process_func = process_func if process_func is not None else lambda x,y : (x,y)

        self.model_without_positional_ids = model_without_positional_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate(self, samples):
        batch_size = len(samples)
        max_prompt_length = 0
        max_response_length = 0
        max_length = 0

        tokenized_sample = []
        
        for i, sample in enumerate(samples):
            prompt, response = self.process_func(sample)
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
            response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
            if self.tokenizer.eos_token_id not in response_tokens[-1:]:
                response_tokens = response_tokens + [self.tokenizer.eos_token_id]

            if len(prompt_tokens) > self.max_prompt_length:
                response_tokens = prompt_tokens[self.max_prompt_length:] + response_tokens
                prompt_tokens = prompt_tokens[:self.max_prompt_length]            
            if len(prompt_tokens) + len(response_tokens) > self.max_length:
                response_tokens = response_tokens[:self.max_length - len(prompt_tokens)]

            full_tokens = prompt_tokens + response_tokens

            max_prompt_length = max(max_prompt_length, len(prompt_tokens))
            max_response_length = max(max_response_length, len(response_tokens))
            max_length = max(max_length, len(full_tokens))
            
            tokenized_sample.append((prompt_tokens, response_tokens, full_tokens))

        input_batch = {
            "input_ids": torch.ones((batch_size, max_length), dtype=torch.long) * self.tokenizer.eos_token_id,
            "attention_mask": torch.zeros((batch_size, max_length), dtype=torch.long),
            "position_ids": torch.zeros((batch_size, max_length), dtype=torch.long),
        }

        gen_batch = {
            "input_ids": torch.ones((batch_size, max_prompt_length), dtype=torch.long) * self.tokenizer.pad_token_id,
            "attention_mask": torch.zeros((batch_size, max_prompt_length), dtype=torch.long),
            "position_ids": torch.zeros((batch_size, max_prompt_length), dtype=torch.long),
        }

        other_batch = {
            "labels": torch.ones((batch_size, max_length), dtype=torch.long) * -100,
            "loss_mask": torch.zeros((batch_size, max_length), dtype=torch.long),
            "response_ids": torch.ones((batch_size, max_response_length), dtype=torch.long) * self.tokenizer.eos_token_id,
            "attention_mask": torch.zeros((batch_size, max_response_length), dtype=torch.long),
        }

        for i, (prompt_tokens, response_tokens, full_tokens) in enumerate(tokenized_sample):
            input_batch["input_ids"][i][:len(full_tokens) - 1] = torch.tensor(full_tokens[:-1], dtype=torch.long)
            input_batch["attention_mask"][i][:len(full_tokens) - 1] = 1
            input_batch["position_ids"][i][:len(full_tokens) - 1] = torch.arange(0, len(full_tokens) - 1, dtype=torch.long)

            other_batch["labels"][i][:len(full_tokens) - 1] = torch.tensor(full_tokens[1:], dtype=torch.long)
            other_batch["loss_mask"][i][max(0, len(prompt_tokens) - 1):len(full_tokens) - 1] = 1

            other_batch["response_ids"][i][:len(response_tokens)] = torch.tensor(response_tokens, dtype=torch.long)
            other_batch["attention_mask"][i][:len(response_tokens)] = 1

            gen_batch["input_ids"][i][-len(prompt_tokens):] = torch.tensor(prompt_tokens, dtype=torch.long)
            gen_batch["attention_mask"][i][-len(prompt_tokens):] = 1
            gen_batch["position_ids"][i][-len(prompt_tokens):] = torch.arange(0, len(prompt_tokens), dtype=torch.long)

        if self.model_type in self.model_without_positional_ids:
            input_batch.pop("position_ids")
            gen_batch.pop("position_ids")

        return input_batch, gen_batch, other_batch


class MultiTurnDataset(Dataset):
    def __init__(self, args, path_or_data, tokenizer, process_func=None, max_samples=None):
        self.model_type = args.model_type
        # self.path = path
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        if isinstance(path_or_data, list):
            self.data = path_or_data
        else:
            with open(path_or_data, 'r', encoding='utf-8') as f:
                self.data = [json.loads(line) for line in f]
        if max_samples is not None:
            self.data = self.data[:max_samples]
        self.tokenizer = tokenizer
        # self.prompt_key = prompt_key
        # self.response_key = response_key
        self.process_func = process_func if process_func is not None else lambda x,y : (x,y)

        self.model_without_positional_ids = model_without_positional_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate(self, samples):
        batch_size = len(samples)
        max_length = 0

        tokenized_sample = []
        
        for i, sample in enumerate(samples):
            input_ids, loss_masks = self.process_func(self.tokenizer, sample)
            assert len(input_ids) == len(loss_masks)

            if len(input_ids) > self.max_prompt_length:
                print(f"len(input_ids) = {len(input_ids)} > max_prompt_length = {self.max_prompt_length}")
                input_ids = input_ids[:self.max_prompt_length]
                loss_masks = loss_masks[:self.max_prompt_length]

            max_length = max(max_length, len(input_ids))
            
            tokenized_sample.append((input_ids, loss_masks))

        input_batch = {
            "input_ids": torch.ones((batch_size, max_length), dtype=torch.long) * self.tokenizer.eos_token_id,
            "attention_mask": torch.zeros((batch_size, max_length), dtype=torch.long),
            "position_ids": torch.zeros((batch_size, max_length), dtype=torch.long),
        }

        gen_batch = {
            # "input_ids": torch.ones((batch_size, max_length), dtype=torch.long) * self.tokenizer.pad_token_id,
            # "attention_mask": torch.zeros((batch_size, max_length), dtype=torch.long),
            # "position_ids": torch.zeros((batch_size, max_length), dtype=torch.long),
        }

        other_batch = {
            "labels": torch.ones((batch_size, max_length), dtype=torch.long) * -100,
            "loss_mask": torch.zeros((batch_size, max_length), dtype=torch.long),
            # "response_ids": torch.ones((batch_size, max_length), dtype=torch.long) * self.tokenizer.eos_token_id,
            # "attention_mask": torch.zeros((batch_size, max_length), dtype=torch.long),
        }

        for i, (input_ids, loss_masks) in enumerate(tokenized_sample):
            input_batch["input_ids"][i][:len(input_ids) - 1] = torch.tensor(input_ids[:-1], dtype=torch.long)
            input_batch["attention_mask"][i][:len(input_ids) - 1] = 1
            input_batch["position_ids"][i][:len(input_ids) - 1] = torch.arange(0, len(input_ids) - 1, dtype=torch.long)

            other_batch["labels"][i][:len(input_ids) - 1] = torch.tensor(input_ids[1:], dtype=torch.long)
            other_batch["loss_mask"][i] = torch.tensor(loss_masks, dtype=torch.long)

            # other_batch["response_ids"][i][:len(input_ids)] = torch.tensor(input_ids, dtype=torch.long)
            # other_batch["attention_mask"][i][:len(input_ids)] = 1

            # gen_batch["input_ids"][i][-len(input_ids):] = torch.tensor(input_ids, dtype=torch.long)
            # gen_batch["attention_mask"][i][-len(input_ids):] = 1
            # gen_batch["position_ids"][i][-len(input_ids):] = torch.arange(0, len(input_ids), dtype=torch.long)

        if self.model_type in self.model_without_positional_ids:
            input_batch.pop("position_ids")
            # gen_batch.pop("position_ids")

        return input_batch, gen_batch, other_batch

