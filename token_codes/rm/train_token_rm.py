import os
from transformers import AutoConfig
import warnings
from transformers.trainer_pt_utils import nested_detach
from collections import Counter

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
)
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import PreTrainedModel
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import torch.nn.functional as F
import wandb
import random

from datasets import Dataset
import json

from accelerate.utils import gather_object
from trl.trainer.utils import RewardDataCollatorWithPadding, compute_accuracy, print_rich_table
from collections import defaultdict
import pandas as pd

class YCDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset = self.read_jsonl_file()
        self.hf_dataset = self.format_dict()

    def read_jsonl_file(self):
        dataset = []
        with open(self.file_path, 'r') as f:
            for line in f:
                data_dict = json.loads(line.strip())
                dataset.append(data_dict)
        return dataset
    
    def format_dict(self):
        data_dict = {}
        data_dict["prompt"] = []
        data_dict["answer"] = []
        data_dict["words"] = []
        data_dict["rewards"] = []
        data_dict["sentence_reward"] = []
        
        for data in self.dataset:
            try:
                word_list = data['word_list']
                reward_list = data['reward_list']
                    
                data_dict["prompt"].append(data["prompt"])
                data_dict["answer"].append(data["answer"])
                data_dict["sentence_reward"].append(data["sentence_reward"])
                data_dict["words"].append(word_list)
                data_dict["rewards"].append(reward_list) 
            except:
                print(data) 
                continue 
            
        hf_dataset =  Dataset.from_dict(data_dict)
        return hf_dataset

def set_seed(seed):
    random.seed(seed)  # Python内建的随机库
    np.random.seed(seed)  # Numpy库
    torch.manual_seed(seed)  # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为所有CUDA设备设置种子
        torch.cuda.manual_seed_all(seed)  # 为所有CUDA设备设置种子，等同于上一行
        torch.backends.cudnn.deterministic = True  # 确保CUDA的确定性行为
        torch.backends.cudnn.benchmark = False  # 关闭动态卷积加速，避免因优化导致的不确定性

set_seed(42)
tqdm.pandas()

@dataclass
class ExtraConfig:
    train_dataset_path: str = field(default='rl_datasets/train/train_dataset_3v1.json')
    test_dataset_path: str = field(default='rl_datasets/eval/eval_dataset_200.json')
    tokenizer_name: str = field(default="allenai/longformer-base-4096")
    num_class_labels: int = field(default=2)
    pos_neg_ratio: float = field(default=3.0)
    class_weight: float = field(default=4.0)
    lora_yc_r: int = field(default=32)
    lora_yc_alpha: int = field(default=64)
    lora_yc_dropout: float = field(default=0.1)
    use_lora: bool = field(default=True)
    local_weight: float = field(default=1.0)
    global_weight: float = field(default=0.25)
    

@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_process = []
        features_unprocess = {
            'labels': [],
            'answer_mask': [],
            'sentence_reward': [],
            'probability_mask': []
        }
        # check if we have a margin. If we do, we need to batch it as well
        for feature in features:
            # check if the keys are named as expected
            if (
                "input_ids" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected` and `attention_mask_rejected`"
                )

            features_process.append(
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"]
                }
            )
            
            labels_padded = feature['labels'] + [0] * (self.max_length - len(feature['labels']))
            answer_mask_padded = feature['answer_mask'] + [0] * (self.max_length - len(feature['answer_mask']))
            probability_mask_padded = feature['probability_mask'] + [0] * (self.max_length - len(feature['probability_mask']))
            sentence_reward = int(feature['sentence_reward'])
            
            # TODO: label 要从0开始!!!
            
            features_unprocess['labels'].append(labels_padded)
            features_unprocess['answer_mask'].append(answer_mask_padded)
            features_unprocess['probability_mask'].append(probability_mask_padded)
            features_unprocess['sentence_reward'].append(sentence_reward)
            
        batch = self.tokenizer.pad(
            features_process,
            padding='max_length',
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        batch_labels = torch.tensor(features_unprocess['labels'])
        batch_answer_mask = torch.tensor(features_unprocess['answer_mask'])
        batch_sentence_reward = torch.tensor(features_unprocess['sentence_reward'])
        batch_probability_mask = torch.tensor(features_unprocess['probability_mask'])

        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch_labels,
            "answer_mask": batch_answer_mask,
            "probability_mask": batch_probability_mask,
            "sentence_reward": batch_sentence_reward
        }
        return batch

class MyTrainer(RewardTrainer):
    def __init__(self, extra_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_config = extra_config
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
            
        if "answer_mask" in inputs:
            answer_mask = inputs.pop("answer_mask")
            
        if "sentence_reward" in inputs:
            sentence_reward = inputs.pop("sentence_reward")
        else:
            sentence_reward = None
            
        if "probability_mask" in inputs:
            probability_mask = inputs.pop("probability_mask")
        else:
            probability_mask = None
            
        outputs = model(**inputs)
        logits = outputs.logits
        
        logits_tensor = logits.view(-1, model.config.num_labels)
        labels_tensor = labels.view(-1)
        
        if return_outputs:
            active_indices = (answer_mask.view(-1) == 1)
        else:
            active_indices = torch.logical_and(answer_mask.view(-1) == 1, probability_mask.view(-1) == 1)
        
        active_logits = logits_tensor[active_indices]
        active_labels = labels_tensor[active_indices.cpu()]
        
        active_labels = active_labels.cuda()
        
        # Local Loss
        loss_local = F.cross_entropy(active_logits, active_labels)
        
        # Global Loss
        valid_labels = labels * answer_mask # shape: (batch_size, 2048)
        nonzero_gt_lens = (valid_labels != 0).sum(dim=1) # shape: (batch_size)
        
        softmax_logtis = F.softmax(logits, dim=2) # shape: (batch_size, 2048, num_labels)
        pred_labels = softmax_logtis.argmax(dim=2) # shape: (batch_size, 2048)
        valid_pred_labels = pred_labels * answer_mask # shape: (batch_size, 2048)
        nonzero_pred_sum = valid_pred_labels.sum(dim=1) # shape: (batch_size)
        
        pred_global_label = nonzero_pred_sum/nonzero_gt_lens # shape: (batch_size)
        
        loss_global = F.mse_loss(pred_global_label, sentence_reward) # TODO: check the loss function
        
        loss =  self.extra_config.local_weight * loss_local + self.extra_config.global_weight * loss_global
        
        wandb.log({"loss_local": loss_local.item(), "loss_global": loss_global.item()})
        
        print(loss_local, loss_global)

        logits_dict = {
            'logits': active_logits, 
            'labels': active_labels
        }
        
        if return_outputs:
            return loss, logits_dict
        else:
            return loss
        
    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()

        logits = logits_dict['logits']
        logits = nested_detach(logits)
        
        labels = logits_dict['labels']

        return loss, logits, labels
    
    def visualize_samples(self, num_print_samples: int):
        """
        Visualize the reward model logits prediction

        Args:
            num_print_samples (`int`, defaults to `4`):
                The number of samples to print. Set to `-1` to print all samples.
        """
        # TODO
        eval_dataloader = self.get_eval_dataloader()
        table = defaultdict(list)
        for _, inputs in enumerate(eval_dataloader):
            # breakpoint()
            # _, logits, labels = self.prediction_step(self.model, inputs, prediction_loss_only=False)
            text = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
            sentence_reward = inputs["sentence_reward"]
            table["text"].extend(gather_object(text))
            table["sentence_reward"].extend(gather_object(sentence_reward).tolist())
            # table["logits"].extend(
            #     gather_object([[round(inner_item, 4) for inner_item in item] for item in logits.tolist()])
            # )
            if num_print_samples >= 0 and len(table["text"]) >= num_print_samples:
                break
        df = pd.DataFrame(table)
        print_rich_table(pd.DataFrame(table))
        if self.accelerator.process_index == 0:
            print_rich_table(df[:num_print_samples])
            if "wandb" in self.args.report_to:
                import wandb
                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

def compute_metrics(pred):
    logits, labels = pred
    logits = torch.from_numpy(logits)
    labels = torch.from_numpy(labels)
    
    # do softmax
    logits = F.softmax(logits, dim=1)
    
    preds = torch.argmax(logits, dim=1)
    
    # TODO: 分别统计preds和labels的分布,并输出
    preds_count = torch.bincount(preds)
    labels_count = torch.bincount(labels)
    
    print("Predictions distribution:")
    for i, count in enumerate(preds_count):
        print(f"Class {i}: {count.item()}")
    
    print("\nLabels distribution:")
    for i, count in enumerate(labels_count):
        print(f"Class {i}: {count.item()}")

    num_classes = logits.shape[1]
    auc_scores = []
    weights = []
    
    for class_idx in range(num_classes):
        y_true_binary = (labels == class_idx).int()
        y_score_binary = logits[:, class_idx]
        
        if len(torch.unique(y_true_binary)) == 1:
            auc = 0.5  # If only one class is present, assign AUC of 0.5
        else:
            auc = roc_auc_score(y_true_binary.numpy().flatten(), y_score_binary.numpy().flatten())
        
        auc_scores.append(auc)
        weights.append(y_true_binary.sum().item())
    
    weights = np.array(weights)
    weighted_auc = np.sum(np.array(auc_scores) * weights) / np.sum(weights)

    # Calculate overall accuracy
    labels = labels.flatten()
    preds = preds.flatten()

    acc = accuracy_score(labels, preds)
    print(labels.shape, preds.shape)
    
    # Calculate precision, recall, and f1-score for each class
    precision = precision_score(labels, preds, average=None)  # Returns an array of precision scores for each class
    recall = recall_score(labels, preds, average=None)  # Returns an array of recall scores for each class
    f1 = f1_score(labels, preds, average=None)  # Returns an array of f1 scores for each class

    save_path = os.path.join(reward_config.output_dir, 'metrics.json')  # Assuming output_dir is defined elsewhere or use a specific path
    
    num_classes = logits.shape[1]
    
    with open(save_path, 'w') as f:
        f.write(f'weighted_auc: {weighted_auc}\n')
        f.write(f'accuracy: {acc}\n')
        for i in range(num_classes):
            f.write(f'precision (class {i}): {precision[i]}\n')
            f.write(f'recall (class {i}): {recall[i]}\n')
            f.write(f'f1 (class {i}): {f1[i]}\n')
        for i in range(len(preds)):
            f.write(f'pred: {preds[i]}, gt: {labels[i]}\n')
    
    print('Weighted AUC:', weighted_auc)
    print('Accuracy:', acc)
    print(f'Precision (Class 0-{num_classes-1}):', precision)
    print(f'Recall (Class 0-{num_classes-1}):', recall)
    print(f'F1-Score (Class 0-{num_classes-1}):', f1)

    metrics_dict = {'weighted_auc': weighted_auc, 'accuracy': acc}
    for i in range(num_classes):
        metrics_dict[f'precision_class_{i}'] = precision[i]
        metrics_dict[f'recall_class_{i}'] = recall[i]
        metrics_dict[f'f1_class_{i}'] = f1[i]

    return metrics_dict
        
if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, ModelConfig, ExtraConfig))
    reward_config, model_config, extra_config = parser.parse_args_into_dataclasses()
    reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    
    wandb.init(project="token_rm", name=reward_config.output_dir)
        
    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(
        model_config
    )
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    tokenizer_name = extra_config.tokenizer_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name
    )
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=extra_config.lora_yc_r,
        lora_alpha=extra_config.lora_yc_alpha,
        lora_dropout=extra_config.lora_yc_dropout,
        target_modules="all-linear",
    )
    
    config = AutoConfig.from_pretrained(model_config.model_name_or_path)
    config.num_labels=extra_config.num_class_labels
    config.problem_type="multi_label_classification"
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_config.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True,
        **model_kwargs
    )

    # print(model)
    
    if extra_config.use_lora:        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )

    ################
    # Dataset
    ################
    # raw_datasets = load_dataset("json", data_files={'train': extra_config.train_dataset_path, 'test': extra_config.test_dataset_path})
    train_dataset = YCDataset(extra_config.train_dataset_path).hf_dataset
    eval_dataset = YCDataset(extra_config.test_dataset_path).hf_dataset
    
    def print_distribute(valid_rewards, tokens, token_start_index, words, word_rewards, answer_tokens):
        # sum_rewards = sum(valid_rewards)

        # TODO: 统计valid_rewards的分布
        reward_distribution = Counter(valid_rewards)
            
        # 检查分数0的占比是否大于分数1和2的占比之和
        # total_valid_rewards = len(valid_rewards)
        zero_count = reward_distribution[0]
        one_count = reward_distribution[1]
        two_count = reward_distribution[2]
        
        too_much_zero = False
    
        if zero_count > one_count + two_count:
            # zero_percentage = zero_count / total_valid_rewards * 100
            # one_percentage = one_count / total_valid_rewards * 100
            # two_percentage = two_count / total_valid_rewards * 100
 
            # print(words)
            # print(word_rewards)
            # print(answer_tokens)
            # print(valid_rewards)
                       
            # print(f'Score 0 percentage: {zero_percentage:.2f}%')
            # print(f'Score 1 percentage: {one_percentage:.2f}%')
            # print(f'Score 2 percentage: {two_percentage:.2f}%')
            
            # print('assign too much zero')
            
            too_much_zero = True
            
        return too_much_zero
    
    def assign_token_rewards(tokens, words, word_rewards):
        token_start_index = tokens.index('Answer') + 2
        token_rewards = [0] * token_start_index
    
        answer_tokens = tokens[token_start_index:]
        
        answer_mask = [0] * token_start_index + [1] * len(answer_tokens)
        
        word_matched = [0] * len(words)
        
        for i in range(len(answer_tokens)):
            answer_tokens[i] = answer_tokens[i].replace('Ġ', '').strip()

        # TODO: 将words和answertokens都转化成小写
        words = [word.lower() for word in words]
        answer_tokens = [token.lower() for token in answer_tokens]
        
        token_index = 0
        while token_index < len(answer_tokens):
            flag_matched = False
            i_matched = 0
            wid_matched = -1
            for i in range(1, min(6, len(answer_tokens) - token_index)):
                sublist = answer_tokens[token_index:token_index + i]
                concatenated_tokens = ''.join(sublist)
                
                for word_index, current_word in enumerate(words):     
                            
                    if concatenated_tokens == current_word:
                        wid_matched = word_index
                        flag_matched = True
                        i_matched = i
                                                
                        if word_matched[word_index] == 0:
                            word_matched[word_index] = 1
                            break
                
                if flag_matched:
                    break
                    
            if flag_matched:
                token_rewards.extend([word_rewards[wid_matched]] * i)
                token_index += i_matched
                flag_matched = False
            else:
                token_rewards.append(0)
                token_index += 1
 
        answer_rewards = token_rewards[token_start_index:]
               
        # TODO: get probability mask
        # 1. 统计answer_rewards中各个分数的分布和位置(分数值可能是：0,1,2)
        score_positions = {0: [], 1: [], 2: []}
        for i, score in enumerate(answer_rewards):
            score_positions[score].append(i)
            
        # 2. 使用probability_mask随机mask掉一部分位置,使得分数1和2的分布比例为1:3到3:1之间
        score1_count = len(score_positions[1])
        score2_count = len(score_positions[2])
        total_count = score1_count + score2_count
        
        if total_count == 0:
            print('total_count is 0!')
            print('------------------------------------------------------')
            print(total_count)
            print(words)
            print(word_rewards)
            print(answer_tokens)
            print(answer_rewards)     
            too_much_zero = True
            return token_rewards, answer_mask, answer_mask, too_much_zero           
        
        score1_ratio = score1_count / total_count
        score2_ratio = score2_count / total_count       
    
        if score1_ratio <= 1/4:
            score1_keep_ratio = score1_ratio
            score2_keep_ratio = 3 * score1_ratio
        elif score1_ratio > 1/4 and score1_ratio <= 3/4:
            score1_keep_ratio = score1_ratio
            score2_keep_ratio = score2_ratio
        else: 
            score1_keep_ratio = 3 * score2_ratio
            score2_keep_ratio = score2_ratio
            
        # 计算分数1和分数2要保留的数量
        score1_keep_count = int(total_count * score1_keep_ratio)
        score2_keep_count = int(total_count * score2_keep_ratio)
        
        # 随机选择要保留的位置
        score1_keep_positions = random.sample(score_positions[1], score1_keep_count)
        score2_keep_positions = random.sample(score_positions[2], score2_keep_count)
        
        # 创建probability_mask
        probability_mask = [0] * len(answer_rewards)
        for pos in score1_keep_positions + score2_keep_positions:
            probability_mask[pos] = 1
            
        # 3. 保证分数0的位置不被mask掉
        for pos in score_positions[0]:
            probability_mask[pos] = 1
            
        # TODO: extent probability_mask to the whole tokens
        probability_mask = [0] * token_start_index + probability_mask
        
        # TODO: 取token_rewards中，被answer mask和probability mask保留的位置的值作为valid_rewards
        valid_rewards = [token_rewards[i] for i in range(len(token_rewards)) if (answer_mask[i] == 1 and probability_mask[i] == 1)]
                
        # get token rewards distribution
        too_much_zero = print_distribute(valid_rewards, tokens, token_start_index, words, word_rewards, answer_tokens)  
                
        return token_rewards, answer_mask, probability_mask, too_much_zero
        
    def preprocess_function(examples):
        new_examples = {}

        question = examples["prompt"]
        answer = examples["answer"]
        words = examples["words"]
        rewards = examples["rewards"]
        sentence_reward = examples["sentence_reward"]
        
        inp = f"Question: {question}\n\nAnswer: {answer}"
        tokenized_inp = tokenizer(inp)
        tokens = tokenizer.tokenize(inp)
                
        labels, answer_mask, probability_mask, too_much_zero = assign_token_rewards(tokens, words, rewards)
            
        labels = [0] + labels + [0]
        answer_mask = [0] + answer_mask + [0]
        probability_mask = [0] + probability_mask + [0]
        
        new_examples["input_ids"] = tokenized_inp["input_ids"]
        new_examples["attention_mask"] = tokenized_inp["attention_mask"]
        new_examples["words"] = words
        new_examples["labels"] = labels
        new_examples["answer_mask"] = answer_mask
        new_examples["probability_mask"] = probability_mask
        new_examples["sentence_reward"] = sentence_reward
        new_examples["too_much_zero"] = too_much_zero
        
        if len(labels) != len(tokenized_inp["input_ids"]):
            raise ValueError(f"Length mismatch between labels and input_ids: {len(labels)} vs {len(tokenized_inp['input_ids'])}")

        if len(answer_mask) != len(tokenized_inp["input_ids"]):
            raise ValueError(f"Length mismatch between answer_mask and input_ids: {len(answer_mask)} vs {len(tokenized_inp['input_ids'])}")
        return new_examples

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    train_dataset = train_dataset.map(
        preprocess_function,
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids"]) <= reward_config.max_length
    )
    print(len(train_dataset))
    # filer out examples that have too much zero rewards
    
    # breakpoint()
    train_dataset = train_dataset.filter(
        lambda x: not x['too_much_zero']
    )
    
    print(len(train_dataset))

    eval_dataset = eval_dataset.map(
        preprocess_function,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids"]) <= reward_config.max_length
    )
    eval_dataset = eval_dataset.filter(
        lambda x: not x['too_much_zero']
    )

    print("train_dataset length:", len(train_dataset))
    print("eval_dataset length:", len(eval_dataset))

    ################
    # Training
    ################
    trainer = MyTrainer(
        extra_config=extra_config,
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=reward_config.max_length),
        compute_metrics=compute_metrics
    )

    trainer.train()
    print("Saving last checkpoint of the model")
    trainer.save_model(reward_config.output_dir)

    trainer.evaluate()