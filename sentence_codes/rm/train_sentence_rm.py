import os
from transformers import AutoConfig
import warnings
from transformers.trainer_pt_utils import nested_detach

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
)
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import PreTrainedModel
from peft import LoraConfig, TaskType, get_peft_model
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import numpy as np
import torch.nn.functional as F
import wandb
import random

from accelerate.utils import gather_object
from trl.trainer.utils import RewardDataCollatorWithPadding, compute_accuracy, print_rich_table
from collections import defaultdict
import pandas as pd

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
    train_dataset_path: str = field(default='dataset_for_rm/train_dataset_4v1.json')
    test_dataset_path: str = field(default='dataset_for_rm/eval_dataset_300.json')
    tokenizer_name: str = field(default="allenai/longformer-base-4096")
    num_class_labels: int = field(default=2)
    pos_neg_ratio: float = field(default=3.0)
    class_weight: float = field(default=4.0)
    lora_yc_r: int = field(default=32)
    lora_yc_alpha: int = field(default=64)
    lora_yc_dropout: float = field(default=0.1)
    use_lora: bool = field(default=True)
    

@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_process = []
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
                    "attention_mask": feature["attention_mask"],
                    "labels": feature["labels"],
                }
            )
        batch = self.tokenizer.pad(
            features_process,
            padding='max_length',
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"]
        }
        return batch
    
class MyTrainer(RewardTrainer):
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
            
        outputs = model(**inputs)
        logits = outputs.logits
        
        if extra_config.class_weight!=0:
            # 计算类别权重
            class_weights = self.compute_class_weights(labels, extra_config.class_weight)

            # 使用带有类别权重的二元交叉熵损失
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
            num_classes = 2
            labels_onehot = F.one_hot(labels, num_classes).float()
            loss = loss_fn(logits, labels_onehot)
        else:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
                
        logits_dict = {'logits': logits, 'labels': labels}
        
        if return_outputs:
            return loss, logits_dict
        else:
            return loss
        
    def compute_class_weights(self, labels, class_weight):
        # 计算每个类别的样本数量
        class_counts = torch.bincount(labels)
        if class_counts.numel() == 1:
            class_counts = torch.cat([class_counts, torch.tensor([0]).to(class_counts.device)])
        # print(labels, class_counts)
        
        # 计算类别权重
        total_samples = labels.numel()
        neg_weight = total_samples / (class_weight * class_counts[0])  # 负样本权重
        pos_weight = total_samples / (class_weight * class_counts[1])  # 正样本权重

        # print(neg_weight, pos_weight)
        
        return torch.tensor([neg_weight, pos_weight])  # 返回负样本和正样本的权重
        
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
            sentence_reward = inputs["labels"]
            table["text"].extend(gather_object(text))
            table["labels"].extend(gather_object(sentence_reward).tolist())
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
    
    preds = torch.argmax(logits, dim=1)

    # Calculate AUC
    auc = roc_auc_score(labels, logits[:, 1])  # Assumes the positive class is labeled as 1

    # Calculate overall accuracy
    acc = accuracy_score(labels, preds)
    
    # Calculate precision, recall, and f1-score for each class
    precision = precision_score(labels, preds, average=None)  # Returns an array of precision scores for each class
    recall = recall_score(labels, preds, average=None)  # Returns an array of recall scores for each class
    f1 = f1_score(labels, preds, average=None)  # Returns an array of f1 scores for each class

    save_path = os.path.join(reward_config.output_dir, 'metrics.json')  # Assuming output_dir is defined elsewhere or use a specific path
    with open(save_path, 'w') as f:
        f.write(f'auc: {auc}\n')
        f.write(f'accuracy: {acc}\n')
        f.write(f'precision (class 0, class 1): {precision[0]}, {precision[1]}\n')
        f.write(f'recall (class 0, class 1): {recall[0]}, {recall[1]}\n')
        f.write(f'f1 (class 0, class 1): {f1[0]}, {f1[1]}\n')
        for i in range(len(preds)):
            f.write(f'pred: {preds[i]}, gt: {labels[i]}\n')
    
    print('AUC:', auc)
    print('Accuracy:', acc)
    print('Precision (Class 0, Class 1):', precision)
    print('Recall (Class 0, Class 1):', recall)
    print('F1-Score (Class 0, Class 1):', f1)

    return {
        'auc': auc,
        'accuracy': acc,
        'precision_class_0': precision[0], 'precision_class_1': precision[1],
        'recall_class_0': recall[0], 'recall_class_1': recall[1],
        'f1_class_0': f1[0], 'f1_class_1': f1[1]
    }
         
if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, ModelConfig, ExtraConfig))
    reward_config, model_config, extra_config = parser.parse_args_into_dataclasses()
    reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    
    if not os.path.exists(reward_config.output_dir):
        os.makedirs(reward_config.output_dir)
    
    wandb.init(project="sentence_rm", name=reward_config.output_dir)
        
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
    
    # 修改配置文件
    config = AutoConfig.from_pretrained(model_config.model_name_or_path)
    # config.max_position_embeddings = reward_config.max_length
    config.num_labels=2
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True,
        **model_kwargs
    )

    print(model)
    
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
    raw_datasets = load_dataset("json", data_files={'train': extra_config.train_dataset_path, 'test': extra_config.test_dataset_path})

    def preprocess_function(examples):
        new_examples = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for question, answer, sentence_reward in zip(examples["prompt"], examples["answer"], examples["sentence_reward"]):
            inp = f"Question: {question}\n\nAnswer: {answer}"
            tokenized_inp = tokenizer(inp)
            label = int(sentence_reward)

            new_examples["input_ids"].append(tokenized_inp["input_ids"])
            new_examples["attention_mask"].append(tokenized_inp["attention_mask"])
            new_examples["labels"].append(label)

        return new_examples

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )

    # breakpoint()
    raw_datasets = raw_datasets.filter(
        lambda x: len(x["input_ids"]) <= reward_config.max_length
    )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    ################
    # Training
    ################
    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=reward_config.max_length),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Saving last checkpoint of the model")
    trainer.save_model(reward_config.output_dir)

    trainer.evaluate()
    predictions = trainer.predict(eval_dataset)
    logits = predictions.predictions
    logits = torch.from_numpy(logits)

    predicted_labels = torch.argmax(logits, dim=1)
    real_labels = torch.tensor(eval_dataset['labels'])

    len_infer = len(predicted_labels)

    save_path = os.path.join(reward_config.output_dir, 'predictions.json')
    with open(save_path, 'w') as f:
        for i in range(len_infer):
            f.write('pred: '+str(predicted_labels[i].item())+' logits:')
            f.write(str(logits[i])+'\n')
            f.write('real: '+str(real_labels[i].item())+'\n')