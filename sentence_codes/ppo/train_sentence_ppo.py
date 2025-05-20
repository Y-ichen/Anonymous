from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from transformers import get_scheduler

import wandb 
import re
from peft import PeftModel, PeftConfig

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

import os
import tyro
import json
from typing_extensions import Annotated
JSONDict = Annotated[Optional[dict], tyro.conf.arg(metavar="JSON", constructor=json.loads)]

tqdm.pandas()

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    model_name: Optional[str] = field(default="mistral-ckpt/checkpoint-4000", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="mistral-ckpt/checkpoint-4000", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="sentence-rm-ckpt/checkpoint-200/", metadata={"help": "the reward model name"})
    
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=5e-6, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="linear", metadata={"help": "the learning rate scheduler type"})
    output_max_length: Optional[int] = field(default=400, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.5,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=True, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=5, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="ckpt/sentence_ppo", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=42, metadata={"help": "the seed"})
    train_epochs: Optional[int] = field(default=2, metadata={"help": "number of epochs"})
    steps: Optional[int] = field(default=1200, metadata={"help": "number of steps"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )

    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    local_rank: Optional[int] = field(default=0, metadata={"help": "local rank"})
    
    project_name: Optional[str] = field(default="sentence_ppo", metadata={"help": "wandb project name"})
    data_file_path: Optional[str] = field(default="datasets/ppo/train_20k.json", metadata={"help": "data file path"})
    tracker_kwargs: Optional[str] = field(default=None, metadata={"help": "tracker kwargs of wandb"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

# 将 JSON 字符串转换成字典
if script_args.tracker_kwargs:
    tracker_kwargs_dict = json.loads(script_args.tracker_kwargs)
else:
    tracker_kwargs_dict = {}
    
reward_model_name = script_args.reward_model_name

if not os.path.exists(script_args.output_dir):
    os.makedirs(script_args.output_dir)

config = PPOConfig(
    steps=script_args.steps,
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
    tracker_kwargs=tracker_kwargs_dict,
    tracker_project_name=script_args.project_name,
)

# train_dataset = load_dataset("/home/aiscuser/ads_yc/rlaif/dataset", data_dir="ppo", split="train")
train_dataset = load_dataset("json", data_files=script_args.data_file_path, split="train")

sent_kwargs = {
    "top_k": None,
    "function_to_apply": "none",
    "batch_size": script_args.batch_size,
    "truncation": True,
}

tokenizer = AutoTokenizer.from_pretrained(
    script_args.tokenizer_name, 
    model_max_length=2048, 
    use_fast=True,
    )

if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

def filter_query_num_10(example):
    return example['query_nums'] == 10

def filter_non_english_answer(example):
    english_pattern = re.compile(r'^[a-zA-Z0-9\s\.,\?!]+$')
    return all(bool(english_pattern.match(ans.strip())) for ans in example['answer'].split(','))

def build_dataset(ds, tokenizer):
    original_columns = ds.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
            "query_nums": [],
            "answer": [],
        }
        for question, answer in zip(examples["sentence1"], examples["sentence2"]):
            # query = "Question: " + question + "\n\nAnswer: "
            query = question
            query_num = len(answer.split(","))
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])
            new_examples["query_nums"].append(query_num)
            new_examples["answer"].append(answer)

        return new_examples

    ds = ds.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    print("unfiltered len:", len(ds))
    ds = ds.filter(lambda x: len(x["input_ids"]) < 2000, batched=False)
    print("filtered len by 2000:", len(ds))
    ds = ds.filter(filter_query_num_10)
    print("filtered len by == 10:", len(ds))
    
    # 过滤掉answer字段中包含非英语内容的数据
    ds = ds.filter(filter_non_english_answer)
    print("filtered len by english answer:", len(ds))
    
    ds.set_format(type="torch")
    return ds

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(train_dataset, tokenizer)
# dataset = dataset.select(range(20000))
print("Dataset length:", len(dataset))

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index



lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
config.model_name,
# load_in_8bit=True,
# torch_dtype=torch.bfloat16,
peft_config=lora_config,
)
        
ref_model = None

# if script_args.adafactor:
optimizer = Adafactor(
    filter(lambda p: p.requires_grad, model.parameters()),
    scale_parameter=False,
    relative_step=False,
    warmup_init=False,
    lr=config.learning_rate,
)

lr_scheduler = get_scheduler(
    name=script_args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=600,
)

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
)

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug

reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name, padding="max_length", max_length = 2048, truncation=True)

reward_tokenizer.pad_token = reward_tokenizer.eos_token
    
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model_name,
    tokenizer=reward_tokenizer,
    # model_kwargs={"load_in_8bit": True},
    # model_kwargs={"torch_dtype": torch.float16},
    return_token_type_ids=False,
)

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

def post_process(texts):
    processed_texts = []
    for text in texts:
        # Remove specific long introductory texts
        text = text.replace("Display Ads Query Prediction Task: An ads search engine exists that generates interesting display ads to be shown to the user on third party web domains.  **Your task** is to predict **10** ads queries based on the user profile and history information that generates interesting display ads that the user is likely to click on and buy. You will receive 4 data sources: ", "")
        text = text.replace("Initially there will be an overall user profile Location:IN ; AgeGroup:Unknown ;  Gender:Unknown.  For the Display Ads Query Prediction Task, provide at most 10 short text search ads queries:", "")

        # Replace specified phrases with keywords
        text = text.replace("1) User search queries", "{\"search\":")
        text = text.replace("2) User visits UET dataset", "\"uet\":")
        text = text.replace("3) User visits edge dataset", "\"edge\":")
        text = text.replace("4) User purchase dataset", "\"purchase\":")
        text = text.replace(" \n\nAnswer", "}\n\nAnswer")
        text = text.replace("[]", "[\"\"]")

        # Replace all semicolons with commas
        text = text.replace(";", ",")

        # Remove all escape characters
        text = text.replace("\\", "")

        # Change single quotes to double quotes for entries within square brackets
        text = re.sub(r'\[([^\]]+)\]', lambda m: '[' + m.group(1).replace("'", "\"") + ']', text)

        # Remove content between "purchase" and "}\n\nAnswer"
        text = re.sub(r'(\"purchase\": \[.*?\]).*?(\}\n\nAnswer)', r'\1\2', text, flags=re.DOTALL)
        
        processed_texts.append(text)
    
    return processed_texts

for epoch in range(script_args.train_epochs):
    last_pipeouts = None
    last_question_tensors = None
    last_response_tensors = None
    for step, batch in tqdm(enumerate(ppo_trainer.dataloader), desc=f"Epoch {epoch} "):

        question_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(  
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # Compute reward score (using the sentiment analysis pipeline)
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]

        # TODO: reform texts
        input_texts = post_process(texts)

        try:
            pipe_outputs = sentiment_pipe(input_texts, **sent_kwargs)
            last_question_tensors = question_tensors
            last_response_tensors = response_tensors
            last_pipeouts = pipe_outputs
        except:
            print(f"Error at step {step} when computing rewards")
            continue

        rewards = []
        for output in pipe_outputs:
            score_pos = 0
            score_neg = 0
            for item in output:
                if item['label'] == 'LABEL_1':
                    score_pos = item['score']
                elif item['label'] == 'LABEL_0':
                    score_neg = item['score']
            logits = [score_neg, score_pos]

            logits = torch.tensor(logits)
            logits = torch.nn.functional.softmax(logits, dim=0)
            
            reward = logits[1] - script_args.reward_baseline
            # reward = (logits[1] - script_args.reward_baseline)*2
            rewards.append(reward)

        try:
            stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
        except:
            print(f"Error at step {step} when stepping")
            continue

        if step !=0 and step % 5 == 0:
            ppo_trainer.save_pretrained(script_args.output_dir + f"epoch_{step}_step_{step}")