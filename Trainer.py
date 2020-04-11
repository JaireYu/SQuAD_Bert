import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
from model import SQuAD_Bert
from utils import set_seed, compute_f1

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.config_class = BertConfig
        self.bert_config = self.config_class.from_pretrained(args.model_name_or_path)
        self.model = SQuAD_Bert(self.args, self.bert_config)
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)

        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        # n是参数的name: BERT_NAME: embeddings.word_embeddings.weight encoder.layer.5.output.LayerNorm.bias等
        # 下面这段代码的意思是，如果no_decay中的任何一个字段都不在name中则对para使用L2正则项, 否则默认设为0, 即bias相关的不带偏置项,
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        # 调度学习率在初期上升，后期下降(warm_up)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0             #总步数
        tr_loss = 0.0
        self.model.zero_grad()      # 清空梯度

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        set_seed(self.args)

        for _ in train_iterator:    # 一次遍历数据集
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):   # 取出一个batch: 原数据集是tuple(5 * Tensor(4478)) 所以一个batch是tuple(5 * Tensor(16))
                self.model.train()  # 告诉pytorch正在训练 而不是预测
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {
                    'input_ids' : batch[0],
                    'attention_mask' : batch[1],
                    'segment_ids' : batch[2],
                    'start_positions' : batch[3],
                    'end_positions' : batch[4]
                }
                outputs = self.model(**inputs)      #该语句自动执行forward, 与显式调用forward不同的是这个过程还会调用一些hooks
                loss = outputs[0]                   # 喂入的是一个batch, loss应该是一个batch的平均值

                if self.args.gradient_accumulation_steps > 1:       #取一个step的平均loss
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0: # 一个step结束, 需要更新参数
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    #梯度截断
                    optimizer.step()    #一个loss的积累过程结束，更新参数
                    scheduler.step()  # Update learning rate schedule, 更新学习率
                    self.model.zero_grad()  #清空梯度
                    global_step += 1

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:    # 200步save model
                        self.save_model()
            self.evaluate()
        return global_step, tr_loss / global_step

    def evaluate(self):
        eval_sampler = SequentialSampler(self.dev_dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        start_real_pos= None
        end_real_pos = None
        start_preds = None
        end_preds = None
        all_p_tag = None
        all_cls_index = None
        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'segment_ids': batch[2],
                    'start_positions': batch[3],
                    'end_positions': batch[4]
                }
                tmp_eval_loss, start_probs, end_probs = self.model(**inputs)

                eval_loss += tmp_eval_loss.mean().item()    # 对batch内的loss取平均(因为已经取过平均了，所以并不是有必要加mean
            nb_eval_steps += 1

            if all_p_tag is None:
                all_p_tag = batch[6].detach().cpu().numpy()
            else:
                all_p_tag = np.append(all_p_tag, batch[3].detach().cpu().numpy())

            if all_cls_index is None:
                all_cls_index = batch[5].detach().cpu().numpy()
            else:
                all_cls_index = np.append(all_cls_index, batch[4].detach().cpu().numpy())

            # Start prediction
            if start_probs is None:
                start_preds = start_probs.detach().cpu().numpy() #intent输出转化成numpy()
                start_real_pos = inputs['start_positions'].detach().cpu().numpy()
            else:
                start_preds = np.append(start_preds, start_probs.detach().cpu().numpy(), axis=0) # np.append()是拼接两个nparray的操作
                start_real_pos = np.append(
                    start_real_pos, inputs['start_positions'].detach().cpu().numpy(), axis=0)

            # End prediction
            if end_probs is None:
                end_preds = end_probs.detach().cpu().numpy() #intent输出转化成numpy()
                end_real_pos = inputs['end_positions'].detach().cpu().numpy()
            else:
                end_preds = np.append(end_preds, end_probs.detach().cpu().numpy(), axis=0) # np.append()是拼接两个nparray的操作
                end_real_pos = np.append(
                    end_real_pos, inputs['end_positions'].detach().cpu().numpy(), axis=0)


        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }
        # actual_match_preds 是[start, end]构成的array
        actual_real_match_span = np.append(start_real_pos.reshape(-1,1), end_real_pos.reshape(-1,1), axis=1)

        #这里已经求出[[score*386][score*386],...] [[pmask*386],....]和[real_pos...]
        actual_match_preds, all_pos_num = self.Get_Max_Score_Index(start_preds, end_preds, all_p_tag, all_cls_index, self.args.null_score_diff_threshold)

        EM_score = (actual_real_match_span == actual_match_preds).mean()
        F1_score = compute_f1(actual_match_preds, actual_real_match_span, all_pos_num)
        total_result = {
            'EM' : EM_score,
            'F1' : F1_score
        }
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results


    @classmethod
    def Get_Max_Score_Index(cls, start_real_pos, end_real_pos, start_preds, end_preds, all_p_tag, all_cls_index, null_score_diff_threshold):
        actual_match_span = np.empty(shape=[0,2])
        all_pos_num = np.array([])
        for start_pred, end_preds, p_tag, cls_index, null_threshold in zip(start_preds, end_preds, all_p_tag, all_cls_index, null_score_diff_threshold):
            scores_dict = {}
            scores_dict[tuple([0,0])] = start_preds[cls_index] + end_preds[cls_index] + null_score_diff_threshold
            for i, start_score in enumerate(start_pred):
                for j, end_score in enumerate(end_pred):
                    if i != cls_index and j != cls_index and p_tag[i] == 0 and p_tag[j] == 0:
                        scores_dict[tuple([i,j])] = start_score + end_score
            max_pos_tuple = max(scores_dict, key = scores_dict.get)
            actual_match_span = np.append(actual_match_span, [[max_pos_tuple[0], max_pos_tuple[1]]], axis=0)
            all_pos_num = np.append(all_pos_num, [np.sum(p_tag == 0)])
        return start_pos_pred, end_pos_pred, actual_match_span

    def save_model(self):
        # Save model checkpoint (Overwrite)
        output_dir = os.path.join(self.args.output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model #意思是只加载model本身
        model_to_save.save_pretrained(output_dir)       # save模型
        torch.save(self.args, os.path.join(output_dir, 'training_config.bin'))  #save_trainingconfig
        logger.info("Saving model checkpoint to %s", output_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.bert_config = self.config_class.from_pretrained(self.args.output_dir)   # 在文件夹中自动加载bert的配置文件
            logger.info("***** Config loaded *****")
            self.model = SQuAD_Bert.from_pretrained(self.args.output_dir, config=self.bert_config,
                                                          args=self.args)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")