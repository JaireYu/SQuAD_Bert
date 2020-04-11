## SQuAD_Bert模型
### model:
bert model + 两层线性层 + softmax
### data_process:
由于bert模型是定长的输入，想把query作为sentence_A, paragraph作为sentence_B, 就必须将其固定长度，处理方法如下：
* 固定query的长度，超过的做截断
* paragraph按照一定的长度分片，两片之间的起始位置差为doc_stride, 相邻的片之间有一定的重叠, 为了保证答案一定在某一片中完整出现
* [CLS] query [SEP] paragraph [SEP] optional_PADDING样式排布，作为输入，同时注意标记padding_token和CLS+para的token以便于预测和训练
* 没有完整出现完整的答案的片标记为[CLS_index, CLS_index] (即等同于impossible)， 但是这种概率比较低.
* 按照stride为128, 片的大小为317, 共得到13.2万条训练数据.
### train:
模型的结果是每个位置的start，end的评分，softmax之后取label对应的概率值，取-log再加上l2正则作为损失函数(用nn.CrossEntropyLoss和weight_decay实现)，优化这一函数来训练模型.

也有的研究表明start的权重应该高于end，所以默认采用了2:1的权重分配来计算loss
### predict_result:
求Score_{ij} = S_i + E_j(j > i, i,j != 0 代表have answer)

求Score_{00} = S_i + E_j(代表no answer)

预测结果时选择所有score中最大的一项的i, j作为结果

还有一些调整是区别no_answer和have_answer的情况，对score_00加上一个threshold.

### Evaluate
计划在Dev数据集上面做评价，用f1和EM指标评价结果，为了减小数据集，和官方的评价有所不同，所有的答案都采用第一条answer作为训练和评价数据，将每一个token是否属于answer_span视为一个二分类问题很容易得到f1的评分. EM则是完全匹配的情况
.

### Result
暂未训练和评估.