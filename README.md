#### Tweet Sentiment Extraction

---
- 本项目是对kaggle中tweet sentiment extraction的变体
    - https://www.kaggle.com/c/tweet-sentiment-extraction
- 原题中要求通过语句和情感对关键词进行提取，项目仅通过原句对关键词进行提取
- 从youtube上的一些教程得到了些许经验

---
#### 项目介绍
- input
    - 项目的输入文件以及一个bert-base-tiny-uncased，我是用的是tiny bert，参数比较少，项目中已经全部改为bert-base-uncased，如果想要使用tiny bert，可以直接修改项目
- model
    - 项目的模型输出
- src
    - 项目的各种文件

---
#### 运行
- 直接运行src/train.py
