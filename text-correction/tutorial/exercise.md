# 练习

收集制作一个文本纠错的习题库。可以包括一些面试的八股和实际应用的问题。

## 习题部分
1. 传统文本纠错一般分为哪几个步骤
    - [ ] 查找，检测，纠正
    - [x] 检测，纠正
    - [ ] 查找，检测，计算，纠正
    - [ ] 查找，检测，纠正，排序

> 解释说明：传统的文本纠错一般分为两大步骤：
>    1. 检测(detect) 
>    2. 纠正(correct)
>
> 如果再进行细分，可以分为：错误检测，候选召回，候选排序，候选筛选等具体步骤。
> 参考：[pycorrector]()

---
2. 下面哪些数据集是用来做文本纠错的
    - [x] sighan2015
    - [x] MuCGEC
    - [ ] CLUE
    - [ ] MNIST
> 解释说明：
>   1. sighan2015:sighan是由台湾师范大学，台湾大学收集的，目前最常用的中文纠错数据集。分为2013，2014，2015，3个版本。 [https://aclanthology.org/W15-3106/](https://aclanthology.org/W15-3106/) 
>   2. MuCGEC: MuCGEC是由苏州大学与阿里巴巴收集的，中文语法纠错数据集。[https://aclanthology.org/2022.naacl-main.227/](https://aclanthology.org/2022.naacl-main.227/)
>   3. CLUE: 中文语言理解测评基准(CLUE)[https://cluebenchmarks.com/](https://cluebenchmarks.com/)
>   4. MNIST: 著名的手写识别数据集。[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
---

3. 常见的文本错误原因有哪些
    - [x] asr转文本时出现错误
    - [x] 键盘或手写输入法出错
    - [ ] 用脸滚键盘产生错误
    - [ ] 猩猩敲键盘产生错误
> 解释说明：
---

4. 常见的错误类型属于谐音/混淆音的是：
    - [x] 王者荣耀 - 亡者农药
    - [ ] 2023年5月 - 2O23年5刀
    - [ ] 伍迪艾伦 - 艾伦伍迪
    - [ ] 高粱 - 高梁
> 解释说明：
---

5. 中文文本纠错有哪些方向
    - [x] 语法纠错
    - [ ] 拼音纠错
    - [x] 拼写纠错
    - [ ] 手写纠错
> 解释说明：
---

6. 下面哪个模型是可以不进行文本对齐的
    - [ ] FASpell
    - [ ] BART-csc
    - [ ] Soft-Masked BERT
    - [x] T5-csc
> 解释说明：
--- 
**TODO**

