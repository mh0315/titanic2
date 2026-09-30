# pytorch

线性模型：$\hat{y}=w*x+b$

MSE 损失函数：$loss=(\hat{y}-y)^2$

`for w in np.arange()`每次采样范围和步长

梯度下降算法 :$ \theta=\theta - \eta \nabla_\theta J(\theta)$

可能陷入鞍点

学习率不要太大或太小

常用随机梯度下降，即用单样本损失函数求导更新数据，但参数间存在依赖

在深度学习中通常取折中的方法，即批量的随机梯度下降

`Dataset`用来获得数据及其标签，`Dataloader`为后面的网络提供不同的数据形式

`_init_`用来存入数据，`_len_`用来返回数据总数，`_getitem_`用来获取单条数据

Tensor 用来存储数据，data用来保存权重值，grad用来保存损失函数对权重的导数

使用Tensor 时每轮结束要使用`w.grad.data.zero`对权重进行清零

使用`class`定义模型，从`nn.Module`中继承

需要构造`_init_`函数，`super(Model,self)._init`不能没有

`self.linear = torch.nn.Linear()`可以用来构造一个对象，能直接完成权重乘以输入加上偏置的计算

BCE 损失：$
L(y,\hat{y})
=
-\bigl[y\log \hat{y} + (1-y)\log(1-\hat{y})\bigr]
$

`torch.nn.functional`引入函数

`torch.save（）`可以用来保存模型,模型名称后加`state_dict()`

可以只保存参数

