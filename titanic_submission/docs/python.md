数据处理步骤：构造新特征 > 划分数据集 > 处理缺失值 > 删除无用特征 > 区别类别特征和数值特征 > 转换类别特征 > 数值特征标准化 > 合并两类数组

# numpy

引入numpy：`impot numpy as np`

定义矩阵：`a = np.array[]`

后面用`.reshape`定义形状

区间也可以`a = np.arange()`

矩阵乘法`a.dot(b)`

平均值`mean()`或者``average()``*记住前面要加np.*

中位数``median()``

累加`cumsum()`

累差`diff()`

输出非零值`nonzero()`

排序`sort()`*注意：逐行排序从左到右递增*

转置`transpose()`或者`a.T`

截断`clip()`

*在指令后面加`aixs=1\0`可以指定对行还是列计算*

可以通过`a [n] [n]\[n,n]`查看矩阵中的数字

`a.flatten()`能将矩阵转变为一维，遍历时用`a.flat`

数组上下合并可以用`vstack(a,b)`，左右合并用`hstack(a,b)`

可以通过`a[:,np.newaxis]`或者`a[np.newaxis,:]`在列和行上增加维度，可以通过这个将行矩阵和列矩阵互换

使用`concatenate((),axis=0/1)`可以指定多个矩阵合并并规定其合并维度

使用`split()`能够对矩阵进行分割，并指定分割形式

`vsplit() hsplit()`用于分割矩阵，用法和合并函数类似

`copy()`可以把一个变量的值赋给另一个变量，并且不产生关联

`str.`表示对后续字符串操作，`extract(r"")`从表达式中提取符合规则的部分，r表示原始字符串

`np.random.default_rng(seed)` 创建一个新的随机数生成器对象 。

`astype()`表示强制转换类型

如果传入的 `y` 是 pandas 的 Series 或列表，`np.asarray` 会把它转成 NumPy 数组。

`n_test = int(round(n * test_size))`用来计算测试集大小

 `.shuffle` 是 **NumPy 随机生成器对象的方法名**

`drop_cols =`用来删除不需要的列

`arr = np.zero(len(data), len(cat_cols))`创造一个初始全为零的二维数组，第一个参数必须是元组

# pandas

可以用`pd.Series([])`定义一个序列

`pd.date_range("",periods=a)`可以按照日期顺序给出序列

`pd.DataFrame()`可以用来按照规定生成数据表

`dtypes`给出数据类型

`index`给出列的名字，`columns`给出行的名字，`values`给出值的名字

`describe()`用来计算数据特征

`T`用来矩阵转置

`sort_index(axis=0/1,ascending=False)`可以用来给数据行或列排序，`False`表示倒序

`sort_value(by=)`可以对数据进行排序

`df.loc[]`可以根据标签选择数据，也可以用来改数据

`df.iloc[]`可以根据位置选择数据，也可以用来改数据

`df.dropna(axis=,how=any\all)`可以用来删除丢失数据行或列，`any`表示存在就删除，`all`表示全部都是才删除

`df.fillna()`用来填充数据

`df.isnull()`可以查看是否有缺失数据

`pd.read\to_csv()`可以用来导入\导出本地数据

`.values` 是 pandas Series 的一个属性，作用是：**把 Series 转成 NumPy 数组。**

`.drop(columns=drop_cols)`：pandas 的方法，表示**按列删除**。

`columns=drop_cols`：指定要删除的列名列表。

`columns=[]`：指定要做独热编码的列

`enumerate（）` 会给每个元素配上索引

`.mean()`可以用来计算均值。`.std()`可以用来计算标准差，后面加上`.replace(0,1)`可以防止除以0。

# Matplotlib

`.plot()`用来放入数据

`.figure()`可以给图表命名和控制规格

`.xlim()\.ylim（）`可以自定义取值范围，`.xlabel（）\.ylabel（）`可以定义标签

`.title`可以给图表命名，`.legend`可以用来图例

