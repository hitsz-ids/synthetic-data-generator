---
sidebar_position: 1
---

# 快速入门


## 单表数据快速合成示例

```python
# 导入相关模块
from sdg.tabular.synthesizers import CTGAN
from sdg.tabular.data import get_single_table
import pandas as pd

# 读取数据
data = get_single_table()
```
真实数据如下：
```
       age  workclass  fnlwgt  ... hours-per-week  native-country  label
0       27    Private  177119  ...             44   United-States  <=50K
1       27    Private  216481  ...             40   United-States  <=50K
2       25    Private  256263  ...             40   United-States  <=50K
3       46    Private  147640  ...             40   United-States  <=50K
4       45    Private  172822  ...             76   United-States   >50K
...    ...        ...     ...  ...            ...             ...    ...
32556   43  Local-gov   33331  ...             40   United-States   >50K
32557   44    Private   98466  ...             35   United-States  <=50K
32558   23    Private   45317  ...             40   United-States  <=50K
32559   45  Local-gov  215862  ...             45   United-States   >50K
32560   25    Private  186925  ...             48   United-States  <=50K

[32561 rows x 15 columns]

```
```python
#定义模型
model = CTGAN()

#训练模型
model.fit(data)

# 生成合成数据
sampled = model.generate(num_rows=10)
```

合成数据如下：
```
   age         workclass  fnlwgt  ... hours-per-week  native-country  label
0   33           Private  276389  ...             41   United-States   >50K
1   33  Self-emp-not-inc  296948  ...             54   United-States  <=50K
2   67       Without-pay  266913  ...             51        Columbia  <=50K
3   49           Private  423018  ...             41   United-States   >50K
4   22           Private  295325  ...             39   United-States   >50K
5   63           Private  234140  ...             65   United-States  <=50K
6   42           Private  243623  ...             52   United-States  <=50K
7   75           Private  247679  ...             41   United-States  <=50K
8   79           Private  332237  ...             41   United-States   >50K
9   28         State-gov  837932  ...             99   United-States  <=50K
```


## 多表数据快速合成示例

```python
# 导入相关模块
from sdg.tabular.synthesizers import CWAMT
from sdg.tabular.data import get_multi_table
import pandas as pd

# 读取数据
data = get_multi_table()
```
真实数据如下：

```
{'tables': {'table1': {'table_name': 'train', 'table_value':          Store  DayOfWeek        Date  ...  Promo  StateHoliday  SchoolHoliday
0            1          5  2015-07-31  ...      1             0              1
1            2          5  2015-07-31  ...      1             0              1
2            3          5  2015-07-31  ...      1             0              1
3            4          5  2015-07-31  ...      1             0              1
4            5          5  2015-07-31  ...      1             0              1
...        ...        ...         ...  ...    ...           ...            ...
1017204   1111          2  2013-01-01  ...      0             a              1
1017205   1112          2  2013-01-01  ...      0             a              1
1017206   1113          2  2013-01-01  ...      0             a              1
1017207   1114          2  2013-01-01  ...      0             a              1
1017208   1115          2  2013-01-01  ...      0             a              1

[1017209 rows x 9 columns]}, 'table2': {'table_name': 'store', 'table_value':       Store StoreType  ... Promo2SinceYear     PromoInterval
0         1         c  ...             NaN               NaN
1         2         a  ...          2010.0   Jan,Apr,Jul,Oct
2         3         a  ...          2011.0   Jan,Apr,Jul,Oct
3         4         c  ...             NaN               NaN
4         5         a  ...             NaN               NaN
...     ...       ...  ...             ...               ...
1110   1111         a  ...          2013.0   Jan,Apr,Jul,Oct
1111   1112         c  ...             NaN               NaN
1112   1113         a  ...             NaN               NaN
1113   1114         a  ...             NaN               NaN
1114   1115         d  ...          2012.0  Mar,Jun,Sept,Dec

[1115 rows x 10 columns]}}, 'relations': {'table1-table2': 'store'}}
```

```python
#定义模型
model = CWAMT()

#训练模型
model.fit(data)

# 生成合成数据
sampled = model.generate(num_rows=10)
```
合成数据如下：
```
{'table1': {'table_name': 'train', 'table_value':     Store  DayOfWeek        Date  ...  Promo  StateHoliday  SchoolHoliday
0    3          2  2013-01-01  ...      0             a              1
1    5          2  2013-01-01  ...      0             a              1
2    5          2  2013-01-01  ...      0             a              1
3    6          2  2013-01-01  ...      0             a              1
4    2          2  2013-01-01  ...      0             a              1
5    1          2  2013-01-01  ...      0             a              1
6    7          2  2013-01-01  ...      0             a              1
7    2          2  2013-01-01  ...      0             a              1
8    8          2  2013-01-01  ...      0             a              1
9    5          2  2013-01-01  ...      0             a              1
10   9          2  2013-01-01  ...      0             a              1
11   3          2  2013-01-01  ...      0             a              1
12   2          2  2013-01-01  ...      0             a              1
13   4          2  2013-01-01  ...      0             a              1
14   4          2  2013-01-01  ...      0             a              1
15   7          2  2013-01-01  ...      0             a              1
16   8          2  2013-01-01  ...      0             a              1
17   10         2  2013-01-01  ...      0             a              1
18   3          2  2013-01-01  ...      0             a              1
19   7          2  2013-01-01  ...      0             a              1

[20 rows x 9 columns]}, 'table2': {'table_name': 'store', 'table_value':    Store StoreType  ... Promo2SinceYear     PromoInterval
0   1         a  ...          2013.0   Jan,Apr,Jul,Oct
1   2         a  ...          2010.0   Jan,Apr,Jul,Oct
2   3         a  ...             NaN               NaN
3   4         c  ...          2012.0   Jan,Apr,Jul,Oct
4   5         c  ...             NaN               NaN
5   6         a  ...          2013.0   Jan,Apr,Jul,Oct
6   7         c  ...             NaN               NaN
7   8         a  ...             NaN               NaN
8   9         a  ...             NaN               NaN
9   10        d  ...          2012.0  Mar,Jun,Sept,Dec

[10 rows x 10 columns]}}
```


## API

除python组件之外，SDG还支持以Restful接口形式调用，具体接口参数请参考 [API文档](https://SDG.readthedocs.io/en/latest/api/index.html)。

