# loc vs iloc
- loc gets rows (or columns) with particular labels from the index.
- iloc gets rows (or columns) at particular positions in the index (so it only takes integers).
- ix usually tries to behave like loc but falls back to behaving like iloc if a label is not present in the index.
# 挑異常值 抓到anom
anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]

# 用nan將異常值取代
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

# 檢查貸款人車齡
plt.hist(app_train[~app_train.OWN_CAR_AGE.isnull()]['OWN_CAR_AGE'])
app_train[app_train['OWN_CAR_AGE'] > 50]['OWN_CAR_AGE'].value_counts()
＊上面是說先把own_car_age>50的挑起來 然後再帶回own_car_age數有多少


# 再把只有 2 值 (通常是 0,1) 的欄位去掉
numeric_columns = list(app_train[numeric_columns].columns[list(app_train[numeric_columns].apply(lambda x:len(x.unique())!=2 ))])


# 檢視這些欄位的數值範圍
for col in numeric_columns:
    app_train[col].plot.hist(title=col)
    plt.show()

# 離群值造成的直方圖看不清楚
# 選擇 OBS_60_CNT_SOCIAL_CIRCLE 小於 20 的資料點繪製

loc_a = app_train["OBS_60_CNT_SOCIAL_CIRCLE"]<20
loc_b = 'OBS_60_CNT_SOCIAL_CIRCLE'

app_train.loc[loc_a, loc_b].hist()
plt.show()

# 挑出非空值的row
app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY']

# 如果欄位中有 NA, describe 會有問題
app_train['AMT_ANNUITY'].describe()

# np.percentile()
a = np.array([[10, 7, 4], [3, 2, 1]])
np.percentile(a, 50) # 全部的50%分位數
np.percentile(a, 50, axis=0) # 縱列50% (10+3)/2=3.5
np.percentile(a, 50, axis=1) # 橫列50% 分別的50%分位數
np.percentile(a, 50, axis=1, keepdims=True) # 保持維度不變

* 後者為加上keepdims的結果	     array([ 7.,  2.]) vs array([[ 7.],[ 2.]])

# 計算四分位數
Ignore NA, 計算五值

five_num = [0, 25, 50, 75, 100]
quantile_5s = [np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'], q = i) for i in five_num]
print(quantile_5s)

# 利用np get median
np.median(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'])

# 計算眾數
from scipy.stats import mode
mode_get = mode(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'])

# 挑選空值用 中位數填補
app_train.loc[app_train['AMT_ANNUITY'].isnull(),'AMT_ANNUITY'] = q_50

# 數值的出現的次數
app_train['AMT_ANNUITY'].value_counts()
# 次數的index數值
value = list(app_train['AMT_ANNUITY'].value_counts().index)
value[0] ==> 基本上是眾數(可用這個眾數來填補)

# 沿縱軸合併
res = pd.concat([df1, df2, df3])

# 沿橫軸合併
result = pd.concat([df1, df4], axis = 1)
result
* 基本上這個方法 合併後其他的會留空
* 下面這個方法 可以用硬串接 基本上就是有值的才串接
result = pd.concat([df1, df4], axis = 1, join = 'inner') # 硬串接

# pd.merge() how=?
pd.merge(df1, df2, on='id', how='outer') 以id這欄做全合併
pd.merge(df1, df2, on='id', how='inner') 以id這欄做部分合併 (id match才做合併)

# 欄-列 逐一解開
df.melt()

# 篩選條件後,用loc
# 取 AMT_INCOME_TOTAL 大於平均資料中，SK_ID_CURR, TARGET 兩欄
sub_df = app_train.loc[app_train['AMT_INCOME_TOTAL'] > app_train['AMT_INCOME_TOTAL'].mean(), ['SK_ID_CURR', 'TARGET']]
sub_df.head()

# GroupBy用法
## groupby後看size
app_train.groupby(['NAME_CONTRACT_TYPE']).size()
## groupby後看他某一col的25%50%....etc
app_train.groupby(['NAME_CONTRACT_TYPE'])['AMT_INCOME_TOTAL'].describe()
## groupby後看他某col的mean
app_train.groupby(['NAME_CONTRACT_TYPE'])['TARGET'].mean()

# 教你怎麼取0:10000的指定欄位
app_train.loc[0:10000, ['NAME_CONTRACT_TYPE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']]

# cut()
##如果我們今天有一些連續性的數值，可以使用cut&qcut進行離散化 cut函数是利用數值區間將數值分類，qcut則是用分位數。換句話說，cut用在長度相等的類別，qcut用在大小相等的類別。
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = bin_cut)
## cut 可以用串列帶入,或是用np.linspace(start, stop, segment point)來製造區間

# random
## random seed
np.random.seed(1) 使得亂數可預測
## random int 
x = np.random.randint(0, 50, 1000) // 0 ~ 50 產生100個亂數
## 常態亂數
y = np.random.normal(0, 10, 1000)
## correlation x, y 相關係數
np.corrcoef(x, y)
## 畫散布圖
plt.scatter(x, y)

# corr做出來是陣列 怎麼辦
corr = np.corrcoef(sub_df['DAYS_EMPLOYED'] / (-365), sub_df['AMT_INCOME_TOTAL'])
print(corr)  -> print("Correlation: %.4f" % (corr[0][1])) // 只取小數點後四位，跟x,y的相關係數
x -> x		 x -> y
[[1.         0.01300472]
 [0.01300472 1.        ]]

# 如果直接畫散布圖 - 看不出任何趨勢或形態 ？
## 將y軸改成log-scale
np.log10(sub_df['AMT_INCOME_TOTAL'] )

# 看col類別
app_train[col].dtype

# 如果只有兩個值的類別欄位就做LE
## 種類 2 種以下的類別型欄位轉標籤編碼 (Label Encoding)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le_count = 0

## 檢查每一個 column
for col in app_train:
    if app_train[col].dtype == 'object':
        # 如果只有兩種值的類別型欄位
        if len(list(app_train[col].unique())) <= 2:
            # 就做 Label Encoder
            le.fit(app_train[col])
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            # 紀錄有多少個 columns 被標籤編碼過
            le_count += 1
 

# 相關係數
app_train.corr()['TARGET'] # 列出target與所有欄位的相關係數
ext_data_corrs = ext_data.corr()  # 若沒有指定哪一個欄位 則是變成全部相比

# sort some series
corr_vs_target.sort_values(ascending=False)

# data := <class 'pandas.core.series.Series'>
## 找小15
data.head(15)
## 找大15
data.tail(15)

# boxplot
## {dataset}.boxplot(column="y", by="x")
app_train.boxplot(column="EXT_SOURCE_3", by="TARGET")
plt.show()

# matplotlib theme
plt.style.use(‘default’) # 不需設定就會使⽤用預設
plt.style.use('ggplot')
plt.style.use(‘seaborn’)# 或採⽤用 seaborn 套件繪圖

# plot懶人包
## 改變繪圖樣式 (style)
plt.style.use('ggplot') 
## 改變樣式後再繪圖一次, 比較效果
plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)
plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');
plt.show()
## 設定繪圖區域的長與寬
plt.figure(figsize = (10, 8))

# kde by sns
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'Gaussian esti.', kernel='gau')
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'Cosine esti.', kernel='cos')
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'Triangular esti.', kernel='tri')
plt.show()

* app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365 這裡的意思是說 先loc出target=0的值對上DAYS_BIRTH再除以365

# 完整分布圖 (distplot) : 將 bar 與 Kde 同時呈現
sns.distplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')
plt.legend() // 顯示圖例
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
plt.show()

# EDA: 把連續型變數離散化
## 主要的⽅法:
等寬劃分：按照相同寬度將資料分成幾等份。缺點是受到異常值的影響比較⼤。
ex:
# 新增欄位 "equal_width_age", 對年齡做等寬劃分, 切成四等份
ages["equal_width_age"] = pd.cut(ages["age"], 4)

等頻劃分：將資料分成幾等份，每等份資料裡⾯的個數是一樣的。
ex:
# 新增欄位 "equal_freq_age", 對年齡做等頻劃分, 切成四等份
ages["equal_freq_age"] = pd.qcut(ages["age"], 4)

聚類劃分：使⽤用聚類演算法將資料聚成幾類，每⼀個類為⼀個劃分。

# sort_index
ex1:
ages["customized_age_grp"].value_counts().sort_values()
>>
(20, 30]     6
(50, 100]    3
(30, 50]     3
(10, 20]     2
(0, 10]      2
## 怎麼辦？？
ages["customized_age_grp"].value_counts().sort_index()


# subplot
plt.subplot(row,column,idx)
## plt.subplot 三碼如上所述, 分別表示 row總數, column總數, 本圖示第幾幅(idx)
plt.subplot(321)
plt.plot([0,1],[0,1], label = 'I am subplot1')
plt.legend()

# heatmap
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)

# kde 潤飾
# 依不同 EXT_SOURCE 逐項繪製 KDE 圖形
for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    # 做 subplot
    print(i, source) # 名稱加col_name
    plt.subplot(1, 3, i + 1)
    
    # KDE 圖形
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source], label = 'target == 0')
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source], label = 'target == 1')
    
    # 加上各式圖形標籤
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)

# tight_layout()
tight_layout() can take keyword arguments of pad, w_pad and h_pad. These control the extra padding around the figure border and between subplots. The pads are specified in fraction of fontsize.

# df.drop()
plot_data.drop(['DAYS_BIRTH'],axis=1, inplace=True)
* axis 一定要加 否則會報錯

# df.sample(n)
>>> df['num_legs'].sample(n=3, random_state=1) # 對df num_legs欄位抽三組

# df.dropna()
我想問的是如果下這個指令 如果一個欄位是空值就是全部刪掉嗎？


# 把 NaN 數值刪去, 並限制資料上限為 100000 : 因為要畫點圖, 如果點太多，會畫很久!
N_sample = 100000
plot_data = plot_data.dropna().sample(n = N_sample)

# N_sample = 100000
# 把 NaN 數值刪去, 並限制資料上限為 100000 : 因為要畫點圖, 如果點太多，會畫很久!
plot_data = plot_data.dropna().sample(n = N_sample))


# 建立 pairgrid 物件
grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False,
                    hue = 'TARGET', vars = [x for x in list(plot_data.columns) if x != 'TARGET'])
## 上半部為 scatter
grid.map_upper(plt.scatter, alpha = 0.2)
## 對角線畫 histogram
grid.map_diag(sns.kdeplot)
## 下半部放 density plot
grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r)
plt.suptitle('Ext Source and Age Features Pairs Plot', size = 32, y = 1.05) # 大標
plt.show()

# np.random.random((row, col))
區間：[0.0, 1.0)
## 那如果要取 -1.0 ~ 1.0 呢?
可以利用簡單數學
2 * np.random.random() -1 

# 將train_data & test_data 欄位改成一致
##調整欄位數, 移除出現在 training data 而沒有出現 testing data 中的欄位
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

# 填補器設定缺失值補中位數
imputer = Imputer(strategy = 'median')

# 縮放器設定特徵縮放到 0~1 區間
scaler = MinMaxScaler(feature_range = (0, 1))
#    x - min
# ------------
# max - min

# 填補器載入各欄中位數
imputer.fit(train)
# 將中位數回填 train, test 資料中的空缺值
train = imputer.transform(train)
test = imputer.transform(app_test)
# 縮放器載入 train 的上下限, 對 train, test 進行縮放轉換
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

# df.to_csv()
submit.to_csv("submission.csv", sep='\t')

# 特徵工程
從事實到對應分數的轉換，我們稱為特徵⼯程

# np.log1p() 數據平滑處理
train_Y = np.log1p(df_train['SalePrice'])
## pred = np.expm1(pred) # log1p()的反函數


# 特徵工程簡化版
LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()
for col in df.columns:
    if df[col] == 'object': # 如果是文字型 / 類別型欄位, 就先補缺 'None' 後, 再做標籤編碼
        df[col] = df[col].fillna('None')
        df[col] = LEncoder.fit_transform(df[col])
    else: # 其他狀況(本例其他都是數值), 就補缺 -1
        df[col] = df[col].fillna(-1)
    df[col] = MMEncoder.fit_transform(df[col].values.reshape(1, 1))

# 將ids跟pred合併 (為了產出csv)
sub = pd.DataFrame({'Id':ids, 'SalePrice': pred})
sub.to_csv('house_baseline.csv', index=False)

# how to do logistic regression
estimator = LogisticRegression()
estimator.fit(train_X, train_Y)
pred = estimator.predict(test_X)

# 秀出資料欄位的類型, 與對應的數量
# df.dtypes : 轉成以欄位為 index, 類別(type)為 value 的 DataFrame
# .reset_index() : 預設是將原本的 index 轉成一個新的欄位, 如果不須保留 index, 則通常會寫成 .reset_index(drop=True)
dtype_df = df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"] # 給col名稱（最上面）
dtype_df = dtype_df.groupby("Column Type").aggregate('count').reset_index() # groupby後是一個物件,需要用一個aggregate方法來聚集總數,在重置索引值
dtype_df

# unique vs nunique
df[int_features].unique()
>>df[int_features].unique()
好像有一種unique方法是會返回唯一值
df[int_features].nunique()

# 檢查是否缺值
## 檢查欄位缺值數量 (去掉.head()可以顯示全部)
df.isnull().sum() // .sort_values(ascending=False).head()

#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)

# 削減文字型欄位, 只剩數值型欄位
df = df[num_features]

# df補空值
df = df.fillna(df.mean())
df[col] = df[col].fillna(df[col].mode()[0])

# 'LotFrontage' 有空缺時, 以同一區 (Neighborhood) 的 LotFrontage 中位數填補 (可以視為填補一種群聚編碼 )
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

# 做線性迴歸
estimator = LinearRegression() // estimator = LogisticRegression() 改作羅吉斯回歸
cross_val_score(estimator, train_X, train_Y, cv=5).mean() # cv=k-fold

# df做最大最小化
df_temp = MinMaxScaler().fit_transform(df)

# df搭配標準化
df_temp = StandardScaler().fit_transform(df)

# 散佈圖 與目標值的散佈圖
import seaborn as sns
import matplotlib.pyplot as plt
sns.regplot(x=df['GrLiveArea'], y=train_Y)
plt.show()

# 對離群值做壞壞的事
## 將 GrLivArea 限制在 800 到 2500 以內, 調整離群值
df['GrLivArea'] = df['GrLivArea'].clip(800, 2500)
## 將 GrLivArea 限制在 800 到 2500 以內, 捨棄離群值
keep_indexs = (df['GrLivArea']> 800) & (df['GrLivArea']< 2500) # 這裡一定要用括號
df = df[keep_indexs]
train_Y = train_Y[keep_indexs]

# 對數去偏
對數去偏就是使用自然對數去除偏態，常見於計數／價格這類非負且可能為0的欄位
因為需要將0對應到0,所以先加一在取對數
還原時使用expm1也就是先取指數後再減一

# 方根去偏
就是將數值減去最小值後開根號,最大值有限時使用(例：成績轉換)

# 分布去偏 (boxbox)
函數的 lambda(λ) 參數為 0 時等於 log 函數，lambda(λ) 為 0.5 時等於開根號 (即sqrt)，因此可藉由參數的調整更靈活地轉換數值，但要特別注意Y的輸入數值必須要為正 (不可為0)

# 直方圖(含kde)
sns.distplot(df['LotArea'][:train_num])
plt.show()

# 標籤編碼
df_temp = pd.DataFrame() # initialize
for c in df.columns: # 每一行都要變
    df_temp[c] = LabelEncoder().fit_transform(df[c])
-----------------------------------------------------
# 獨熱編碼
df_temp = pd.get_dummies(df)
# 只對某col做獨熱
df = pd.get_dummies(df, columns = ["Ticket"], prefix="T") // prefix是針對column name

train_X = df_temp[:train_num]

# 看時間
print(f'shape : {train_X.shape}')
print(f'score : {cross_val_score(estimator, train_X, train_Y, cv=5).mean()}')
print(f'time : {time.time() - start} sec')


# 標籤編碼 vs 獨熱編碼 in 線性迴歸 & 梯度提升樹
線性迴歸時, 獨熱編碼不僅時間花費很多, 準確率也大幅下降
梯度提升樹時, 分數有小幅提升, 執行時間則約為兩倍
可見獨熱不僅計算時間會增加不少, 也不太適合用在線性迴歸上
##使用時機
綜合建議非深度學習時，類別型特徵建議預設採標籤編碼;深度學習時，預設採獨熱編碼因非深度學習時主要是樹狀模型 (隨機森林 / 梯度提升樹等基於決策樹的模型)，⽤兩次門檻就能分隔關鍵類別;但深度學習主要依賴倒傳遞，標籤編碼會不易收斂
* 當特徵重要性⾼，且可能值較少時，才應該考慮獨熱編碼

# ========類別型特徵預設編碼方式=======
## 均值編碼 (Mean Encoding)  : 使⽤目標值的平均值，取代原本的類別型特徵
###上面的問題
如果交易樣本非常少, 且剛好抽到極端值, 平均結果可能會有誤差很⼤ => 平滑化 ( Smoothing )
均值編碼平滑化公式:
新類別均值 = (原類別平均*類別樣本數+全部的總平均*調整因子)/類別樣本數+調整因子

* 調整因⼦子⽤用來來調整平滑化的程度，依總樣本數調整
小提醒：均值編碼容易overfitting(可利用cross_val_score確認前後分數 來驗證是否合適)

# 均值編碼範例
# 均值編碼 + 線性迴歸
data = pd.concat([df[:train_num], train_Y], axis=1)
for c in df.columns:
    mean_df = data.groupby([c])['SalePrice'].mean().reset_index()
    mean_df.columns = [c, f'{c}\_mean']
    data = pd.merge(data, mean_df, on=c, how='left')
    data = data.drop([c] , axis=1)
print(mean_df)
data = data.drop(['SalePrice'] , axis=1)
estimator = LinearRegression()

# 計數編碼
計數編碼是計算類別在資料中的出現次數，當⽬標平均值與類別筆數呈正/負相關時，可以考慮使用當相異類數量相當⼤時，其他編碼⽅式效果更差，可以考慮雜湊編碼以節省時間
註 : 雜湊編碼效果也不佳，這類問題更好的解法是嵌入式編碼(Embedding)，但是需要深度學習並有其前提，因此這裡暫時不排入課程

# 加上 'Ticket' 欄位的計數編碼
# 第一行 : df.groupby(['Ticket']) 會輸出 df 以 'Ticket' 群聚後的結果, 但因為群聚一類只會有一個值, 因此必須要定義運算
# 例如 df.groupby(['Ticket']).size(), 但欄位名稱會變成 size, 要取別名就需要用語法 df.groupby(['Ticket']).agg({'Ticket_Count':'size'})
# 這樣出來的計數欄位名稱會叫做 'Ticket_Count', 因為這樣群聚起來的 'Ticket' 是 index, 所以需要 reset_index() 轉成一欄
# 因此第一行的欄位, 在第三行按照 'Ticket_Count' 排序後, 最後的 DataFrame 輸出如 Out[5]
count_df = df.groupby(['Ticket'])['Name'].agg({'Ticket_Count':'size'}).reset_index()
# count_df = df.groupby(['Ticket'])['Name'].agg({'Ticket_Count':'size'}).reset_index()
# # 但是上面資料表結果只是 'Ticket' 名稱對應的次數, 要做計數編碼還需要第二行 : 將上表結果與原表格 merge, 合併於 'Ticket' 欄位
# # 使用 how='left' 是完全保留原資料表的所有 index 與順序
df = pd.merge(df, count_df, on=['Ticket'], how='left')
count_df.sort_values(by=['Ticket_Count'], ascending=False).head(10)

## 特徵雜湊
# 這邊的雜湊編碼, 是直接將 'Ticket' 的名稱放入雜湊函數的輸出數值, 為了要確定是緊密(dense)特徵, 因此除以10後看餘數
# 這邊的 10 是隨機選擇, 不一定要用 10
df_temp['Ticket_Hash'] = df['Ticket'].map(lambda x:hash(x) % 10)

# 跟雜湊差這行程式碼
df_temp['Ticket_Count'] = df['Ticket_Count'] // 與上述比較

# 找none replace 0
df_train['Resolution'].str.replace('NONE', '0')

# 時間型特徵
最常用的是特徵分解-拆解成年/月／日/時/分/秒的分類值
週期循環特徵是將時間"循環"特性改成特徵⽅式, 設計關鍵在於⾸尾相接, 因此我們需要使用 sin /cos 等週期函數轉換
常見的週期循環特徵有 - 年週期(季節) / 周期(例假日) / 日週期(日夜與生活作息), 要注意的是最⾼與最低點的設置
## 使用方法：
1.大概就是先用apply對欄位中每一個值先做parse...
df['pickup_datetime'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S UTC')
這時候得到的會是string -> datetime format
2.這時候可以用上面得到的東西,分別get parse後的結果
df['pickup_year'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%Y')).astype('int64')

# 群聚編碼(Group by Encoding)
1.類似均值編碼的概念，可以取類別平均值 (Mean) 取代險種作為編碼但因為比較像性質描寫，因此還可以取其他統計值，如中位數 (Median)，眾數(Mode)，最大值(Max)，最⼩值(Min)，次數(Count)...等
2.數值型特徵對文字型特徵最重要的特徵組合方式 常見的有 mean, median, mode, max, min, count 等
## 所以什麼時候需要群聚編碼呢？
->與數值特徵組合相同的時候
先以領域知識或特徵重要性挑選強⼒特徵後, 再將特徵組成更強的特徵兩個特徵都是數值就⽤特徵組合, 其中之⼀是類別型就用聚類編碼

# 群聚編碼
# 生活總面積(GrLivArea)對販售條件(SaleCondition)做群聚編碼
# 寫法類似均值編碼, 只是對另一個特徵, 而非目標值
df['SaleCondition'] = df['SaleCondition'].fillna('None')
mean_df = df.groupby(['SaleCondition'])['GrLivArea'].mean().reset_index()
mode_df = df.groupby(['SaleCondition'])['GrLivArea'].apply(lambda x: x.mode()[0]).reset_index()
median_df = df.groupby(['SaleCondition'])['GrLivArea'].median().reset_index()
max_df = df.groupby(['SaleCondition'])['GrLivArea'].max().reset_index()
## 下行是用mean_df&mode_df 先合併而且是用pd.merge
temp = pd.merge(mean_df, mode_df, how='left', on=['SaleCondition'])
temp = pd.merge(temp, median_df, how='left', on=['SaleCondition'])
temp = pd.merge(temp, max_df, how='left', on=['SaleCondition'])
temp.columns = ['SaleCondition', 'Area_Sale_Mean', 'Area_Sale_Mode', 'Area_Sale_Median', 'Area_Sale_Max']
temp
## 繼續跟主df合併
df = pd.merge(df, temp, how='left', on=['SaleCondition'])
df = df.drop(['SaleCondition'] , axis=1)
df.head()

=== 下面可以過濾一些string ===
# 只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')

# 削減文字型欄位, 只剩數值型欄位
df = df[num_features]
df = df.fillna(-1)
MMEncoder = MinMaxScaler()
df.head()

# 特徵選擇概念
特徵需要適當的增加與減少
增加特徵：特徵組合，群聚編碼
減少特徵：特徵選擇
## 特徵選擇的三個方法
過濾法 (Filter) : 選定統計數值與設定⾨門檻，刪除低於⾨門檻的特徵 ex:相關係數過濾法
包裝法 (Wrapper) : 根據⽬目標函數，逐步加入特徵或刪除特徵
嵌入法 (Embedded) : 使⽤用機器學習模型，根據擬合後的係數，刪除係數低於⾨檻的特徵本⽇內容將會介紹三種較常⽤的特徵選擇法 ex:L1(Lasso)嵌入法，GDBT(梯度提升樹)嵌入法

# 相關係數過濾法
power by heatmap..
找到⽬標值 (房價預估目標為SalePrice)之後，觀察其他特徵與⽬標值相關係數
預設顏⾊越紅表⽰越正相關，越藍負相關因此要刪除紅框中顏色較淺的特徵 : 訂出相關係數門檻值，特徵相關係數絕對值低於門檻者刪除SalePrice

# Lasso(L1) 嵌入法
使⽤Lasso Regression 時，調整不同的正規化程度，就會⾃然使得⼀部分的特徵係數為０，因此刪除的是係數為０的特徵，不須額外指定⾨門檻，但需調整正規化程度

# GDBT梯度提升樹 嵌入法
使用梯度提升樹擬合後，以特徵在節點出現的頻率當作特徵重要性，以此刪除重要性低於⾨檻的特徵，這種作法也稱為 GDBT 嵌入法由於特徵重要性不只可以刪除特徵，也是增加特徵的關鍵參考

# 相關係數法實作
## 計算df整體相關係數, 並繪製成熱圖 計算整體的 只會留數值欄位
import seaborn as sns
import matplotlib.pyplot as plt
corr = df.corr()
print(corr)
sns.heatmap(corr)
plt.show()
## 篩選相關係數大於 0.1 或小於 -0.1 的特徵 (要有.index 否則會挑不到)
high_list = list(corr[(corr['SalePrice']>0.1) | (corr['SalePrice']<-0.1)].index)
print(high_list)
------ 要記得pop target !!!

# Lasso(L1) 實作
## step 1
from sklearn.linear_model import Lasso
L1_Reg = Lasso(alpha=0.001)
train_X = MMEncoder.fit_transform(df)
L1_Reg.fit(train_X, train_Y)
L1_Reg.coef_
## step 2
from itertools import compress
L1_mask = list((L1_Reg.coef_>0) | (L1_Reg.coef_<0))
L1_list = list(compress(list(df), list(L1_mask)))
L1_list
## step 3
### L1_Embedding 特徵 + 線性迴歸
train_X = MMEncoder.fit_transform(df[L1_list])
cross_val_score(estimator, train_X, train_Y, cv=5).mean()

# 這邊筆記一下compress的用法
compress

compress 的使用形式如下：

compress(data, selectors)
compress 可用於對數據進行篩選，當 selectors 的某個元素為 true 時，則保留 data 對應位置的元素，否則去除：
>>> from itertools import compress
>>>
>>> list(compress('ABCDEF', [1, 1, 0, 1, 0, 1]))
['A', 'B', 'D', 'F']
>>> list(compress('ABCDEF', [1, 1, 0, 1]))
['A', 'B', 'D']
>>> list(compress('ABCDEF', [True, False, True]))
['A', 'C']

＊我覺得有點像是快速做遮罩的概念

# 特徵的重要性
## 用決策樹來說明
1. 特徵重要性預設⽅式是取特徵決定分支的次數
2. 但分⽀次數以外，還有兩種更直覺的特徵重要性 : 特徵覆蓋度、損失函數降低量本例的特徵覆蓋度(假定八個結果樣本數量量⼀樣多)

* sklearn 當中的樹狀模型，都有特徵重要性這項⽅方法     (.feature_importances_)，⽽而實際上都是分⽀次數


# 因為需要把類別型與數值型特徵都加入, 故使用最簡版的特徵工程
LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()
for c in df.columns:
    df[c] = df[c].fillna(-1)
    if df[c].dtype == 'object':
        df[c] = LEncoder.fit_transform(list(df[c].values))
    df[c] = MMEncoder.fit_transform(df[c].values.reshape(-1, 1))
df.head()

# 找出你的重要性(by randomForest)
# 隨機森林擬合後, 將結果依照重要性由高到低排序
estimator = RandomForestRegressor()
estimator.fit(df.values, train_Y)
# estimator.feature_importances_ 就是模型的特徵重要性, 這邊先與欄位名稱結合起來, 才能看到重要性與欄位名稱的對照表
feats = pd.Series(data=estimator.feature_importances_, index=df.columns)
feats = feats.sort_values(ascending=False)
feats >>> 再利用feat去挑出前50% 的index

## 製作四特徵 : 加, 乘, 互除(分母加1避免除0) 看效果 (Note: 數值原本已經最大最小化介於 [0,1] 區間, 這四種新特徵也會落在 [0,1] 區間)
df['Add_char'] = (df['GrLivArea'] + df['OverallQual']) / 2
df['Multi_char'] = df['GrLivArea'] * df['OverallQual']
df['GO_div1p'] = df['GrLivArea'] / (df['OverallQual']+1) * 2
df['OG_div1p'] = df['OverallQual'] / (df['GrLivArea']+1) * 2
train_X = MMEncoder.fit_transform(df)
cross_val_score(estimator, train_X, train_Y, cv=5).mean()

# 分類預測的集成
已知來自法國的旅客⽣生存機率是 0.8，且年齡 40 到 50 區間的生存機率也是 0.8那麼同時符合兩種條件的旅客，⽣存機率應該是多少呢?
解法 : 邏輯斯迴歸(logistic regression)與其重組我們可以將邏輯斯迴歸理解成「線性迴歸 + Sigmoid 函數」⽽ sigmoid 函數理解成「成功可能性與機率的互換」這裡的成功可能性正表示更可能，負表⽰較不可能

# 葉編碼原理
樹狀模型作出預測時,模型預測時就會將資料分成好幾個區塊,也就是決策樹的葉點,每個葉點資料性質接近,可視為資料的一種分組
雖然不適合直接沿用樹狀模型機率，但分組⽅式有代表性，因此按照葉點將資料離散化，比之前提過的離散化⽅式更更精確，這樣的編碼我們就稱為葉編碼的結果，是⼀組模型產⽣的新特徵，我們可以使用邏輯斯回歸，重新賦予機率 (如下葉圖)，也可以與其他算法結合 (例例如 : 分解機Factorization Machine )使資料獲得新⽣
## 目的
葉編碼的⽬的是重新標記資料，以擬合後的樹狀狀模型分歧條件，將資料離散化，這樣比⼈為寫作的判斷條件更精準，更符合資料的分布情形

step:
每棵樹視為一個新特徵(葉點就是特徵有幾個值)
每個新特徵均為分類型特徵,決策樹的葉點與該特徵一一對應
最後再以邏輯斯回歸合併

# 有點不太明白 先記錄
# 因為訓練邏輯斯迴歸時也要資料, 因此將訓練及切成三部分 train / val / test, 採用 test 驗證而非 k-fold 交叉驗證
# train 用來訓練梯度提升樹, val 用來訓練邏輯斯迴歸, test 驗證效果
train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.5)
train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.5)

# 梯度提升樹調整參數並擬合後, 再將葉編碼 (＊.apply) 結果做獨熱 / 邏輯斯迴歸
# 調整參數的方式採用 RandomSearchCV 或 GridSearchCV, 以後的進度會再教給大家, 本次先直接使用調參結果
gdbt = GradientBoostingClassifier(subsample=0.93, n_estimators=320, min_samples_split=0.1, min_samples_leaf=0.3, 
                                  max_features=4, max_depth=4, learning_rate=0.16)
onehot = OneHotEncoder()
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
gdbt.fit(train_X, train_Y)
onehot.fit(gdbt.apply(train_X)[:, :, 0])
lr.fit(onehot.transform(gdbt.apply(val_X)[:, :, 0]), val_Y)
# 將梯度提升樹+葉編碼+邏輯斯迴歸結果輸出
pred_gdbt_lr = lr.predict_proba(onehot.transform(gdbt.apply(test_X)[:, :, 0]))[:, 1]
fpr_gdbt_lr, tpr_gdbt_lr, _ = roc_curve(test_Y, pred_gdbt_lr)
# 將梯度提升樹結果輸出
pred_gdbt = gdbt.predict_proba(test_X)[:, 1]
fpr_gdbt, tpr_gdbt, _ = roc_curve(test_Y, pred_gdbt)
# 畫roc_curve 
plt.plot([0, 1], [0, 1], 'k--') # 不太懂
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.plot(fpr_rf_lr, tpr_rf_lr, label='RF + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

# 如何解決過擬合或欠擬合
##過擬合
•增加資料量•降低模型複雜度•使用正規化 (Regularization)
##⽋擬合
•增加模型複雜度•減輕或不使⽤正規化

# train_test_split 函數切分(train, test row數一樣)
ex:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# K-fold Cross-validation 切分資料
kf = KFold(n_splits=5) ＃最少要兩切 預設為３ 這是建立一個物件
i = 0
for train_index, test_index in kf.split(X):
    i +=1 
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("FOLD {}: ".format(i))
    print("X_test: ", X_test)
    print("Y_test: ", y_test)
    print("-"＊30)

# np.where用法
np.where(y==1)[0] #ｙ是一個array裡的數字 出來的是索引
np.where(x > 0.5, 1, 0)

# 回歸 vs. 分類
機器學習的監督式學習中主要分為回歸問題與分類問題。
回歸代表預測的目標值為實數 (-∞⾄至∞) -> 回歸問題是可以轉化為分類問題
分類代表預測的目標值為類別 (0 或 1)

# 二元分類 (binary-class) vs. 多元分類 (Multi-class)
二元分類，顧名思義就是⽬標的類別僅有兩個。像是詐騙分析 (詐騙⽤戶 vs. 正常⽤戶)、瑕疵偵測 (瑕疵 vs. 正常)多元分類則是⽬標類別有兩種以上。
如⼿寫數字辨識有 10 個類別(0~9),影像競賽 ImageNet 更是有⾼達 1,000 個類別需要分類

# Multi-class vs. Multi-label
當每個樣本都只能歸在⼀個類別，我們稱之為多分類 (Multi-class) 問題；⽽一個樣本如果可以同時有多個類別，則稱為多標籤 (Multi-label)。了解專案的⽬標是甚麼樣的分類問題並選⽤適當的模型訓練。

# from sklearn import datasets
X, y = datasets.make_regression(n_features=1, random_state=42, noise=4) # 生成資料集
model = LinearRrgression()
model.fit(X, y)
prediction = model.predict(X)

# 評估方法
mae = metrics.mean_absolute_error(prediction, y) # 使用 MAE 評估
mse = metrics.mean_squared_error(prediction, y) # 使用 MSE 評估
r2 = metrics.r2_score(prediction, y) # 使用 r-square 評估
auc = metrics.roc_auc_score(y_test, y_pred) # 使用 roc_auc_score 來評估。 **這邊特別注意 y_pred 必須要放機率值進去!**

# 資料二元化後評估
f1 = metrics.f1_score(y_test, y_pred_binarized) # 使用 F1-Score 評估
precision = metrics.precision_score(y_test, y_pred_binarized) # 使用 Precision 評估
recall  = metrics.recall_score(y_test, y_pred_binarized) # 使用 recall 評估
##資料二元化
threshold = 0.5 
y_pred_binarized = np.where(y_pred>threshold, 1, 0) # 使用 np.where 函數, 將 y_pred > 0.5 的值變為 1，小於 0.5 的為 0

# np.newaxis()
X =  np.array([[1,2,3],[4,5,6], [7,8,9]])
print(X[:, np.newaxis, 2])
>> 3
6
9

# 用sklearn建立線性迴歸模型
## 建立一個線性回歸模型
regr = linear_model.LinearRegression()
## 將訓練資料丟進去模型訓練
regr.fit(x_train, y_train)
## 將測試資料丟進模型得到預測結果
y_pred = regr.predict(x_test)
## 可以看回歸模型的參數值
print('Coefficients: ', regr.coef_)
## 預測值與實際值的差距，使用 MSE
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
## 畫出回歸模型與實際資料的分佈
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
plt.show()

# 用sklearn建立羅吉斯迴歸模型
## 讀取鳶尾花資料集
iris = datasets.load_iris()
## 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1, random_state=4)
## 建立模型
logreg = linear_model.LogisticRegression()
## 訓練模型
logreg.fit(x_train, y_train)
## 預測測試集
y_pred = logreg.predict(x_test)
## 精準度
acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

# 機器學習模型中的⽬標函數
•損失函數 (Loss function)損失函數衡量預測值與實際值的差異異，讓模型能往正確的⽅向學習
•正則化 (Regularization)正則化則是避免模型變得過於複雜，造成過擬合 (Over-fitting)
前面使用的損失函數是MSE,MAE
## 為了避免 Over-fitting，我們可以把正則化加入⽬標函數中，此時目標函數 = 損失函數 + 正則化
## 因為正則化可以懲罰模型的複雜度，當模型越複雜時其值就會越⼤

# 正則化
正則化函數是⽤來衡量模型的複雜度
有 L1 與 L2 兩種函數
L1：αΣ|weights|    # Lasso = Linear Regression 加上 L1 （可以把某些特徵變為０達到特徵篩選）
L2：αΣ(weights)^2  # Ridge = Linear Regression 加上 L2 （可以處理共線性,解決高度相關的原因是，能夠縮減 X 的高相關特徵)
其中有個超參數α可以調整正則化的強度，LASSO 與 Ridge 就是回歸模型加上不同的正則化函數
這兩種都是希望模型的參數值不要太⼤，原因是參數的數值變⼩，噪音對最終輸出的結果影響越小，提升模型的泛化能力，但也讓模型的擬合能⼒下降

# how to lasso ＆ ridge
## 建模的時候
lasso = linear_model.Lasso(alpha=1.0) // 不用加LinearRegression
ridge = linear_model.Ridge(alpha=1.0)

# 決策樹 (Decision Tree)
從訓練資料中找出規則，讓每⼀次決策能使訊息增益 (Information Gain) 最大化訊息
增益越⼤代表切分後的兩群資料，群內相似程度越⾼
## 建模
clf = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=0, min_samples_split=2, min_samples_leaf=1)
Criterion: 衡量資料相似程度的 metric
Max_depth: 樹能⽣長的最深限制
Min_samples_split: ⾄少要多少樣本以上才進⾏切分
Min_samples_leaf: 最終的葉⼦ (節點) 上⾄少要有多少樣本
clf.features_importances_ 

# 訊息增益
決策樹模型會用 features 切分資料，該選用哪個 feature 來切分則是由訊息增益的⼤大⼩小決定的。希望切分後的資料相似程度很⾼，通常使⽤吉尼係數來來衡量相似程度。
## 決策樹的特徵重要性（Feature importance)
1.我們可以從構建樹的過程中，透過 feature 被⽤來切分的次數，來得知哪些features 是相對有用的
2.所有 feature importance 的總和為 1
3.實務上可以使⽤ feature importance 來了解模型如何進行分類
## how to get feature importance
print(iris.feature_names)
## feature importance numeric
print("Feature importance: ", clf.feature_importances_)
## classfier or regressor
There is a huge difference between classifiers and regressors. Classifiers predict one class from a predetermined list or probabilities of belonging to a class. Regressors predict some value, which could be almost anything.
Differeng metrics are used for classification and regression. 
So it isn't a good idea to use classifier for regression problem and vice versa.

# Gini vs Entropy
Gini impurity and Information Gain Entropy are pretty much the same. And people do use the values interchangeably. Below are the formulae of both:
ref :: https://datascience.stackexchange.com/questions/10228/when-should-i-use-gini-impurity-as-opposed-to-information-gain

# sklearn 建立決策樹模型
根據回歸/分類問題分別建立不同的 Classifier
from sklearn.tree_model import DecisionTreeRegressor
from sklearn.tree_model import DecisionTreeClassifier
clf = DecisionTreeClassifier()

## 建立模型四步驟

在 Scikit-learn 中，建立一個機器學習的模型其實非常簡單，流程大略是以下四個步驟

1. 讀進資料，並檢查資料的 shape (有多少 samples (rows), 多少 features (columns)，label 的型態是什麼？)
    - 讀取資料的方法：
        - **使用 pandas 讀取 .csv 檔：**pd.read_csv
        - **使用 numpy 讀取 .txt 檔：**np.loadtxt 
        - **使用 Scikit-learn 內建的資料集：**sklearn.datasets.load_xxx
    - **檢查資料數量：**data.shape (data should be np.array or dataframe)
2. 將資料切為訓練 (train) / 測試 (test)
    - train_test_split(data)
3. 建立模型，將資料 fit 進模型開始訓練
    - clf = DecisionTreeClassifier()
    - clf.fit(x_train, y_train)
4. 將測試資料 (features) 放進訓練好的模型中，得到 prediction，與測試資料的 label (y_test) 做評估
    - clf.predict(x_test)
    - accuracy_score(y_test, y_pred)
    - f1_score(y_test, y_pred)

# DT vs. CART
原本 DT 是根據 information gain(IG) 來決定要怎麼切割
CART 是找個 impurity function(IF) 來決定要怎麼切割

# 決策樹的缺點
- 若不對決策樹進行限制 (樹深度、葉⼦上至少要有多少樣本等)，決策樹非常容易Over-fitting 
- 為了解決策樹的缺點，後續發展出了隨機森林的概念，以決策樹為基底延伸出的模型

# 集成模型 - 隨機森林 (Random Forest)
集成 (Ensemble) 是將多個模型的結果組合在⼀起，透過投票或是加權的⽅方式得到最終結果

# Where is random?
- 決策樹⽣成時，是考慮所有資料與特徵來做切分的
- ⽽隨機森林的每⼀棵樹在⽣成過程中，都是隨機使用⼀部份的訓練資料與特徵代表每棵樹都是⽤隨機的資料訓練⽽成的

# Random Forest
在 training data 中, 從中取出一些 feature & 部份 data 產生出 Tree (通常是CART)
並且重複這步驟多次, 會產生出多棵 Tree 來
最後利用 Ensemble (Majority Vote) 的方法, 結合所有 Tree, 就完成了 Random Forest

1. 準備 training data 
## Bootstrap
為了讓每棵有所不同, 主要就是 training data 的採樣結果都會不太一樣
## Bagging
一種採樣方式, 假設全體 training data 有N筆, 你要採集部分資料, 
但是又不想要採集到全體的資料 (那就不叫採集了), 要如何做?
一般常見的方式為: 從 N 筆 data 挑資料, 一次挑一筆, 挑出的會再放回去, 最後計算的時候重複的會不算(with replacement), 假設最後為y, N > y

因為是用 bagging on data, 所以每棵 Tree 在建立的時候, 都會是用不一樣的 data 去建立的
- ** Random Forest 所建立的每棵 Tree, 在 data 跟 feature 上, 都有一定程度上的不同
- ** 設定最少要 bagging 出 (k / 2) + 1 的 feature, 才比較有顯著結果, K 為原本的 feature 數量,或者另外一個常見設定是 square(k)

2. Build Tree
這邊, 就沒什麼好說的了, 只要將前述的 data & feature 準備好, 餵入 CART 就可以了
唯一要注意的事情, Random Forest 不須要讓任何的 Tree 做 prune

3. Ensemble
簡單來說就是合體
給我一筆 data, 我會讓這 50 棵 Tree 分別去預估可能的 class, 最後全體投票, 多數決決定
如果今天是用 Regression 的 RF, 則是加總起來除以總棵數, 就是預估的答案

4. Out Of Bag (bagging沒用到的data)
衡量可能的錯誤率
** 因為重複採樣的關係, 平均來講, 每棵大約會有 1/3 training data 採樣不到
所以收集這些 data, 最後等到 Forest 建立完成之後, 將這些 data 餵進去判斷, 最後得出錯誤率
這方式稱為 Out-Of-Bag (OOB)

## Notes
1. 若隨機森林中樹的數量太少，造成嚴重的Overfit，是有可能會比決策樹差。但如果都是⽤預設的參數，實務上不太會有隨機森林比決策樹差的情形，要特別注意程式碼是否有誤
2. 隨機森林中的每一棵樹，是希望能夠沒有任何限制，讓樹可以持續生長 (讓樹生成很深，讓模型變得複雜) 不要過度生長，避免 Overfitting
隨機森林: 希望每棵樹都能夠盡量複雜，然後再通過投票的方式，處理過擬合的問題。因此希望每棵樹都能夠盡量的生長
0.632 bootstrap: 這是傳統的統計問題，採用取後放回的方式，抽取與資料量同樣大小的 N 筆資料，約會使用 63.2 % 的原生資料。

# Random Forest 建模
from sklearn.ensemble import RandomForestClassifier // 代表隨機森林是個集成模型
## 讀取鳶尾花資料集
iris = datasets.load_iris()
## 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=4)
## 建立模型 (使用 20 顆樹，每棵樹的最大深度為 4)
clf = RandomForestClassifier(n_estimators=20, max_depth=4)
## 訓練模型
clf.fit(x_train, y_train)
## 預測測試集
y_pred = clf.predict(x_test)

# note:
同樣是樹的模型，所以像是 max_depth, min_samples_split 都與決策樹相同可決定要⽣成數的數量，越多越不容易過擬和，但是運算時間會變長
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(
n_estimators=10, #決策樹的數量
criterion="gini",
max_features="auto", #如何選取 features         
max_depth=10,
min_samples_split=2,
min_samples_leaf=1)

# 梯度提升機 Gradient Boosting Machine
隨機森林使⽤的集成⽅法稱為 Bagging (Bootstrap aggregating)，用抽樣的資料與 features ⽣成每⼀棵樹，最後再取平均
Boosting 則是另一種集成方法，希望能夠由後⾯生成的樹，來修正前⾯樹學不好的地方要怎麼修正前面學錯的地⽅方呢？計算 Gradient!
每次生成樹都是要修正前⾯樹預測的錯誤，並乘上 learning rate 讓後面的樹能有更多學習的空間

## Bagging vs. Boosting
Bagging 是透過抽樣 (sampling) 的⽅式來生成每⼀棵樹，樹與樹之間是獨立生成的
Boosting 是透過序列 (additive)的⽅式來生成每一顆樹，每棵樹都會與前⾯的樹關聯，因為後⾯的樹要能夠修正

# 使用Sklearn中的梯度提升機
梯度提升機同樣是個集成模型，透過多棵決策樹依序⽣生成來來得到結果，緩解原本決策樹容易易過擬和的問題，實務上的結果通常也會比決策樹來來得好
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingClassifier()

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(
loss="deviance", #Loss 的選擇，若若改為 exponential 則會變成Adaboosting 演算法，概念相同但實作稍微不同
learning_rate=0.1, #每棵樹對最終結果的影響，應與 n_estimators 成反比
n_estimators=100 #決策樹的數量量)

# Random Forest vs. Gradient boosting
決策樹計算特徵重要性的概念是，觀察某⼀特徵被⽤來切分的次數⽽定。
假設有兩個⼀模一樣的特徵，在隨機森林中每棵樹皆為獨立，因此兩個特徵皆有可能被使用，最終統計出來的次數會被均分。
在梯度提升機中，每棵樹皆有關連，因此模型僅會使⽤其中⼀個特徵，另⼀個相同特徵的重要性則會消失

# coding
clf = GradientBoostingClassifier()

# 超參數調整
之前接觸到的所有模型都有超參數需要設置
•LASSO，Ridge: α的⼤⼩
•決策樹：樹的深度、節點最⼩樣本數
•隨機森林：樹的數量
這些超參數都會影響模型訓練的結果，建議先使用預設值，再慢慢進⾏調整超參數會影響結果，
但提升的效果有限，資料清理與特徵工程才能最有效的提升準確率，調整參數只是⼀個加分的⼯具。
## how to 調整
窮舉法 (Grid Search)：直接指定超參數的組合範圍，每⼀組參數都訓練完成，再根據驗證集 (validation) 的結果選擇最佳數
隨機搜尋 (Random Search)：指定超參數的範圍，⽤均勻分布進⾏參數抽樣，用抽到的參數進行訓練，再根據驗證集的結果選擇最佳參數，隨機搜尋通常都能獲得更佳的結果
## step by step
若持續使⽤同⼀份驗證集 (validation) 來調參，可能讓模型的參數過於擬合該驗證集，正確的步驟是使用 Cross-validation確保模型泛化性
1. 先將資料切分為訓練/測試集，測試集保留不使⽤
2. 將剛切分好的訓練集，再使⽤Cross-validation 切分 K 份訓練/驗證集
3. 用 grid/random search 的超參數進行訓練與評估
4. 選出最佳的參數，⽤該參數與全部訓練集建模
5. 最後使用測試集評估結果
** 超參數調整對最終結果影響很⼤嗎？
超參數調整通常都是機器學習專案的最後步驟，因為這對於最終的結果影響不會太多，多半是近⼀步提升 3-5 % 的準確率，但是好的特徵工程與資料清理是能夠一口氣提升 10-20 ％的準確率！

## coding
# 設定要訓練的超參數組合
n_estimators = [50, 100, 150]
max_depth = [1, 3, 5]
param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)
# 建立搜尋物件，放入模型及參數組合字典 (n_jobs=-1 會使用全部 cpu 平行運算)
grid_search = GridSearchCV(reg, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1) # reg是上面建立的模型
# 開始搜尋最佳參數
grid_result = grid_search.fit(x_train, y_train)
# 預設會跑 3-fold cross-validadtion，總共 9 種參數組合，總共要 train 27 次模型 

# 印出最佳結果與最佳參數
print("Best Accuracy: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# 集成
集成是使用不同方式,結合不同的分類器,作為綜合預測的做法統稱。
將模型截長補短,也可以說是機器學習裡的合議制/多數決。
其中又分為資料面的集成 : 如裝袋法(Bagging) / 提升法(Boosting)
以及模型與特徵的集成 : 如混合泛化(Blending) / 堆疊泛化(Stacking)

- 裝袋法(Bagging)裝袋法是將資料放入袋中抽取，每回合結束後全部放回袋中重抽再搭配弱分類器取平均/多數決結果，最有名的就是前⾯學過的隨機森林
- 提升法(Boosting)提升法則是由之前模型的預測結果，去改變資料被抽到的權重或目標值。將錯判資料被抽中的機率放⼤，正確的縮小，就是⾃適應提升 (AdaBoost, Adaptive Boosting)如果是依照估計誤差的殘差項調整新⽬標值，則就是梯度提升機 (Gradient Boosting Machine)的作法，只是梯度提升機還加上用梯度來選擇決策樹分支

## 資料集成Bagging / Boosting
•使⽤不同訓練資料 + 同⼀種模型，多次估計的結果合成最終預測
## 模型與特徵集成Voting / Blending / Stacking
•使⽤用同⼀資料 + 不同模型，合成出不同

# 混合泛化 ( Blending )
將不同模型的預測值加權合成，權重和為1 如果取預測的平均 or ⼀人一票多數決(每個模型權重相同)，則又稱為投票泛化(Voting)
- 優點：
1. 容易使⽤
不只在⼀般機器學習中有用，影像處理或⾃然語⾔處理等深度學習，也⼀樣可以使用因為只要有預測值(Submit 檔案)就可以使用，許多跨國隊伍就是靠這個方式合作另⼀方面也因為只要⽤預測值就能計算，在競賽中可以快速合成多種比例的答案，妥善消耗掉每⼀天剩餘的 Submit 次數
2. 效果顯著
Kaggle 競賽截止⽇前的 Kernel，有許多只是對其他人的輸出結果做Blending，但是因為分數較高，因此也有許多⼈樂於推薦與發表在2015年前的⼤賽中，Blending 仍是主流，有競賽的送出結果，是上百個模型的 Blending

注意: 個別單模效果都很好(有調參)並且模型差異大，單模要好尤其重要，如果單模效果差異太大，Blending 的效果提升就相當有限
## coding
# 混合泛化預測檔 (依 Kaggle 傳回分數調整比重, 越準確者比重越高, 依資料性質有所不同)
blending_pred = linear_pred * 0.30 + gdbt_pred * 0.67 + rf_pred * 0.03
sub = pd.DataFrame({'Id': ids, 'SalePrice': np.expm1(blending_pred)})
sub.to_csv('house_blending.csv', index=False)

# 檢查 DataFrame 空缺值的狀態
def na_check(df_data):
    data_na = (df_data.isnull().sum() / len(df_data)) * 100
    data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :data_na})
    display(missing_data.head(10))
na_check(df)

# 特徵工程
# Sex : 直接轉男 0 女 1
df["Sex"] = df["Sex"].map({"male": 0, "female":1})
# Fare : 用 log 去偏態, 0 則直接取 0
df["Fare"] = df["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
# Age : 缺值用中位數補
df["Age"] = df["Age"].fillna(df['Age'].median())
df["Title"] = pd.Series(df_title) # 就做成一個有index的series方便df
df["Title"] = df["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df["Title"] = df["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
df["Title"] = df["Title"].astype(int)

# predict_proba
返回的是一个 n 行 k 列的数组， 第 i 行 第 j 列上的数值是模型预测 第 i 个预测样本为某个标签的概率，并且每一行的概率和为1

# Stacking
Stacking 是在原本特徵上，⽤模型造出新特徵

相對於 Blending 的改良不只將預測結果混合，⽽是使用預測結果當新特徵更進⼀步的運⽤了資料輔助集成，但也使得 Stacking 複雜許多
Stacking 主要是把模型當作下一階的特徵編碼器來使⽤，但是待編碼資料與⽤來訓練編碼器的資料不可重複 (訓練測試的不可重複性)若將訓練資料切成兩組 : 待編碼資料太少，下⼀層的資料筆數就會太少，訓練編碼器的資料太少，則編碼器的強度就會不夠，這樣的困境該如何解決???
## solution
-> K-Fold 
將資料拆成 K 份 (圖中 K=5)，每 1/K 的資料要編碼時，使⽤其他的 K-1 組資料訓練模型/編碼器
## Q & A
Q1：能不能新舊特徵⼀起用，再用模型預測呢?
A1：可以，這裡其實有個有趣的思考，也就是 : 這樣不就可以⼀直一直無限增加特徵下去? 這樣後面的特徵還有意義嗎?不會 Overfitting 嗎?...其實加太多次是會 Overfitting 的，必需謹慎切分Fold以及新增次數
Q2：新的特徵，能不能再搭配模型創特徵，第三層第四層...⼀直下去呢?
A2：可以，但是每多⼀層，模型會越複雜 : 因此泛化(⼜稱為魯棒性)會做得更好，精準度也會下降，所以除非第一層的單模調得很好，否則兩三層就不需要繼續往下了
Q3：既然同層新特徵會 Overfitting，層數加深會增加泛化，兩者同時⽤是不是就能把缺點互相抵銷呢?
A3：可以!!⽽且這正是 Stacking 最有趣的地⽅，但真正實踐時，程式複雜，運算時間又要再往上⼀個量級，之前曾有⼤神寫過 StackNet 實現這個想法，⽤JVM 加速運算，但實際上使⽤時調參困難，後繼使⽤的⼈就少了

- Stacking 結果分數真的比較高嗎? 不⼀定，有時候單模更⾼高，有時候 Blending 效果就不錯，視資料狀況而定
- Stacking 可以做參數調整嗎? 主要差異是參數名稱寫法稍有不同
-「分類問題」的 Stacking 要注意兩件事：
1. 記得加上 use_probas=True(輸出特徵才會是機率值)
2. 輸出的總特徵數會是：模型數量 * 分類數量(回歸問題特徵數=模型數量)

## Notes:
1. 堆疊泛化因為將模型預測當作特徵時，要避免要編碼的資料與訓練編碼器的資料重疊，因此設計上看起來相當複雜
2. 堆疊泛化理論上在堆疊層數上沒有限制，但如果第一層的單模不夠複雜，堆疊⼆三層後，改善幅度就有限了
3. 混合泛化相對堆疊泛化來說，優點在於使⽤容易，缺點在於無法更深入的利用資料更進一步混合模型

# coding
# 堆疊泛化套件 mlxtend, 需要先行安裝(使用 pip 安裝即可)在執行環境下
from mlxtend.regressor import StackingRegressor

# 因為 Stacking 需要以模型作為第一層的特徵來源, 因此在 StackingRegressor 中,
# 除了要設本身(第二層)的判定模型 - meta_regressor, 也必須填入第一層的單模作為編碼器 - regressors
# 這裡第二層模型(meta_regressor)的參數, 一樣也需要用 Grid/Random Search, 請參閱講義中的 mlxtrend 網頁
meta_estimator = GradientBoostingRegressor(tol=10, subsample=0.44, n_estimators=100, 
                                           max_features='log2', max_depth=4, learning_rate=0.1)
stacking = StackingRegressor(regressors=[linear, gdbt, rf], meta_regressor=meta_estimator)
## 後面繼續用stack fitting....


# 期中考筆記
## 找欄位空值
np.isnan(df['col'])
## pd time format
pd.to_datetime(df['col'], format='%Y%m%d')
## 取出時間
pd.Timedelta(15, 'D')
## dummies不支援nan所以
tmp = pd.get_dummies(train['weekday'].replace(-1, np.nan), prefix='weekday_')

# 非監督學習算法概要
聚類分析 : 尋找資料的隱藏模式
降低維度 : 特徵數太大且特徵間相關性高，以此⽅式縮減特徵維度
其他 : 關聯法則 (購物籃分析)、異常值偵測、探索性資料分析等
## note:
在不清楚資料特性、問題定義、沒有標記的情況下，非監督式學習技術可以幫助我們理清資料脈絡特徵數太龐大的情況下，非監督式學習可以幫助概念抽象化，⽤更簡潔的特徵描述資料非監督式學習以聚類算法及降低維度算法爲主

# 非監督
- 分群  k-means

# k-means 聚類算法
聚類算法⽤於把族群或資料點分隔成⼀系列的組合，使得相同 cluster 中的資料點比其他的組更相似
把所有資料點分成 k 個 cluster，使得相同 cluster 中的所有資料點彼此儘量相似，而不同 cluster 的資料點儘量不同。
距離測量（e.g. 歐氏距離）⽤於計算資料點的相似度和相異度。
每個 cluster 有⼀個中心點。中⼼點可理解為最能代表 cluster 的點。

# k-means 算法流程
1. 假設下圖是我們的 training set，我們目標是將資料分成 2 群
2. 隨機選取 2 個點，稱爲 cluster centroid.
3. 對每一個training example 根據他距離哪一個cluster centroid 較近, 標記為其中一個。
4. 然後把 centroid 移到同一群 training examples 的中⼼點 (update centroid)反覆進⾏ cluster assignment 及 update centroid, 直到 cluster assignment 不再導致 training example 被 assign 爲不同的標記 (算法收斂)
目標: 使總體群內平方誤差最小。

## note:
Random initialization: initial 設定的不同，會導致得到不同 clustering 的結果，可能導致 local optima，⽽非 global optima。
因爲沒有預先的標記，對於 cluster 數量多少才是最佳解，沒有標準答案，得靠⼿動測試觀察。

## coding:
from sklearn.cluster import KMeans
estimators = [('k_means_8', KMeans(n_clusters=8)),
              ('k_means_3', KMeans(n_clusters=3)),
              ('k_means_bad_init', KMeans(n_clusters=3, n_init=1,
                                               init='random'))]
fignum = 1
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
for name, est in estimators: # 上述的estimators 分別拆開
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    
# fit data
    est.fit(X)
    
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2], # 因為呈現的關係而這樣畫
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_title(titles[fignum - 1])
    ax.dist = 12 // 距離
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134) 
## azim : float, optional Azimuthal viewing angle, defaults to -60.
## elev : Elevation viewing angle, defaults to 30. Elevation viewing angle, defaults to 30.

for name, label in [('cls0', 0),
                    ('cls1', 1),
                    ('cls2', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([]) # 刪除刻度
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_title('Ground Truth')
ax.dist = 12

#fig.show()                                           

## Supervised learning vs. clustering
Supervised learning: 找出決策邊界(decision boundary)
Clustering: 目標在於找出資料結構

# K-mean 觀察 : 使用輪廓分析
## 評估的困難：非監督因為沒有⽬標值，因此無法使⽤目標值的預估與實際差距，來評估模型的優劣
類型：
- 有⽬標值的分群：如果資料有⽬標值，只是先忽略⽬標值做非監督學習，則只要微調後，就可以使⽤原本監督的測量函數評估準確性
- 無目標值的分群：但通常沒有⽬標值/⽬標值非常少才會⽤非監督模型，這種情況下，只能使用資料本⾝的分布資訊，來做模型的評估
精神：
同⼀群的資料點應該很近，不同群的資料點應該很遠，所以設計⼀種當同群資料點越近 / 不同群資料點越遠時越⼤的分數當資料點在兩群交界附近，希望分數接近 0
方法：
對任意單一資料點 i，「與 i 同⼀群」的資料點，距離 i 的平均稱為 ai 
「與 i 不同群」的資料點中，不同群距離 i 平均中，最近的稱為bi 
i 點的輪廓分數 si : (bi-ai) / max{bi, ai}
其實只要不是刻意分錯，bi 通常會大於等於 ai，所以上述公式在此條件下可以化簡為 1 - ai / bi 

## coding:
for n_clusters in range_n_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2) # 兩個圖片物件, fig用做調整 , (1,3)的話 要開(ax1, ax2, ax3)
    fig.set_size_inches(18, 7)
    
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim(0, len(X) + (n_clusters+1)*10) # clusters越來越多
    
    # 宣告 KMean 分群器, 對 X 訓練並預測
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_label = clusterer.fit_predict(X)
    
    # 計算所有點的 silhouette_score 平均
    silhouette_avg = silhouette_score(X, cluster_label)
    print(f'For n clusters {n_clusters} Silhoutte_avg: {silhouette_avg}')
    
    # 計算所有樣本的 The silhouette_score
    sample_silhouette_values = silhouette_samples(X, cluster_label)
    y_lower = 10
    
    for i in range(n_clusters):
        # 過濾label 然後排序
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels==i]
        
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                  0, ith_cluster_silhouette_values,
                  facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i)) # 字的位置
        y_lower = y_upper + 10
        
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    # 垂直線
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    # ax2 想要看每個樣本點的分群狀態,從另一個角度觀察分群
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')
    # s , linewidths
    # ax2 將中心點標注出來
    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c='white', alpha=1, s=200, edgecolor='k')
    # s for scale
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i , alpha=1, s=50, edgecolor='k')
        
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    # 最高級的標題
    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
plt.show()

## notes:
輪廓分數:
是⼀種同群資料點越近 / 不同群資料點越遠時會越⼤的分數，除了可以評估資料點分群是否得當，也可以用來評估分群效果
要以輪廓分析觀察 K-mean，除了可以將每個資料點分組觀察以評估資料點分群是否得當，也可⽤平均值觀察評估不同 K 值的分群效果

# 階層分群法
階層式分析⼀種構建 cluster 的層次結構的算法。該算法從分配給⾃己 cluster 的所有資料點開始。
然後，兩個距離最近的 cluster 合併為同一個 cluster。最後，當只剩下⼀個 cluster 時，該算法結束。

# 流程
不指定分群的數量
1. 每筆資料為⼀個 cluster
2. 計算每兩兩群之間的距離
3. 將最近的兩群合併成一群
4. 重覆步驟 2、3，直到所有資料合併成同⼀cluster

# 階層分群距離計算⽅式
- Single Link: 群聚與群聚間的距離可以定義為不同群聚中最接近兩點間的距離
- Complete Link: 群聚間的距離定義為不同群聚中最遠兩點間的距離，這樣可以保證這兩個集合合併後, 任何⼀對的距離不會⼤於 d。
- Average Link: 群聚間的距離定義為不同群聚間各點與各點間距離總和的平均。

## 優點:
1. 概念簡單，易於呈現
2. 不需指定群數缺點只適⽤於少量資料，⼤量資料會很難處理

## 缺點:
適⽤於少量資料，⼤量資料會很難處理

## K-means vs. 階層分群
K-mean 要預先定義群數(n of clusters)
階層分群可根據定義距離來分群(bottom-up)，也可以決定羣數做分羣 (top-down)

# coding
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
# 設定模型估計參數
estimators = [('hc_ward', AgglomerativeClustering(n_clusters=3, linkage="ward")),
              ('hc_complete', AgglomerativeClustering(n_clusters=3, linkage="complete")),
              ('hc_average', AgglomerativeClustering(n_clusters=3, linkage="average"))]

# 降維
why？
1. 壓縮資料, 有助於加速 learning algorithms（圖片仍保有輪廓和特徵）
2. 特徵組合及抽象化, 組合出新的、抽象化的特徵，減少冗餘的資訊
3. 資料視覺化, 特徵太多時，很難 visualize data, 不容易觀察資料

## PCA(Principal components analysis)
PCA 透過計算 eigen value, eigen vector, 可以將原本的 features 降維⾄特定的維度
•透過 PCA，可以將這 100 個 features 降成 2 個features
•新 features 為舊 features 的線性組合 * 新 features 彼此不相關Uncorrelated
將這些不必要的資訊捨棄除了可以加速 learning , 也可以避免⼀點overfitting。

## 應用
組合出來的這些feat,可以近而用來做supervised learning

## notes:
1. 不建議早期做PCA(可能underfitting)
2. 建議在optimization階段時,考慮PCA並觀察影響

# plt.clf() plt.cla()
matplotlib.pyplot.clf(): Clear the current figure. [ref]
matplotlib.pyplot.cla(): Clear the current axes. [ref]

## coding
from sklearn import decomposition
# 估計參數
centers = [[1, 1], [-1, -1], [1, -1]]
pca = decomposition.PCA(n_components=3)
# 建模
pca.fit(X)
X = pca.transform(X)
# 建空白畫布
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
# text3D
pca.fit(X)
## 簡單來說就是設定文字位置 
for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]: 
    ax.text3D(X[y == label, 0].mean(), # X[y==0]第一個column的值，也就是類別為0的第一個特徵的值
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
## Reorder the labels to have colors matching the cluster results
## 基本上只是調整三種花的顏色顯示, 讓相鄰的兩種花色對比較明顯而已,沒有太大意義, 你也可以試著調整其他順序
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

## 清刻度
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()

# PCA in 高維度.高複雜性 / ⼈可理解的資料集
由於 PCA 的強⼤，如果資料有意義的維度太低，則前幾個主成分就可以將資料解釋完畢
使⽤用一般圖形資料->高維
(須兼顧內容的複雜性與可理解性)

# coding
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
## pipeline
# 定義 PCA 與隨後的邏輯斯迴歸函數
logistic = SGDClassifier(loss='log', penalty='l2', max_iter=10000, tol=1e-5, random_state=0)
pca = PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
# 載入手寫數字辨識集
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
# 先執行 GridSearchCV 跑出最佳參數
param_grid = {
    'pca__n_components': [4, 10, 20, 30, 40, 50, 64],
    'logistic__alpha': np.logspace(-4, 4, 5),
}
search = GridSearchCV(pipe, param_grid, iid=False, cv=5, return_train_score=False)
search.fit(X_digits, y_digits)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
# 繪製不同 components 的 PCA 解釋度
pca.fit(X_digits)

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6,6))

ax0.plot(pca.explained_variance_ratio_, linewidth=2)
ax0.set_ylabel('PCA explained variance')

ax0.axvline(search.best_estimator_.named_steps['pca'].n_components, linestyle=':', label='n_components chosen')
ax0.legend(prop=dict(size=12))

# 繪製不同採樣點的分類正確率
results = pd.DataFrame(search.cv_results_)
components_col = 'param_pca__n_components'
best_clfs = results.groupby(components_col).apply(lambda g:g.nlargest(1, 'mean_test_score'))

best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score', legend=False, ax=ax1)# yerr是那個直直的線啊，另外那個是ax1模板丟給ax
ax1.set_ylabel('Classfication accuracy (val)')
ax1.set_xlabel('n_components')
#避免兩個圖重疊，使用tight_layout分開
# lim控制顯示範圍
plt.xlim(0,70)
plt.tight_layout()
plt.show()

# 降維方法- t-SNE(T 分佈隨機近鄰嵌入 t-SNE)
## why t-SNE????
1. PCA 是⼀種線性降維方式，因此若特徵間是非線性關係，會有underfitting 的問題
2. 求共變異數矩陣進⾏奇異值分解，因此會被資料的差異性影響，無法很好的表現相似性及分佈。

## definition:
t-SNE 也是⼀種降維⽅方式，但它⽤了更複雜的公式來表達⾼維和低維之間的關係。
高維的資料 -> ⽤ gaussian distribution 的機率密度函數近似
低維的資料 -> 用 t 分佈來近似
在⽤ KL divergence 計算相似度，再以梯度下降 (gradient descent) 求最佳解。

# 總結
優點：當特徵數量過多時，使用 PCA 可能會造成降維後的 underfitting，這時可以考慮使⽤t-SNE
缺點：t-SNE 的需要比較多的時間執⾏

# coding:
from sklearn import manifold
tsne = manifold.TSNE(n_components=2, random_state=0, init='pca', learning_rate=200., early_exaggeration=12.)
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min) # min-max
    
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)
    for i in range(n_samples):
        plt.text(X[i, 0], X[i, 1], str(y[i]), color=plt.cm.Set1(y[i]/10), fontdict={'weight':'bold', 'size':9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])
        for i in range(n_samples):
            dist = np.sum((X[i] - shown_images) ** 2, 1) # 算距離 兩個維度相加
            if np.min(dist) < 4e-3: # 太遠的不算
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i]) # offsetbox 是繪圖物件, .AnnotationBbox是加入數字的圖示(包含數字的方框)
            ax.add_artist(imagebox)
            plt.xticks([])
            plt.yticks([])
            if title is not None:
                plt.title(title)
# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")

X_tsne = tsne.fit_transform(X)

plot_embedding(X_tsne,
               "t-SNE embedding of the digits")

plt.show()

# np.r_ , np.c_
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.c_[a,b]

print(np.r_[a,b])
print(c)
print(np.c_[c,a])
1 2 3 4 5 6]

[[1 4]
 [2 5]
 [3 6]]
 
[[1 4 1]
 [2 5 2]
 [3 6 3]]

# tsne:流形還原
流形還原就是將高維度上相近的點，對應到低維度上相近的點，沒有資料點的地⽅方不列列入考量範圍
簡單的說，如果資料結構像瑞士捲⼀樣，那麼流形還原就是把它攤開鋪平 (流形還原資料集的其中⼀種，就是叫做瑞⼠捲-Swiss Roll)

-> 盡量保持資料點之間的遠近關係
# codding
# 設定模型與繪圖參數
n_samples = 300
n_components = 2
(fig, subplots) = plt.subplots(2, 5, figsize=(15, 6))
perplexities = [4, 6, 9, 14, 21, 30, 45, 66, 100]

# 設定同心圓資料點 
X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
red = y == 0 # 將 y 為 0 的 index set 存成變數 red
green = y == 1 # 將 y 為 1 的 index set 存成變數 green
# 繪製資料原圖
ax = subplots[0][0] # call 上面的plt.subplots[0][0]
ax.set_title("Original")
ax.scatter(X[red, 0], X[red, 1], c="r")
ax.scatter(X[green, 0], X[green, 1], c="g")
# 隱藏刻度與標籤
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
# 繪製不同 perplexity 下的 t-SNE 分群圖
for i, perplexity in enumerate(perplexities):
    if i<4:
        ax = subplots[0][i+1]
    else: # 第二行之後
        ax = subplots[1][i-4]

    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, init='random',
                         random_state=0, perplexity=perplexity)
    Y = tsne.fit_transform(X)
    t1 = time()
    print("circles, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
    ax.set_title("Perplexity=%d" % perplexity)
    ax.scatter(Y[red, 0], Y[red, 1], c="r")
    ax.scatter(Y[green, 0], Y[green, 1], c="g")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')
    
plt.show()

# 卷積神經網路 (CNN, Convolutional Neural Network)
設計⽬標：影像處理
結構改進：CNN 參考像素遠近省略神經元，並且用影像特徵的平移不變性來共用權重，⼤幅減少了影像計算的負擔
衍生應用：只要符合上述兩種特性的應用，都可以使用 CNN 來計算，例如AlphaGo 的 v18 版的兩個主網路都是 CNN

# 遞歸神經網路 (RNN, Recurrent Neural Network)
設計目標：時序資料處理
結構改進：RNN 雖然看似在 NN 外增加了時序間的橫向傳遞，但實際上還是依照時間遠近省略了部分連結
衍伸應⽤：只要資料是有順序性的應用，都可以使用 RNN 來計算，近年在自然語言處理 (NLP) 上的應用反而成為大宗

# 深度學習 - 巨觀結構
輸入層：輸入資料進入的位置
輸出層：輸出預測值的最後一層
隱藏層：除了上述兩層外，其他層都稱為隱藏層
# 深度學習 - 微觀結構
啟動函數(Activation Function)：位於神經元內部，將上一層神經元的輸入總和，轉換成這⼀個神經元輸出值的函數
損失函數(Loss Function)：定義預測值與實際值的誤差⼤小
倒傳遞(Back-Propagation)：將損失值，轉換成類神經權重更新的⽅法

## 批次⼤⼩越小 : 學習曲線越不穩定、但收斂越快
## 學習速率越⼤ : 學習曲線越不穩定、但收斂越快，但是與批次⼤小不同的是 - 學習速率⼤於一定以上時，有可能不穩定到無法收斂
## 當類神經網路層數不多時，啟動函數 Sigmoid / Tanh 的效果比 Relu 更好
## L1 / L2 正規化在非深度學習上效果較明顯，⽽正規化參數較⼩才有效果

# keras
只要是做分類問題的時候，都要先對 label 做 one-hot encoding
## keras onehot
# 將目標轉為 one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
===========================================================================
from keras.utils import np_utils
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)
## 序列模型
序列模型是多個網路層的線性堆疊
Sequential 是一系列模型的簡單線性疊加，可以在構造函數中傳入一些列的網路層：
from keras.models import Sequential
from keras.layers import Dense, Activation
model = Sequential()
model.add(Dense(32, _input_dim=784))
model.add(Activation(“relu”))

## 常用參數說明
Dense         實現全連接層                       Dense(units, activation, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')
Activation    對上層輸出應用激活函數              Activation(activation)
Dropout       對上層輸出應用dropout以防止過擬和    Dropout(ratio)
Flatten       對上層輸出一維化                    Flatten()
reshape       對上層輸出reshape                  Reshape(target_reshape)

# 流程
equential 序貫模型序貫模型為最簡單的線性、從頭到尾的結構順序，一路路到底Sequential 模型的基本元件
一般需要：
•Model 宣告
•model.add，添加層
•model.compile,模型訓練
•model.fit，模型訓練參數設置 + 訓練
•模型評估
•模型預測

# keras api coding

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

#主要輸入接收新聞標題本身，即一個整數序列（每個整數編碼一個詞）。
#這些整數在1 到10,000 之間（10,000 個詞的詞彙表），且序列長度為100 個詞
#宣告一個 NAME 去定義Input
main_input = Input(shape=(100,), dtype='int32', name='main_input')


# Embedding 層將輸入序列編碼為一個稠密向量的序列，
# 每個向量維度為 512。
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# LSTM 層把向量序列轉換成單個向量，
# 它包含整個序列的上下文信息
lstm_out = LSTM(32)(x)

#插入輔助損失，使得即使在模型主損失很高的情況下，LSTM 層和Embedding 層都能被平穩地訓練
news_output = Dense(1, activation='sigmoid', name='news_out')(lstm_out)

#輔助輸入數據與LSTM 層的輸出連接起來，輸入到模型
import keras
news_input = Input(shape=(5,), name='news_in')
x = keras.layers.concatenate([lstm_out, news_input])


# 堆疊多個全連接網路層
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
#作業解答: 新增兩層
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# 最後添加主要的邏輯回歸層
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

# train

# 宣告 Model API, 採用上述的layer
model = Model(inputs=[main_input, news_input], outputs=[main_output, news_output])

model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'news_out': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'news_out': 0.2})
# 名稱要用上面參數給定的值 但是train model的時候要給變數名稱

# 看模型總
model.summary()


# Multi-layer Perception多層感知
MLP為⼀種監督式學習的演算法
此算法將可以使用非線性近似將資料分類或進行迴歸運算

多層感知機是⼀種前向傳遞類神經網路，至少包含三層結構(輸入層、隱藏層和輸出層)，並且利用到「倒傳遞」的技術達到學習(model learning)的監督式學習

MLP優點：
•有能⼒建立非線性的模型
•可以使⽤$partial_fit$建立real-time模型
MLP缺點：
•擁有⼤於一個區域最⼩值，使用不同的初始權重，會讓驗證時的準確率浮動
•MLP模型需要調整每層神經元數、層數、疊代次數
•對於特徵的預先處理很敏感

# 損失函數
損失函數中的損失就是「實際值和預測值的落差」
損失函數⼤致可分為：分類問題的損失函數和回歸問題的損失函數•Numerical Issues
•在回歸問題稱為「殘差(residual)」
•在分類問題稱為「錯誤率(error rate)
## coding
from keras import losses
model.compile(loss= ‘mean_squared_error‘, optimizer='sgd’)
•其中，包含 y_true， y_pred 的傳遞，函數是表達如下：
keras.losses.mean_squared_error(y_true, y_pred)
# 損失函數-CrossEntropy
要⽤ Cross Entropy 取代 MSE，因為，在梯度下時，Cross Entropy 計算速度較快，
使用時機：
•整數目標：Sparse categorical_crossentropy
•分類目標：categorical_crossentropy
•⼆分類⽬標：binary_crossentropy。
## Keras 上的調⽤方式：
from keras import losses
model.compile(loss= ‘categorical_crossentropy ‘, optimizer='sgd’)
•其中, 包含y_true， y_pred的傳遞, 函數是表達如下：
keras.losses.categorical_crossentropy(y_true, y_pred)
# 損失函數-Hinge Error (hinge)
是⼀種單邊誤差，不考慮負值同樣也有多種變形，squared_hinge、categorical_hinge
## 使用時機：
•適⽤於『⽀援向量機』(SVM)的最⼤間隔分類法(maximum-margin classification)
## Keras 上的調⽤方式：
from keras import losses
model.compile(loss= ‘hinge‘, optimizer='sgd’)
•其中，包含 y_true，y_pred 的傳遞, 函數是表達如下:
keras.losses.hinge(y_true, y_pred)

# 卷積層1+池化層1
model.add(Conv2D(filters=32,kernel_size=(3,3),
                 input_shape=(32, 32,3), 
                 activation='relu', 
                 padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 啟動函數
定義了每個節點（神經元）的輸出和輸入關係的函數為神經元提供規模化非線性化能⼒，讓神經網路具備強⼤的擬合能⼒
## sigmoid
特點是會把輸出限定在 0~1 之間，在 x<0 ，輸出就是 0，在 x>0，輸出就是 1，這樣使得數據在傳遞過程中不容易發散
## cons
⼀是 Sigmoid 容易過飽和，丟失梯度。這樣在反向傳播時，很容易出現梯度消失的情況，導致訓練無法完整
二是 Sigmoid 的輸出均值不是 0原函數及導數圖如下：SigmoidDeriv. Sigmoid

## Softmax
Softmax 把⼀個 k 維的 real value 向量（a1,a2,a3,a4....）映射成⼀個（b1,b2,b3,b4....）
其中 bi 是⼀個 0～1 的常數，輸出神經元之和為 1.0，所以可以拿來做多分類的機率預測
為什麼要取指數？
•第⼀個原因是要模擬 max 的⾏為，所以要讓大的更大。
•第⼆個原因是需要⼀個可導的函數

## note:
二分類問題時 sigmoid 和 softmax 是一樣的，求的都是 cross entropy loss

## Tanh
tanh 讀作 Hyperbolic Tangenttanh 也稱為雙切正切函數，取值範圍為 [-1,1]。
tanh 在特徵相差明顯時的效果會很好，在循環過程中會不斷擴⼤特徵效果

## ReLu (負的不做更新)
修正線性單元（Rectified linear unit，ReLU）
•在 x>0 時導數恆為1
•對於 x<0，其梯度恆為 0，這時候它也會出現飽和的現象，甚至使神經元直接無效，從⽽其權重無法得到更新（在這種情況下通常稱為 dying ReLU）
•Leaky ReLU 和 PReLU 的提出正是為了解決這⼀問題

## Elu
•ELU 函數是針對 ReLU 函數的⼀個改進型，相比於 ReLU 函數，在輸入為負數的情況下，是有⼀定的輸出的這樣可以消除 ReLU 死掉的問題還是有梯度飽和和指數運算的問題
•這樣可以消除 ReLU 死掉的問題
•還是有梯度飽和和指數運算的問題

## PReLU
•參數化修正線性單元（Parameteric Rectified Linear Unit，PReLU）屬於 ReLU 修正類啟動函數的一員。

## Leaky ReLU
•當α=0.1 時，我們叫 PReLU 為Leaky ReLU，算是 PReLU 的⼀種特殊情況
RReLU 以及 Leaky ReLU 有⼀些共同點，即爲負值輸入添加了一個線性項。

## MaxOut
Maxout 是深度學習網路路中的⼀層網路
Maxout 神經元的啟動函數是取得所有這些「函數層」中的最⼤值
優點是計算簡單，不會過飽和
缺點是過程參數相當於多了⼀倍

# 如何選擇正確的啟動函數
1. 根據各個函數的優缺點來配置
如果使⽤ ReLU，要⼩心設置 learning rate(「dead」神經元) 可以用PReLU or Leaky
2. 根據問題的性質
(a) 用於分類器，sigmoid函數及其組合通常效果更好
(b) 梯度消失問題，有時候要避免使用sigmoid, tanh
(c) 神經元Dead問題，使用PReLU
(d) Relu建議使用在隱藏層中
3. 考慮DNN損失函數和啟動函數
•如果使⽤ sigmoid 啟動函數，則交叉熵損失函數⼀般肯定比均⽅差損失函數好
•如果是 DNN ⽤於分類，則⼀般在輸出層使用 softmax 啟動函數
•ReLU 啟動函數對梯度消失問題有一定程度的解決，尤其是在CNN模型中。

# 梯度消失 Vanishing gradient problem
原因：前⾯的層比後⾯的層梯度變化更小，故變化更慢
結果：Output 變化慢 -> Gradient ⼩ -> 學得慢
Sigmoid，Tanh  都有這樣特性不適合⽤在 Layers 多的DNN 架構

# coding
## Sigmoid 數學函數表示方式
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

## Sigmoid 微分
def dsigmoid(x):
    return (x * (1 - x))

## Softmax 數學函數表示方式
def softmax(x):
     return np.exp(x) / float(sum(np.exp(x)))

# 梯度下降Gradient Descent

通過尋找最⼩值，控制⽅差，更新模型參數，最終使模型收斂
•wi+1 = wi - di·ηi,  i=0,1,...
•參數η是學習率。這個參數既可以設置為固定值，也可以⽤一維優化⽅法沿著訓練的⽅向逐步更新計算
•參數的更新分為兩步：第一步計算梯度下降的⽅向，第⼆步計算合適的學習

# SGD
x←x − α ∗dx (x沿負梯度⽅方向下降)
帶 momentum 項的 SGD 
如下形式：
v= ß ∗v − a∗d
x←x+v
其中ß即 momentum 係數，通俗的理解上⾯式子就是，
如果上一次的 momentum（即ß ）與這⼀次的負梯度方向是相同的，那這次下降的幅度就會加大，所以這樣做能夠達到加速收斂的過程
如果上⼀次的 momentum 與這⼀次的負梯度⽅向是相反的，那這次下降的幅度就會縮減，所以這樣做能夠達到減速收斂的過程

## 缺點
梯度下降法的缺點包括：
•靠近極⼩值時速度減慢。
•直線搜索可能會產⽣一些問題。
•可能會「之字型」地下降
avoid local minima
•在訓練神經網絡的時候，通常在訓練剛開始的時候使用較大的learning rate，隨著訓練的進行，我們會慢慢的減小learning rate

# coding
def GD(w_init, df, epochs, lr):    
    """  梯度下降法。給定起始點與目標函數的一階導函數，求在epochs次反覆運算中x的更新值
        :param w_init: w的init value    
        :param df: 目標函數的一階導函數    
        :param epochs: 反覆運算週期    
        :param lr: 學習率    
        :return: x在每次反覆運算後的位置   
     """    
    xs = np.zeros(epochs+1) # 把 "epochs+1" 轉成dtype=np.float32    
    x = w_init    
    xs[0] = x    
    for i in range(epochs):         
        dx = df(x)        
        # v表示x要跨出的幅度        
        v = - dx * lr        
        x += v        
        xs[i+1] = x    
    return xs

def GD_decay(w_init, df, epochs, lr, decay):        
    xs = np.zeros(epochs+1)
    x = w_init
    xs[0] = x
    v = 0
    for i in range(epochs):
        dx = df(x)
        # 學習率衰減 
        lr_i = lr * 1.0 / (1.0 + decay * i) // 每次衰減lr
        # v表示x要改变的幅度
        v = - dx * lr_i
        x += v
        xs[i+1] = x
    return xs

# Backpropagation
誤差反向傳播的簡稱，⼀種與最優化⽅法（如梯度下降法）結合使⽤的該⽅法對網路中所有權重計算損失函數的梯度。這個梯度會反饋給最優化方法，⽤來更新權值以最⼩化損失函數。
反向傳播要求有對每個輸入值想得到的已知輸出，來計算損失函數梯度。
因此，它通常被認為是⼀種監督式學習⽅法，可以對每層疊代計算梯度。反向傳播要求⼈工神經元（或「節點」）的啟動函數可微

## error rate
更改 init data，輸出會有變動，模型的執⾏結果跟預期有落差也是變動，這個落差就是 error rate
•Error rate = (Target 輸出)–(實際輸出) 
•導入 activation function，以 MSE loss function 為例

## 優點
具有任意複雜的模式分類能力和優良的多維函數映射能⼒，解決了簡單感知器不能解決的異常或者⼀些其他的問題。
•從結構上講，BP 神經網路路具有輸入層、隱含層和輸出層。
•從本質上講，BP 算法就是以網路誤差平⽅目標函數、採用梯度下降法來計算⽬標函數的最⼩值。

## 缺點
①學習速度慢，即使是一個簡單的過程，也需要幾百次甚⾄至上千次的學習才能收斂。
②容易陷入局部極⼩值。
③網路層數、神經元個數的選擇沒有相應的理論指導。
④網路推廣能⼒有限。
## 應用
①函數逼近。②模式識別。③分類。④數據壓縮

## steps
第1階段：解函數微分
•每次疊代中的傳播環節包含兩步：
•（前向傳播階段）將訓練輸入送入網路路以獲得啟動響應
•（反向傳播階段）將啟動響應同訓練輸入對應的⽬標輸出求差，從⽽獲得輸出層和隱藏層的響應誤差。
第2階段：權重更新
•Follow Gradient Descent 
•第 1 和第 2 階段可以反覆循環疊代，直到網路對輸入的響應達到滿意的預定的⽬標範圍為止。

# coding
# 定義並建立一神經網路
class mul_layer():
    def _ini_(self):
        self.x = None
        self.y = None
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x*y
        return out
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

# back_propagation
# how much did we miss the target value?    
# l2_error 該值說明了神經網路預測時“丟失”的數目。     
# l2_delta 該值為經確信度加權後的神經網路的誤差，除了確信誤差很小時，它近似等於預測誤差。   
# l1_error 該值為 l2_delta 經 syn1 加權後的結果，從而能夠計算得到中間層/隱層的誤差。   
# l1_delta 該值為經確信度加權後的神經網路 l1 層的誤差，除了確信誤差很小時，它近似等於 l1_error 。


for iter in range(50000):
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0)) # 兩格矩陣相乘
    l2 = nonlin(np.dot(l1,syn1)) 
    
    '''
    新增
    l2_error 該值說明了神經網路預測時“丟失”的數目。
    l2_delta 該值為經確信度加權後的神經網路的誤差，除了確信誤差很小時，它近似等於預測誤差。
    '''
 
    # how much did we miss?
    l2_error = y - l2
    if iter%10000 == 0:
        print(f"第{iter}次", np.mean(l2_error))
    l2_delta = l2_error * nonlin(l2, deriv=True) # error * sigmold diff
    
    l1_error = l2_delta.dot(syn1.T) # 轉置syn1 因為要回推回去
    
    l1_delta = l1_error * nonlin(l1,True)
    
    # update weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
     # syn1 update weights
    
print("Output After Training:")
print(l1)
print("\n\n")

# optimizer
優化算法的功能，是通過改善訓練方式，來最小化(或最⼤化)損失函數 E(x)
優化策略和算法，是⽤來更新和計算影響模型訓練和模型輸出的網絡參數，使其逼近或達到最優值
## 最常⽤的優化算法-Gradient Descent
## 動量Momentum
## SGD-隨機梯度下降法(stochastic gradient decent)
## coding
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
•lr：<float> 學習率。
•Momentum 動量：<float> 參數，⽤於加速 SGD 在相關⽅方向上前進，並抑制震盪。
•Decay(衰變)：<float> 每次參數更新後學習率衰減值。
•nesterov：布爾值。是否使⽤ Nesterov 動量

Mini-batch sizes，簡稱為「batchsizes」，是算法設計中需要調節的參數。
•較小的值讓學習過程收斂更快，但是產⽣生更多噪聲。
•較大的值讓學習過程收斂較慢，但是準確的估計誤差梯度。
•batch size 的默認值最好是 32 盡量選擇 2 的冪次⽅，有利於 GPU 的加速。
•調節 batch size 時，最好觀察模型在不同 batch size 下的訓練時間和驗證誤差的學習曲線。
•調整其他所有超參參數之後再調整 batch size 和學習率。

# Adagrad
對於常⾒的數據給予比較⼩的學習率去調整參數，對於不常⾒的數據給予比較⼤的學習率調整參數
1. 每個參數都有不同的 learning rate
2. 根據之前所有 gradient 的 root mean square 修改
優：減少手動調節
缺：分母不斷積累 收縮漸小

# RMSprop
RMSProp 這種⽅方法是將 Momentum 與 AdaGrad 部分相結合。
RMSprop 是為了解決 Adagrad 學習率急劇下降問題的
抑制梯度的鋸齒下降,RMSProp 不需要⼿動配置學習率超參數

# Adam
Adam，結合 AdaGrad 和 RMSProp 兩種優化算法的優點。
除了像 RMSprop ⼀樣存儲了過去梯度的平方 vt 的指數衰減平均值，也像momentum ⼀樣保持了過去梯度 mt 的指數衰減平均值
• Adam 就是在 RMSprop 的基礎上加了 bias-correction 和momentum
• 隨著梯度變的稀疏，Adam 比 RMSprop 效果會好

# 如何選擇優化器
如果輸入數據及比較稀疏：該使⽤某種⾃適應學習率的⽅法
ex:Adagrad, RMSprop, Adam

如果想使訓練深層網絡模型快速收斂或所構建的神經網絡較為複雜
ex:Adam

# 儲存模型
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# 對optimaizer調整參數
opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# 訓練神經網路的細節與技巧
## what is overfitting
1. 訓練集損失下降遠比驗證及損失來得快
2. 驗證集的損失隨時間增長，反而上升

## coding for test is it overfitting or not
model.fit(x_train, y_train,
epochs=EPOCHS,
batch_size=BATCH_SIZE,
validation_data=(x_valid, y_valid) # 可改成validation_split=0.9 自動切
shuffle=True
)

train_loss = model.history.history["loss"]
val_loss = model.history.history['val_loss']

