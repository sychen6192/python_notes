# 使用於多行的string
string = '''hidalskdjasldsldkfsdflkmsdflkmsdf'''

# 四捨五入 奇進偶退
round(2.5)
round(1.5)

# 字串相加
print("aaa"+"bbb")

# 字碼轉換 ASCII, UNICODE
print(chr(97), ord('紹'))

# 字串前加r = 取消逸出字元的功能
print("可愛的\n昭昭")
print(r"可愛的\n昭昭")

# I/O篇
print(value1, value2, sep='以什麼隔開', end='以什麼結尾', file=sys.stdout(系統標準輸出=>螢幕, flush=False)

#格式化輸出
<不用逗號隔開變數和格式!>
print("...輸出格式區..." %(變數, ...))
print("%d%s%f%x%o" %(.,..,..,.,))

# 精準輸出
print("你不是%s" % "昭昭")
x = 100 
print("x=/%6d/" % x)
y = 10.5
print("y=/%4.3f/" % y)
<重點整理>
m 總位數 包含小數點
n 小數後幾位
/%+-|m.nf/
-為靠左對齊
+為顯示+號

# format用法補充
print("你喜歡{}昭昭".format("大"))
print("你喜歡{}昭昭".format(160))
format似乎不需要管資料型別。

# 開啟檔案I/0
fstream1 = open("C:\\Users\\sychen\\PycharmProjects\\python_note\\out1.txt", mode="w")
print("testing", file=fstream1)
fstream1.close()
fstream2 = open("C:\\Users\\sychen\\PycharmProjects\\python_note\\out1.txt", mode="a")
print("嗨阿昭", file=fstream2)
fstream2.close()
<常用mode>
+r 預設read
+w 寫入
+a addition
+x 開新檔案

# input 資料
name = (input("輸入名字:"))
'str'=type(name)

# 列出內建function
dir(__builtins__)

# list快速賦值
list_test = [1, 2, 3, 4, 5]
list1, list2, list3, list4, list5 = list_test

# list slices
name_list[-n:] # 取得串列後n名
name_list[-1] # 取得  後1名
name_list[start:end:step]

#統計list資料
最大值:max(list_test)
最小值:min(list_test)
總和:sum(list_test)
長度:len(list_test)
append:list_test += [6, 7, 8, 9, 10]
加入末端元素:list.append('addthis')
刪除末端:list.pop()
刪除某資料:list.remove('data')
插入元素:list.insert('index',元素內容)<全部元素往後挪>
反轉串列:list.reverse()
排序串列:list.sort() <小->大>，大到小加入參數list.sort.(reverse=True)
排序串列賦值:list_sorted = sorted(list, reverser=True)
返回index: list.index(1) <第一次搜尋到的值之索引值>
次數: list.count(1)
組合:char.join(seq)
delete: del list_test[1]

# 串列內含串列
num = [1, 2, 3, 4, 5, [6, 7, 8]]
num[5][0]
 
# 串列A.append(串列B)
list_add = [5, 6]
list = [1, 2, 3, 4, [5, 6]]

# 串列A.extend(串列B)
list_add = [5, 6]
list = [1, 2, 3, 4, 5, 6]

# 串列的賦值(記憶體位址複製)
list_dup = list
id(list_dup)會相等於id(list)
=>亦即串列的賦值為記憶體的賦值
所以再之後若只有進行list.append('6')
list_dup 也會隨之改變 

# 串列的拷貝(shallow copy)
list_dup = list[:]
id(list_dup)≠id(list)
所以再之後若只有進行list.append('6')
list_dup 也會不會隨之改變 

***字串單一內容無法更改***
str = "HI"
str[0]="a"
TypeError: 'str' object does not support item assignment

# 字串method
lower()
upper()
title()	 開頭capitalize
rstrip() 刪尾端空白
lstrip() 刪開端空白
strip()  刪兩邊空白
-------------------
len()計算字串長度
max()最大值,unicode為比較基準
min()最小值

# 字串list化

x = list('Zhaozhao')
print(x)
output
>>>['Z', 'h', 'a', 'o', 'z', 'h', 'a', 'o']

# split()處理字串
以字串空格為分隔符號，將字串拆開成串列
str = "Zhaozhao is good"
sList = str.split()
print(sList)
output
>>>['Zhaozhao', 'is', 'good']
len(sList)為3

# in /not in
判斷一物件是否屬於另一物件

# is /not is 
比較兩物件是否相同

# enumerate
enumerate()方法可以將iterable類數值的元素用計數值與元素配對方式回傳
test = ["1", "2", "3"]
enu_list = enumerate(test) # args2=(start=10) 從10開始列舉，預設為0
<注意>enu_list此時仍為物件，需要串列化
print(list(enu_list))

# for的簡短寫法(僅有一行)
for player in players: print(player)

# range()範例
range(start, end, step)
print(list(range(1,5)))
>>[1, 2, 3, 4]

# 進階串列應用
squares = [num ** 2 for num in range(1, n+1)]

# for迴圈 應用於enumerate(列舉)
friends =["zhao", "hong", "demi"]
for ppl in enumerate(friends):
    print(ppl)
for count, ppl in enumerate(friends):
    print(count+1, ppl, sep=".", end=" ")
 >>
(0, 'zhao')
(1, 'hong')
(2, 'demi')
1.zhao 2.hong 3.demi 
*重點:可以用count print出列舉順序

# tuple 元組
->優點:更安全的保護資料(不被更動),增加程式執行速度
元組和串列資料型態結構完全相同
差異:元素值與元素個數不可更動(不可更動的串列)，但可以重新定義
可用:min(tuple), max(tuple), len(tuple)
num_tuple = (元素1, ..., 元素n)

list<=>tuple
tuple(list), list(tuple)

# zip()用法
將相對應的元素打包成tuple後傳給zip物件(所以需使用list具象化)
若放在zip中串列參數不相等，則多出的元素不匹配。
ex:
fields = ["Name", "Age", "height"]
info = ["Zhao", 24, 160]
zipdata= zip(fields, info)
print(list(zipdata))
>>[('Name', 'Zhao'), ('Age', 24), ('height', 160)]

# zip(*zipdata)用法
相當於拆包。
會將上面範例從zipdata拆回fields&info

# 字典dict
資料結構:
"key:value"
name_dict = {key1:value1}

建立空字典: name_dict = {}
複製字典: name_dict_dup = name_dict.copy() <注意此複製記憶體位址不同>

查值:name_dict[key1] >> value1
查key數量:len(name_dict)
驗證元素: key in name_dict >> bool(True, False)

增加:name_dict[key2] = value2
刪值:del name_dict[key1]
刪全部key:name_dict.clear() >>　name_dict = {}
刪整個dict:del name_dict

遍歷字典method:
name_dict = {"Name": " Zhao",
             "Age": 24,
             "height": 160}
print(name_dict.items())

for a, b in name_dict.items():
    print(a, b)
>>
dict_items([('Name', ' Zhao'), ('Age', 24), ('height', 160)])
Name  Zhao
Age 24
height 160

遍歷keys:
name_dict.keys()
遍歷keys+sorted:
sorted(name_dict.keys())
遍歷value:
name_dict.values()
遍歷value且不重複:
set(name_dict.values())

# 字典range()修改
for soldier in armys[35:38]:
	soldier["tag"] = "blue"
	soldier["score"] = 5
	soldier["speed"] = "median"

# 另外，字典內包含串列或是字典是被允許的
範例:字典內字典parse
record = {'shao': {
    'lastname':'陳',
    'firstname': '紹雲',
    'city': 'tpe'},
          'zhao':{
    'lastname':'陳',
    'firstname': '昭昭',
    'city': 'tpe'},
    }
# 雙層迴圈
for account, info in record.items():
    print("使用者帳號:{}".format(account))
    print("姓氏:{}".format(info['lastname']))
    print("名字:{}".format(info['firstname']))
    print("城市:{}".format(info['city']))

# 用串列製作字典
seq1 = ['name', 'city']
list_dict = dict.fromkeys(seq1, 'init') <key:seq1, value:'init'>
print(list_dict)

# get取值
list_dict.get('name')

# setdefault取值
如果有則取值
y = list_dict.setdefault('name1')
如果沒有則自動添加查詢之key, value默認為None
y = list_dict.setdefalut('name2', 'shao')
查key"name2"之value, 若name2不存在則創立一個<name2:shao>

# pop刪值
dict.pop()

# 建立集合(大括號的串列型式)
langs = {'Python', 'C', 'Java'}
->集合會去除重複值
wrong ex:
mixed_set = {1, 'Python', [2, 5, 6]}
why?
因為串列為可變的元素所以不可以當作集合元素所以要用tuple方式來定義

# 建立空字典與空集合
empty_dict = {}
empty_set = set()

# 字串存成集合
x = set("Zhao zhao is good")
print(x)
>>{'a', 's', 'd', 'o', 'i', 'Z', 'g', ' ', 'h', 'z'}

# list去除相同值
x = set(x)
x = list(x)
type(x)>>list

# 集合的操作
交集      &
聯集      |
差集      -
對稱差集  ^
等於     ==
不等於   !=
是成員   in
不是成員 not in

# 集合的method
set.add()
set.copy()
此處的copy為shallow copy 並不是複製他的記憶體位址

set.remove()
set.discard()
set.pop()
上面前兩個都是在刪除集合內的元素
但是remove在沒有值的情況下會返回KeyError,discard則不會
後兩個則是差在傳回值
set.pop()是隨機刪值且set.pop()會傳回那個值，set.discard()都是傳回None

set.clear()
>> set()
--------------------------------
# isdisjoint()
如果兩個集合沒有共同元素則傳回True, else False

# issubset()
  A.issubset(B) # A是B的子集則回傳True
  A.issuperset(B) # A是B的父集合回傳True

# intersection_update()
  A.intersection_update(*B) # 找交集賦值給A
# difference_update()
  A.difference_update(B) # 將交集的元素刪掉之後賦值給A
# symmetric_difference_update()
  A.symmetric_difference_update(B) #對稱差集後賦值給A

# update()
  A.update(B) # 將集合B加到集合A

# frozenset
->frozenset是不可變集合。
就像是tuple是不可變串列
不可以用add(), remove() 來更動凍結集合的內容

# function預設參數
 def foo(name, name1 = 'Hong'):
    print(name1 +' is '+ name)

**預設參數要放在後面(name1 = 'Hong')

# 回傳多筆值 EX
def multi(x1, x2):
    add = x1 + x2
    sub = x1 - x2
    mul = x1 * x2
    div = x1 / x2
    return add, sub, mul, div
x1 = x2 = 10
add, sub, mul, div = multi(x1, x2)
print(add, sub, mul, div)

# 副本串列複製放進函式中
kitchen(list[:])

# 基本傳遞處理任意數量的參數
def dress(*clothes):
    print("幫昭昭穿上")
    for cloth in clothes:
        print("---", cloth)

dress('衣服','褲子', '襪子')
>>>
幫昭昭穿上
--- 衣服
--- 褲子
--- 襪子

*需先傳遞一般參數後再傳遞任意數量參數
* 傳入參數時需定義參數

# 任意數量的關鍵字參數
def dress(**clothes):
    print("幫昭昭穿上")
    for color, cloth in clothes.items():
        print("---", color, cloth)

dress(衣服 = '紅色', 褲子 = '藍色', 襪子 = '白色')
>>>
幫昭昭穿上
--- 衣服 紅色
--- 褲子 藍色
--- 襪子 白色

* 傳入參數時需定義參數以及值


# 匿名函數lambda
lambda arg1[, arg2,...argn]:expression
ex:
cal = lambda x, y : x * y
print(cal(10, 10))>>100

# filter(function, iterable())
def find_odd(x):
	return x if (x%2==1) else return None
list = [....]
obj = filter(find_odd, list)
res = [item for item in obj]
print(res)
find_odd:用做是篩選的函式
list:是欲篩選的元素串列
obj:是篩選後的結果
res:將obj串列化

*** Filter客家寫法(篩選奇數)
print(list(filter(lambda x: (x % 2 == 1), [1, 2, 3, 4, 5, 6])))
>>[1, 3, 5]


# map(function, iterable())
print(list(map(lambda x: x ** 2, [1, 2, 3, 4, 5, 6])))
>>[1, 4, 9, 16, 25, 36]

-----------------------------------------------------
OO in python

# private 概念
>> 加上雙底線

class Banks():

    def __init__(self, uname):
        self.__name = uname
        self.__balance = 0
        self.__title = 'Taiwan Bank'
        self.__rate = 30
        self.__service_charge = 0.01

    def save_money(self, money):
        self.balance += money
        print("存款", money, "完成")

    def withdraw_money(self, money):
        self.balance -= money
        print("提款", money, "完成")

    def get_balance(self):
        print(self.__name.title()+"現在戶頭有:"+str(self.__balance)+'元')

    def usd_to_twd(self, amount):
        self.result = self.__cal_value(amount)
        return self.result

    # 私有函數
    def __cal_value(self, amount):
        return int(self.__rate*amount*(1-self.__service_charge))


zhao_account = Banks('zhao')
money = 50
print(zhao_account.usd_to_twd(money))
# 下面會失敗
print(zhao_account.__cal_value(money))

# 衍生類別與基底類別有相同名稱的方法:多型

# 衍生類別引用基底類別的方法
super().__init__(arg1, ...)

# 三代同堂的類別與取得基底類別的屬性
連續繼承只需要在各子類別的__init__中加入super().__init__()
即可從最下面呼叫最上面之method

# type(物件)
可了解某個物件變數的資料類型或是所屬類別關係
# isinstance()
語法:isinstance(物件, 類別) <boolean>
繼承後的類別其物件也是 父類別的物件 (True)

-----特殊屬性-----

# __doc__
列印出函式或是類別的註解
print(function.__doc__)

# __name__

if __name__ == '__main__':
	doSomeThing()
**程式若是自己執行則 -> __name__ = __main__

總結: __name__是可以判別這個程式是自己執行還是被其他程式import當作模組使用。

# __str__
class Name:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return '%s' % self.name

shao = Name('shao')
print(shao)

**將物件的輸出轉換成string

＃ __repr__()
如果是在python shell視窗輸入類別變數，系統是呼叫__repr__()做回應
加在
def __str__(self):
	return ....
__repr__ = __str__             <<<<加這行

# import的方法
假設已有模組建立在同一資料夾,為myfun.py

method 1:
import myfun

myfun.dress('黑色衣服', '排球短褲', '夾腳拖')
myfun.drink('大杯', '珍珠鐵觀音奶茶')

method 2:
from myfun import dress, drink

dress('黑色衣服', '排球短褲', '夾腳拖')
drink('大杯', '珍珠鐵觀音奶茶')

method 3:
from myfun import *
>>也可以，但不夠明確不曉得自己import了什麼函數，其餘同上

method 4:
from myfun import dress as 穿穿, drink as 喝喝

穿穿('黑色衣服', '排球短褲', '夾腳拖')
喝喝('大杯', '珍珠鐵觀音奶茶')

(似乎支援Non-ASCII CHARCTER 的命名方式，但不建議)

# import class
from myfun import Banks, Banciao_Banks
其餘將類別當作函式操作即可
你也可以將繼承的父類別獨立模組後 import給子類別用

# random module
random.randint(min, max)
在min~max間抽取一個數字

random.choice(list)
在List中隨機傳回一個元素

random.shuffle(list)
打亂List中的元素

# time module
time.time()
傳回自1970/01/01 00:00:00AM 以來的秒數

time.sleep(s)
工作暫停s秒

time.asctime()
列出目前系統時間

time.localtime()
返回目前的時間結構資料，可利用索引來獲得個別內容

# sys module
這個模組主要控制python shell 視窗訊息

sys.version
列出版本信息

sys.stdin.readline()
讀取螢幕輸出，可用做賦值，賦值後屬性應該是字串
(readline(char_number)，readline內參數用作讀取多少個字元數)

sys.stdout.write("love zhao")
(write參數必須是str)

# keyword module
keyword.kwlist
列出Python所有關鍵字

keyword.iskeyword()
傳入字串是否為關鍵字?  >> boolean
keyword.iskeyword('def')
>>True

# 資料夾與檔案路徑
相對路徑
"."  目前資料夾
".." 上一層資料夾

".\shao.py" = "shao.py"
目前資料夾可省略

# os module

os.getcwd()
取得目前工作目錄

os.path.abspath()
取得絕對路徑，通常用此辦法將檔案或資料夾相對路徑轉為絕對路徑
os.path.abspath('.')
os.path.abspath('..')

os.path.relpath()
傳回從start到path的相對路徑
os.path.relpath(path, start)

---以下為檢查路徑的方法 exists/isabs/isdir/isfile ---
os.path.exists()
os.path.isabs()
os.path.isdir()
os.path.isfile()

---檔案與目錄的操作 mkdir/rmdir/remove/chdir ---
*此處模組在os底下

os.mkdir()
os.rmdir() -> 這個是刪資料夾的
os.remove() -> 這個是刪檔案的
os.chdir() 改變目前工作資料夾

os.path.join()
此方法可以將參數內的字串結合為一個檔案路徑
ex:
os.path.join(os.path.abspath('.'), 'test'

os.path.getsize()
獲得特定檔案大小(位元組)

os.listdir()
以串列方式列出特定工作目錄內容

(glob module)
glob.glob('*.py')
在當前工作目錄下找副檔名為py的檔案並用串列表示

os.walk('資料夾')
遍歷目錄樹
for dirName, sub_dirNames, filenames in os.walk('oswalk'):
    print(dirName)
    print(sub_dirNames)
    print(filenames)

# 讀取檔案範例1
fn = 'out1.txt'
file_obj = open(fn) #傳回檔案物件
data = file_obj.read()
file_obj.close()
print(data)

# 讀取檔案範例2(不需close)
with open(fn) as file_obj:
    print(file_obj.read())

# 逐行讀取 (使用範例二)
with open(fn) as file_obj:
	for line in file_obj:
		print(line)

# 逐行讀取 BY readlines()
obj_list = file_obj.readlines()
逐行讀取放到串列中

# 字串尋找取代
string = string.replace(舊字串, 新字串)

# 數據搜尋
if keyword in string_obj:

# 數據比對回傳索引值
index = str.find('昭昭')
若沒有找到 回傳-1

# 寫入檔案
with open("file", 'w') as file_obj:
	file_obj.write('寫字進去')
*註:無法寫入數值資料，需轉換成字串

# 寫入檔案(附加)
with open('out1.txt', 'a') as file_obj:
    file_obj.write('second\n')
*\n攸關到下一次輸出位置

# shutil module
主要為執行檔案或目錄的複製、刪除、更動位置、更改名字

# shutil.copy()
檔案複製
shutil.copy(source, destination)

# shutil.copytree('dir', 'des-dir')
包含目錄底下的子目錄或檔案也一起被複製

# shutil.move(source, destination)
用作檔案移動或是更名，目錄的移動或更名

# shutil.rmtree()
可用作刪除裡面含有資料的目錄(os.rmdir()則只能刪除空目錄)

#zipfile()

寫入zipfile
fileZip = zipfile.ZipFile('out.zip', 'w')

import zipfile
import glob
fileZip = zipfile.ZipFile('out.zip', 'w')
for file in glob.glob('oswalk/*'): #遍歷目錄下
    fileZip.write(file, os.path.basename(file), zipfile.ZIP_DEFLATED)
fileZip.close()
*註:
basename()   用于去掉目錄的路徑，只返回文件名
>>> import os

>>> os.path.basename('d:\\library\\book.txt')
'book.txt'

讀取zipfile
fileZip Info = zipfile.ZipFile('out.zip', 'r')
*以列表列出所有壓縮檔案
fileZipInfo.namelist()
*列出檔案資料
for info in fileZipInfo.infolist():
	print(info.filename, info.file_size, info.compress_size)

解壓縮zipfile
fileUnZip = zipfile.ZipFile('out.zip')
fileUnZip.extractall('hi') # 解壓縮結果存入hi目錄
fileUnZip.close()

# 編碼格式encoding
範例:
file_obj = open("out.txt", encoding="utf-8(cp950)", mode = "r")

# 認識utf-8的BOM
使用WINDOWS作業系統的記事本utf-8執行編碼時，作業系統會在文件前端增加"位元組順序記號"簡稱BOM
(用作判斷文字是否以Unicode表示)
範例
['\ufeffpython語言\n', '王者歸來']
\ufeff u代表是Unicode編碼格式 fe ff 是16進位編碼格式

with open(".txt", encoding = "utf-8-sig") as file_obj:
(utf-8-sig 可去除\ufeff)

# pyperclip module
剪貼簿的複製貼上應用

pyperclip.copy('昭昭')
pyperclip.paste()

===== debug & logging =====

# try - except
語法:
try:
	指令
except 異常物件:
	異常處理程序
# try - except - else
語法:
try:
	指令
except 異常物件:
	異常處理程序
else:
	正確處理程序

# 分析單一文件字數
key:
data_list = data.split()
print(len(data_list))

# 常見的異常處理物件
AttributeError 通常是指物件沒有這個屬性
Exception 一般錯誤皆可使用
FileNotFoundError 找不到open()這個檔案
IOError 再輸入或輸出時發生錯誤
IndexError 索引超出範圍區間
KeyError 在映射中沒有這個鍵
MemoryError 需求記憶體空間超出範圍
NameError 物件名稱未宣告
SyntaxError 語法錯誤
SystemError 直譯器的系統錯誤
TypeError 資料型別錯誤
ValueError 傳入無效參數
ZeroDivisionError 除數為0

# 多個例外
try:
	...
except ZeroDivisionError:
	...
except TypeError:
	...

# 多個例外包在一起
try:
	...
except (arg1, arg2 ...):
	...

# 使用python內建錯誤訊息
except Exception as e:
    print(e)

# except all 捕捉所有異常

try:
	...
except:
	...

# 丟出異常
raise Exception('密碼長度不足')
這時原先Exception()內的字串 會透過
"except Exception as err:"
傳給err

# 紀錄traceback字串
traceback.format_exc()
<發生錯誤時直接用在except裡面>
可搭配with open('.txt') as file_obj做log使用

# finally
finally為搭配try使用,必須放在except和else之後
無論如何都會執行finally內的程式碼!!

# 程式斷言assert
語法:assert 條件, '字串'
如果條件符合則繼續執行下去，若不符合則顯示字串至traceback以及拋出異常
(程式斷言一般用在開發階段)

# logging 程式日誌
logging分為五個等級 從最低到最高如下

DEBUG     logging.debug()
INFO      logging.info()
WARNING   logging.warning()
ERROR     logging.error()
CRITICAL  logging.critical()

可使用下列函數設定顯示資訊的等級:
logging.basicConfig(level=logging.DEBUG)
意即此等級或更高的等級logging會被顯示

可使用下列函數設定輸出FORMAT:
logging.basicConfig(level=logging.CRITICAL, format='')
>>沒有前導輸出訊息
CRITICAL:root:critical -> critical

                    制定level等級					時間-level:括號內容
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s : %(message)s')

# 停用程式日誌logging

logging.disable(level)
ex:
logging.disable(logging.CRITICAL)

# 正規表達式

Regex
PhoneRule = re.compile(r'\d\d\d\d-\d\d\d-\d\d\d')  # 建立物件
pattern 可改成:r'\d{4}-\d{3}-\d{3}'

# re_obj.search()返回第一個符合值
example:
PhoneRule = re.compile(r'\d\d\d\d-\d\d\d-\d\d\d')
phoneNum = PhoneRule.search(str)
print(phoneNum)
>>電話號碼是:<re.Match object; span=(0, 12), match='0926-577-858'>
若要獲取字串0926-577-858則需使用
phoneNum.group()
使用group()傳回比對符合的不同分組 例如:group()或group(0)傳回第一個符合的文字
group(1)則傳回括號的第一組文字
group(2)則傳回括號的第二組文字
指的是pattern = r'(\d{2})-(\d{8})'
				  ^第一組  ^第二組

# re_obj.findall() 返回所有符合值成串列

PhoneRule = re.compile(r'\d\d\d\d-\d\d\d-\d\d\d')
phoneNum = PhoneRule.search(str)
直接賦值給phoneNum為串列

**以上為利用re.complie()建立物件後操作**

以下可省略使用
re.search(pattern, string, flags)
re.findall(pattern, string, flags)

re.search後
要用group來print出找到的值
需用賦值後的phoneNum.group()返回第一個符合
phoneNum.groups() 所以小括號內的值已串列返回

# escape符號的應用
使用escape來跳出符號(跟group的括號作區別)
ex:
\(02\)8261-8100

# 多個pattern BY PIPE| 多個分組的管道搜尋
pattern = '昭|紹' # pipe間不可留有空白
可以使用括號Pipe
但是re.findall()後僅會print出pipe
pattern = '子(昭|紹)'
input < 子昭
output >> 昭

# 使用問號?做搜尋
表示可有可無
語法: (na)? <括號起來可有可無的東西>

# 使用*號做搜尋
表示某字串可以從0到多次
語法: (na)*

# 使用+號做搜尋
表示某字串可以從1到多次
語法: (na)+

# 搜尋時忽略大小寫

re.findall(pattern, msg, re.I)

# 大括號設定比對次數
pattern = '(昭){3,5}' # 昭出現三到五次才返回找到

(如果想設定三次以上 則可以使用{3,})

# 貪婪與非貪婪搜尋(python預設為greedy)
這邊直接帶例子
import re


def searchStr(pattern, msg):
    txt = re.search(pattern, msg)
    if txt:
        print('搜尋成功', txt.group())
    else:
        print("not Found")


searchStr("(昭){3,5}", "昭昭昭昭昭不要")
>>搜尋成功 昭昭昭昭昭
searchStr("(昭){3,5}?", "昭昭昭昭昭不要")
>>搜尋成功 昭昭昭
*PATTERN 後面加一個問號 搜尋到3個即返回結果

# 特殊字元

=====特殊字元表=====
字元
\d  0-9之間的整數字元
\D  \d以外的其他字元
\s  空白、定位、Tab鍵、換行、換頁字元
\S  \s以外的其他字元
\w  數字、字母和底線_字元，[A-Za-z0-9_]
\W  \w以外字元

# 字元分類

[a-z]:代表小寫字元
[A-Z]:代表大寫字元
[2-5]:代表2-5數字
另,[2-5.]:代表2-5數字以及.數字

*差異: 一個有中括弧一個沒有，如果要做單純的pattern比對則不需要括弧
[aeiou] 則是要找含有這些字元的字串

# 字元分類的^字元
搜尋不再這些字元內的字元(反向搜尋)

# 正規表示法的^字元
以什麼為首的字串

# 正規表示法的$字元
ex
pattern = '\d$' # 測試最後的字元是數字

# 搜尋開始到結束皆是數字的字串
pattern = '^\d$'
(所以只能有一個字串!)

# 單一字元使用萬用字元"."
萬用字元"."表示可以搜尋除了<換行字元>以外的所有字元，但僅限定一個字元
*註:若是真正的"."字元，需escape(\.)

# 換行字元的處理
通常".*"的搜尋碰上換行字元就停止，所以需要FLAG ARGS 要加上"re.DOTALL"
ex:
txt = re.search(pattern, msg, re.DOTALL)

# re.match()
re.match()與re.search()差在只比對字串開頭，失敗即傳回None

# MatchObject 幾個重要的方法
當使用re.match()或re.search()搜尋成功時，會產生MatchObject物件

方法如下
group()	可傳回搜尋到的字串
end()	可傳回搜尋到的字串的結束位置
start()	可傳回搜尋到的字串的起始位置
span()	可傳回搜尋到的字串的(起始,結束)位置

# re.sub()

語法: result = re.sub(pattern, newstr, msg) # msg是整個欲處理的字串
搜尋成功-> 用newstr取代，回傳結果給result(result是替代後的結果)
搜尋失敗-> 則將msg內容回傳給result變數，msg內容不改變

ex:
def searchStr(pattern, newstr, msg):
    txt = re.sub(pattern, newstr, msg)
    if txt:
        print('取代成功:', txt)
    else:
        print("not Found")


msg = '同學陳吳子昭給同學黃慶瑄一本書要同學李承鴻轉交給同學陳紹雲'
pattern = r'同學\w{2}(\w)'
newstr = r'**\1'  # 這邊斜線1必須防止溢出
searchStr(pattern, newstr, msg)	

# 整理re.IGNORECASE/re.DOTALL/re.VERBOSE
		無視大小寫/換行的也印出/可使用註解
若要同時使用可以用pipeline

# 讀取Word檔案內容
===建立物件===
wdoc = docx.Document('檔案名稱') # 建立docx物件wdoc
===獲得Paragragh和Run數量===
len(wdoc.paragraghs) # wdoc是上面建立的物件
len(wdoc.paragraphs[n].runs) # n是第幾段或稱Paragraph編號
===列出Paragraph內容===(第n段Paragraph內容)
print(wdoc.paragraph[n].text)
===列出Paragragh內的Run內容===(第n段Paragraph第m個Run內容)
print(wdoc.paragraph[n].runs[m].text)

# 儲存檔案
wdoc.save('word.docx')

# 建立標題
wdoc.add_heading('我的標題', level=n)

# 建立段落
ptr = wdoc.add_paragrapg('') # 段落物件

# 建立RUN
ptr.add_run('') # 用段落物件建立Run

# 強制換頁輸出
wdoc.add_page_break()

# 插入照片
wdoc.add_picture('img_file', width=Cm(), height=Cm())

# 建立表格
table = wdoc.add_table(rows=2, cols=2)
table.rows[0].cells[0].text = '項目' # 表格操作 
table.style = 'LightShading-Accent1' # 表格樣式

# 計算表格rows和cols長度
len(table.rows)
len(table.columns)

# 列印表格
for row in table.rows:
	for cell in table.cells:
		print(cell.text)

# 增加表格
new_row = table.add_row()
new_row.cells[0].text = ''
new_row.cells[1].text = ''

# Paragragph樣式
wdoc. = docx.Document()
wdoc.add_paragraph('昭昭', style='ListNumber')
wdoc.add_paragraph('昭昭', style='ListBullet')

# Run樣式
wdoc = docx.Document()
ptr = wdoc.add_paragraph('我是第一段落')
run1 = ptr.add_run('我是粗體')
run1.bold = True  <<<
run2 = ptr.add_run('我是斜體')
run2.italic = True  <<<
wdoc.save('')

# PDF操作
import PyPDF2

# 建立PDF物件
pdf_obj = open('QA.pdf', 'rb') # 以二進位方式開啟
pdfRD = PyPDF2.PdfFileReader(pdf_obj) # 建立物件
print("PDF頁數是: ", pdfRD.numPages)

# 讀取PDF內容
page_obj = pdfRD.getPage(n) # 讀取第n頁內容
txt = page_obj.extractText() # 提取第n頁文字
print(txt) # 亂碼 中文不支援

# 測試PDF物件是否加密
if pdfRD.isEncrypted: # 用前面建立的物件屬性判斷是否加密
    print("檔案有加密")
else:
    print("檔案沒有加密")


# PDF解密
*注意:這邊只有對特定物件解密,下面只有對pdfRd物件解密
pdfRd.decrypt('password')

# 建立新的PDF檔案
1.建立一個PdfWr物件
PdfWr = PyPDF2.PdfFileWriter()
2.將已有PdfRd物件一次一頁複製到pdfWr物件
PdfWr.addPage(pdfRd.getPage(0))
3.使用write()方法將pdfWriter物件寫入PDF檔案
PdfOutFile = open('out.pdf', 'wb') # 此處為'wb'
PdfWr.write(pdfOutFile)
pdfOutFile.close()

# 旋轉頁面
pdf_obj = open('*.pdf', 'rb')
pdfRd = PyPDF2.PdfFileReader(pdf_obj)
page1 = pdfRd.getPage(0)
page1 = page1.rotateClockwise(90)

# 加密pdf檔案
建立
PdfWr = PyPDF2.PdfFileWriter()
PdfWr.addPage('')
此處已經加完密
PdfWr.encrypt('password') # 加密
Output = open('out.pdf', 'wb')
PdfWr.write(Output) # 將前面建立的pdf物件 加入到上面開啟的pdf物件
Output.close() # 跟上面Open是成對的 

# merge pdf

# 開啟
pdf_obj1 = open('test.pdf', 'rb')
pdfRdSSE = PyPDF2.PdfFileReader(pdf_obj1)
SSEpage = pdfRdSSE.getPage(0)

# 開啟
pdf_obj2 = open('watermark.pdf', 'rb')
pdfRdWat = PyPDF2.PdfFileReader(pdf_obj2)
WATpage = pdfRdWat.getPage(0)

# 融合
SSEpage.mergePage(WATpage)

PdfWr = PyPDF2.PdfFileWriter()
PdfWr.addPage(SSEpage)
output = open('test.pdf', 'wb')
PdfWr.write(output)
output.close()

# openpyxl 範例

import openpyxl

fn = 'sales.xlsx'
# 開啟檔案
wb = openpyxl.load_workbook(fn)

# 取得工作表名稱
allsheet = wb.get_sheet_names()

# 取得目前工作的工作表
ws = wb.get_active_sheet()

# 設定目前工作的工作表
ws = wb.get_sheet_by_name('2020Q1')

# 取得工作表內容
print("儲存格A1 = ", ws['A1'].value)
print("儲存格A2 = ", ws['A2'].value)
print("儲存格C5 = ", ws['C5'].value)


# 列出儲存格位置
print("儲存格C5 = ", ws['A5'].column, ws['A5'].row, ws['A5'].coordinate) 
^^^ 這邊做出來的結果是 儲存格C5 =  1 5 A5 (??)

# 取得工作表內容欄位數和行數
print("工作表欄位和行數 = ", ws.max_column, ws.max_row)

# 利用cell取得儲存格內容
for i in range(1, ws.max_column+1):
    print(ws.cell(column=i, row=5).value, end=' ')

 >> 李連杰 8864 6799 7842 =SUM(B5:D5)

可以看到這邊顯示的是資料表的函數，若是要選擇輸出資料則需要在
開啟檔案那邊加上args data_only=True
wb = openpyxl.load_workbook(fn, data_only=True)

# 我們來列出A4:E6內容看看
for j in range(4, 7):
    for i in range(1, 6):
        print("%5s" % ws.cell(column=i, row=j).value, end=' ')
    print() # 換行

# 工作表物件ws的rows和columns
當建立工作表物件ws成功後，會自動產生下列數據產生器(generators)
ws.rows:工作表數據產生器以行方式包裹，每一行用一個Tuple包裹
ws.columns:工作表數據產生器以欄方式包裹，每一行用一個Tuple包裹
type(ws.rows)
>> <class 'generator'>
(由於ws.rows是數據產生器,若是要取得他們的內容則需要先將他們轉成串列，然後再用索引的方式取得資料)

example:

for item in list(ws.columns)[0]:
    print(item.value)
for item in list(ws.rows)[2]:
    print(item.value, end=' ')

逐行獲得工作表所有內容:
for row in ws.rows:
    for cell in row:
        print(cell.value, end=' ')
    print()

逐欄獲得工作表所有內容:
for col in ws.columns:
    for cell in col:
        print(cell.value, end=' ')
    print()

# 如何用整數取代欄位名稱
先import
from openpyxl.utils import get_column_letter, column_index_from_string

兩個轉換的方法
get_column_letter(數值)
column_index_from_string(字母)

範例:
print(get_column_letter(1)) >> 'A'
print(get_column_letter(5)) >> 'E'
print(column_index_from_string('E'))  >> 5
print(column_index_from_string('A'))  >> 1

# # 我們用另外一個方法來列出A4:E6內容看看(elegant way)
for row in ws['A4':'E6']:
    for cell in row:
        print(cell.value, end=' ')
    print()
==================================================
# 寫入Excel檔案

# 建立空白活頁簿
wb = openpyxl.Workbook()

# 獲得目前工作表
ws = wb.get_active_sheet()

print("目前工作表名稱 = ", ws.title)
ws.title = 'My sheet'
print("新工作表名稱 = ", ws.title)

# 儲存excel
wb.save('out0828.xlsx')

# 複製Excel檔案
wb = openpyxl.load_workbook('sales.xlsx')
wb.save('sales_copy.xlsx')

# 建立工作表
wb.create_sheet(index=0, title = 'first_sheet')
wb.create_sheet(index=1, title = 'second_sheet')

# 刪除工作表
wb.remove_sheet(wb.get_sheet_by_name(u'sheet_name'))  # 這邊必須用get_sheet_by_name 而不是名字

# 寫入儲存格
wb = openpyxl.Workbook()
ws = wb.get_active_sheet()
ws['A1'] = 'Python'
ws['A2'] = '1000'
wb.save('output.xlsx')

# 串列寫入儲存格
row1 = ['昭昭', '紹紹', '鴻', 'Demi']
ws.append(row1)
row2 = ['1000', '1000', '800', '60']
ws.append(row2)
* 一行一行寫，或是用串列中串列 使用for迴圈寫入

# excel 字型
from openpyxl.styles import Font # 導入模組

font_mod = Font(name='微軟正黑體', size=24, bold=True, italic=True, color)  # 樣式設定
ws['A1'].font = font_mod  # 套用設定

# 使用EXCEL函式
例:A1-A4加總
ws['A5'] = '=SUM(A1:A4)'

# 設定EXCEL的高度和寬度
from openpyxl.styles import Alignment

ws.row_dimensions[1].height = 40
ws.column_dimensions['A'].width = 20

# 儲存格對齊
ws['A1'].alignment = Alignment(horizontal='center', vertical='center')
horizontal = 'left';'center';'right'
vertical = 'top';'center';'bottom'

# 合併儲存格
ws.merge_cells('A1:E1')  # 合併A1:E1的儲存格

# 取消合併
ws.unmerge_cells('A1:E1') # 沒測試成功

# 另一個get active方法
ws = wb.active

# 建立 BarChart

ex:
wb = openpyxl.Workbook()
ws = wb.active

rows = [
    ['', '2020年', '2021年'],
    ['亞洲', 100, 300],
    ['歐洲', 400, 600],
    ['美洲', 500, 700],
    ['非洲', 200, 100]]

for row in rows:
    ws.append(row)

chart = BarChart()
chart.title = '深石軟件銷售表'
chart.y_axis.title = '業績金額' 
chart.x_axis.title = '地區'

data = Reference(ws, min_col=2, max_col=3, min_row=1, max_row=5) # B2:C5的ws
chart.add_data(data, titles_from_data=True) # 把data加入
xtitle = Reference(ws, min_col=1, min_row=2, max_row=5) # A2:A5的ws
chart.set_categories(xtitle) # 下面的分類
ws.add_chart(chart, 'E1') # 放圖表
wb.save('test.xlsx')

# 建立3D BarChart
chart = BarChart3D()

# 建立PieChart
chart = PieChart()

# 建立PieChart3D
chart = PieChart3D()

==========CSV文件==========

CSV = Comma-Separated Values (逗號分隔值)

# 開啟csv檔案
with open('file.csv') as csv_file:
    csvReader = csv.reader(csv_file) # 建立Reader物件
    listReport = list(csvReader) # 將資料轉成串列

# 迴圈reader物件資料
with open('file.csv') as csv_file:
    csvReader = csv.reader(csv_file)
    for row in csvReader:
        print("Row %s = " % csvReader.line_num, row) # csvReader.line_num 行數

* 也可以用把資料轉成串列後，用迴圈print出

# 用串列讀取csv內容(需先轉成串列)
list = [[1,2,3],[4,5,6]]
如果想讀1 => list[0][0]

# DictReader()

with open('file.csv') as csv_file:
    csvReader = csv.DictReader(csv_file)
    for row in csvReader:
        print(row)
>>
OrderedDict([('first_name', 'Eli'), ('last_name', 'Manning'), ('city', 'New York')])
OrderedDict([('first_name', 'Kevin '), ('last_name', 'James'), ('city', 'Cleveland')])
OrderedDict([('first_name', 'Mike'), ('last_name', 'Jordon'), ('city', 'Chicago')])

# 用key-value方式獲得
import csv

with open('csvPeople.csv') as csv_file:
    csvReader = csv.DictReader(csv_file)
    for row in csvReader:
        print(row['first_name'], row['last_name'])


# 寫入csv檔案

法1:
csvFile = open('檔案名稱', 'w', newline='') # 'w'是write only模式
...
csvFile.close()

法2:
with open('檔案名稱', 'w', newline='') as csv_file:
    ...

# 建立writer物件

with open('檔案名稱', 'w', newline='') as csv_file:  # newline = '' 避免輸出時每行之間多空一行。
    outWriter = csv.writer(csv_file)

# 輸出串列 writerow()
with open('csvPeople.csv', 'w', newline = '') as csv_file:
    csvWriter = csv.writer(csv_file) # 建立寫入物件
    csvWriter.writerow(['名字', '年齡', '城市'])

# 複製csv檔案
範例:
import csv

with open('csvPeople.csv') as csv_file:
    csvReader = csv.reader(csv_file)
    csvReport = list(csvReader)

with open('csvReport.csv', 'w', newline='') as csv_file1:
    csvWriter = csv.writer(csv_file1)
    for row in csvReport:
        csvWriter.writerow(row)

*大致上說起來須建立一個reader物件，來將csv內容串列化
再new一格writer物件，將串列用迴圈的方式使用csvWriter.writerow()依序寫入

# delimiter關鍵字
delimiter是分隔符號，用在writer的方法內，本來是逗號，可以改成別的
ex:
csvWriter = csv.writer(csv_file, delimiter=\t)
逗號改成定位點字元

# 寫入字典資料 DictWriter()
fields = ['名字', '年齡', '城市']
dictWriter = csv.DictWriter(csv_File, fieldnames=fields)
dictWriter.header() # 設定上面的fieldname
dictWriter.writerow({dict.....})

*或是有dictList(定義串列，元素為字典)
用for迴圈來遍歷list裡的元素，個別用dictWriter.writerow()

# webbrowser module
webbrowser.open('url')

# requests module
htmlfile = requests.get(url) # 產生response物件

# 操作response物件
htmlfile.status_code # 取得網頁內容是否成功代碼(requests.codes.ok=200)
htmlfile.text # 全文(可搭配len使用)

*可利用status_code + re.findall(pattern, htmlfile.text)，來找出pattern出現次數

# raise_for_status()
可以處理網址正確但後面附加檔案錯誤的問題
try:
    htmlfile.raise_for_status()
except Exception as err:
    ...

#  406 Client Error: Not Acceptable for url
406錯誤就是網頁伺服器阻擋
solution:
1.加入headers
headers = {'User-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64)\
            AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101\
            Safari/537.36', }
2.再requests.get(url, headers=headers) 加入參數

# 儲存下載的網頁
with open(fn, 'wb') as file_obj:    # 以二進位儲存
    for diskStorage in htmlfile.iter_content(10240):    # Response物件處理
        size = file_obj.write(diskStorage)  # Response物件寫入
        print(size)
    print('以 %s 儲存網頁html檔案成功' % fn)

# BeautifulSoup module
import bs4,requests

html_file = requests.get(url)
soup_obj = bs4.BeautifulSoup(html_file.text, 'lxml')

*後表明的是解析html的方法，其有三
'html.parser'
'lxml'
'html5lib'

# 爬蟲亂碼解決
解法:response_obj.encoding = 'utf-8'
html_file = requests.get(url)
html_file.encoding = 'utf-8'

# 得到title標籤的內容
承上
soup = bs4.BeautufulSoup(html_file.text, 'lxml')
print(soup.title)
*若要去除標籤
soup.title.text

# 傳回第一個找到的標籤
(此例從soup物件以建立後開始)
obj_tag = soup.find('h1')   # 尋找h1標籤
print(obj_tag) # 可搭配.text去除標籤

# 傳回所有符合的標籤(傳回串列)
objTag = soup.find_all('h1')
for item in objTag:
    print(item.text)

# html元素內容屬性與getText()

<p>Marching onto the path of <b>Web Design Expert</b></p>
分三種:
outerHTML
<p>innerHTML</p>
    <b>textContent</b>

這邊使用getText()則是取得textContent
for i in range(len(objTag)):
    print(objTag[i].getText())

# Select() 主要是以css選擇器來尋找元素(傳回串列)
html_file = requests.get(url)
obj_soup = bs4.BeautifulSoup(html_file.text, 'lxml')
ex:
obj_soup.select('p'):尋找所有<p>標籤的元素
obj_soup.select('img'):尋找所有<img>標籤的元素
obj_soup.select('p .class'):尋找所有<p>且class屬性為class
obj_soup.select('p #id'):尋找所有<p>且id屬性為
obj_soup.select('div strong'):尋找所有在<section>元素內的<strong>元素
obj_soup.select('div > strong'):尋找所有在<section>內的<strong>元素
obj_soup.select('input[name]'):所有在<input>標籤且有name屬性的元素

# .attrs
objTag = objSoup.select('#author')
objTag[i].attrs
ex:(字典型態)
>> {'id': 'author'}

# getText() vs .text
print(str(objTag[i])) # 有strong標籤
print(objTag[i].getText()) # 無
print(objTag[i].text) # 無

# {tag}.get()
why .get()?
get主要用在標籤內沒有內文的狀況，例如:<img src="..." width="..." />
如果你要get src的部分
>> imgTag[i].get('src')

# 爬蟲實戰重點(抓取照片)

1.是否成功抓取
html_file = requests.get("url")
html_file.raise_for_status() # 可用這行來check是否成功抓取

2.儲存到特定資料夾的方法
pictFile = open(os.path.join(destDir, os.path.basename(img_url)), 'wb') # basename()用作刪除\image\
for diskStorage in picture.iter_content(10240):
    pictFile.write(diskStorage)
pictFile.close()

3.複習創資料夾
if os.path.exists("document") is False:
    os.mkdir("document")

# 威力彩範例
import requests
from bs4 import BeautifulSoup
import re
import os

headers = {'User-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64)\
            AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101\
            Safari/537.36', }

url = "https://www.taiwanlottery.com.tw/"
html_file = requests.get(url)
soup = BeautifulSoup(html_file.text, 'lxml')
lotTag = soup.select(".contents_box02")
ball = lotTag[0].find_all('div', {'class': 'ball_tx ball_green'})
print("本期開出的號碼是:", end=' ')
for i in range(len(ball)//2):
    print(ball[i].getText(), end=' ')
print()
print("本期照順序的號碼:", end=' ')
for i in range(len(ball)//2, len(ball)):
    print(ball[i].getText(), end=' ')
print()
print("第二區:", end=' ')
red_ball = lotTag[0].find_all('div', {'class': 'ball_red'})
print(red_ball[0].text)

* 重點:這邊的find_all是為了找出所有的綠球，綠球在div底下
所以要用find_all('div', {'class': 'ball_red'})
(找尋所有標籤是'div'，此標籤類別class是'ball_red'的結果)

# cmd
執行:
python 路徑\python C:\Users\sychen\PycharmProjects\python_note\0909.py
若需要input:
python 路徑\python C:\Users\sychen\PycharmProjects\python_note\0909.py 參數1 ... 參數n
    
print(sys.argv[0])
>> 路徑

print(sys.argv[1])
>> input 參數1

example:
import sys, webbrowser

print(sys.argv[0])
if len(sys.argv) > 1:
    address = " ".join(sys.argv[1:])
webbrowser.open("http://www.google.com.tw/maps/place/" + address)

==========Selenium==========
(Selenium可以用在超連結，input，避免406 not found...)

# 尋找html文件的元素
find_element_by_id(id): 傳回第一個相符id的元素
find_elements_by_id(id): 傳回所有相符的id元素，以串列傳回

find_element_by_class_name(name): 傳回第一個相符的Class元素
find_elements_by_class_name(name): 傳回所有相符的Class元素，以串列傳回

find_element_by_name(name): 傳回第一個相符name屬性的元素
find_elements_by_name(name): 傳回所有相符name屬性的元素，以串列傳回

find_element_by_css_selector(selector): 傳回第一個相符CSS selector的元素
find_elements_by_css_selector(selector): 傳回所有相符CSS selector的元素，以串列傳回

find_element_by_partial_link_text(text): 傳回第一個內含有text的<a>元素
find_elements_by_partial_link_text(text): 傳回所有內含相符text的<a>元素，以串列傳回

find_element_by_link_text(text): 傳回第一個完全相同text的<a>元素
find_elements_by_link_text(text): 傳回所有完全相同text的<a>元素，以串列傳回

find_element_by_tag_name(name):不區分大小寫，傳回第一個相符name的元素
find_elements_by_tag_name(name):不區分大小寫，傳回所有相符name的元素，以串列傳回

>>若沒找到相符的則傳回NoSuchElement error(try..except..)
>>若找到物件，可以用下列方法屬性獲得內容
tag_name:元素名稱
text:元素內容
location:字典，元素位於頁面的座標
clear():刪除文字欄位或文字區域欄位的文字
get_attribute(name):獲得元素name屬性的值
is_displayed():元素能否被看見(boolean)
is_enabled():元素能否立即使用(boolean)
is_selected():元素的核取方塊是否勾選(boolean)

# 點擊超連結
eleLink = browser.find_element_by_link_text('連結名稱')
eleLink.click()

# formbox by id
txtbox = browser.find_element_by_id("key")
txtbox.send_keys("輸入字元")
time.sleep(3)
txtbox.submit()

# python browser button
from selenium.webdriver.common.keys import Keys

ele = browser.find_element_by_tag_name('body')
time.sleep(3)
ele.send_keys(Keys.PAGE_DOWN)
time.sleep(3)
ele.send_keys(Keys.END)
time.sleep(3)
ele.send_keys(Keys.PAGE_UP)
time.sleep(3)
ele.send_keys(Keys.HOME)

# python瀏覽器操作
browser.forward() # 前一頁
browser.back() # 回一頁
browser.refresh() # 更新網頁
browser.quit() # 關閉網頁

==========Matplot========
先import
import matplotlib.pyplot as plt

squares = [1, 4, 9, 16, 25, 36, 49, 64]
seq = [1, 2, 3, 4, 5, 6, 7, 8] # 更改圖表起始值, 可能跟plt.axis衝突
plt.plot(seq, squares, linewidth=3) # 畫線代值, 線條寬度
plt.axis([0, 8, 0, 70]) # X,Y軸刻度範圍
plt.title("Not Support Chinese", fontsize=24)
plt.xlabel("Value", fontsize=16)
plt.ylabel("Square", fontsize=16)
plt.tick_params(axis='both', labelsize=12, color='red')  # both代表應用到x和y軸, color=刻度顏色
plt.show() # 顯示繪製的圖形

# 多組數據的應用
plot(第一組, 第二組)
ex:
plt.plot(seq1, squares1, seq2, squares2)

# 線條色彩與樣式
語法:
plt.plot(seq, value, 'g--')
族繁不及備載..
這邊帶一個例子:
plt.plot(seq, data1, 'g--', seq, data2, 'r-.', seq, data3, 'y:', seq, data4, 'k.')
                     '綠色虛線'          '紅色虛線+點'       '黃色點線'         '黑色單點'
'-*':線加星號
'-o':線加圓點
'-^':線加三角形
'-s':線加方形

# 刻度設計
seq = [2018, 2019, 2020]
plt.xtick(seq)

# legend()用法
lineBenz, = plt.plot(seq, benz, '-*', label='Benz') # 須注意要有','
lineBMW, = plt.plot(seq, bmw, '-o', label='BMW')
lineLexus, = plt.plot(seq, lexus, '-^', label='Lexus')
plt.legend(handles=[lineBenz, lineBMW, lineLexus], loc='best') # loc有0~10可設定
plt.show()

# 圖例放到圖表外面的方法
plt.legend(handles=[lineBenz, lineBMW, lineLexus], loc='upper left', bbox_to_anchor=(1, 1)) # 我也不知道為什麼
plt.tight_layout(pad=7) # 解決留白不足

# 保存圖檔
plt.savefig('out.jpg', bbox_inches='tight') # tight=將圖表外多餘的空間刪除

# scatter 系列點
xpt = list(range(1, 101))
ypt = [x**2 for x in xpt] # 這個用法記得
plt.scatter(xpt, ypt, color='y') # 黃色
plt.show()

# 設置繪圖區間
語法:([xmin, xmax, ymin, ymax])
plt.axis([0, 100, 0, 10000])
plt.scatter(xpt, ypt, color=(0,1,0))

==========Numpy========

# numpy module
# 建立簡單的陣列
import numpy as np
np.linspace(start, end, num) # [1. 2. 3.]
np.arange(start, stop, step) # [1 2 3]

# 繪製波形
xpt = np.linspace(0, 10, 500)
ypt = np.sin(xpt)
plt.scatter(xpt, ypt)
plt.show()

# 不等寬度散點圖
plt.scatter(xpt, ypt, s=lwidth)
寬度可以藉由xpt代入函數控制

# 色彩映射color mapping
x = np.arange(100)
y = x
plt.scatter(x, y, s=50, c=x, cmap='rainbow')
* s:點寬度 cmap:rainbow(紅高紫低) c:顏色根據x變動
* cmap有很多種映射表 可至https://matplotlib.org/examples/color/colormaps_reference.html查詢

# np.random.random()應用
語法: np.random.random(num) # 產生num個0.0到1.0之間的數字

# list隨機選1-7
random.choice([1, 2, 3, 4, 5, 6, 7])

# 隱藏座標
plt.axes().get_xaxis().set_visible(False)
plt.axes().get_yaxis().set_visible(False)

# 繪製多個圖表
plt.figure(1) # 建立圖一
plt.plot(seq, data1, '-*') # 繪製圖一
plt.figure(2) # 建立圖二
plt.plot(seq, data2, '-o') # 繪製圖二
plt.title('figure2') # 這邊的設定只對figure2有效
plt.show()

* 看plt.figure()的位置向下都是statement

# 含有子圖的圖表
subplot(x1, x2, x3)
x1:垂直幾張
x2:水平幾張
x3:這個是第幾張(左上到右下)

ex:
# 先定義subplot，在plot
plt.subplot(1, 2, 1) # (垂直, 水平, 第幾張)
plt.plot(seq, data1, '-*')
plt.subplot(1, 2, 2)
plt.plot(seq, data2, '-o')
plt.show()

# 長條圖bar()
votes = [1, 2, 3]
x = np.arange(len(votes))
plt.bar(x, votes, width=0.35) # x軸, y_bar, 寬度
// 改刻度
plt.xticks(x, ['x', 'y', 'z'])
plt.yticks(np.arange(0, 450, 30))

#　擲骰範例
def dice_generator(times, sides):
    for i in range(times):
        ranNum = randint(1, sides)
        dice.append(ranNum)


def dice_count(sides):
    for i in range(1, sides+1):
        frequency = dice.count(i)　# 數i有幾個
        frequencies.append(frequency)


times = 600
sides = 6
dice = []
frequencies = []
dice_generator(times, sides)
dice_count(sides)
x = np.arange(6)
plt.bar(x, frequencies, width=0.35, color='g')
plt.xticks(x, [1, 2, 3, 4, 5, 6])
plt.yticks(np.arange(0, 150, 15))
plt.title('Test 600 times')
plt.ylabel('Frequency')
plt.show()

# header
with open(fn) as csv_file:
    csvReader = csv.reader(csv_file)
    headerRow = next(csvReader) # next 從迭代器中撈出下一個
print(headerRow)

=>複習
for i, header in enumerate(headerRow):
這行會把串列裡的tuple 讀出
enumerate範例:
[(0,'紹紹'), (1, '昭昭'), (2, '喵喵')]
拆開後會變成
0 紹紹
1 昭昭
2 喵喵

# csv讀取row
for row in csvReader: # 每行都掃
    print(row[3]) # 只抓每行的第三列

* 若讀入的csv file(字串型態)，畫出來的圖不會order，所以要先轉成int

# 設置繪圖區大小
寬:n*width像素 高:n*height像素
plt.figure(dpi=n, figsize=(width, height))

# 日期格式解析
將字串轉成日期物件ex
dateObj = datetime.strptime('2017/1/1', '%Y/%m/%d')
print(dateObj)

# 圖表增加日期刻度
用上述dateObj加入plt.plot(dateObj)

# 將日期位置旋轉
fig = plt.figure(dpi=80, figsize=(12, 8)) # 繪圖區大小
plt.plot(date, highTemps) # 圖標增加日期刻度
fig.autofmt_xdate(rotation=60) # 日期旋轉
* 需要先plt.plot後再旋轉調整

# 兩張圖Plot再一起
plt.plot(date, highTemps)
plt.plot(date, lowTemps)

# 填滿兩條線的中間區域
plt.fill_between(date, lowTemps, highTemps, color='y', alpha=0.2) # alpha為透明度0.2

# 軸刻度應用於日期
plt.axis([date[0], date[30], 0, 40])
* 需要用串列傳入

==========json==========(可參照課本的轉換表)
資料格式有兩種:
object:{}
array:[]

# 物件
def:物件是用"key: value"配對儲存
範例:
{"string":value, "string2": value}
*key必須要是文字，且必須用雙引號，也不可用註解

# 陣列
def:陣列是由一系列的value組成
範例:
["string", number, object, array]

# dumps() 將python資料轉成json
轉換過後的資料型別都是字串。<class: 'str'>
ex:
listObj = [{'Name':'Peter', 'Age':25, 'Gender':'M'}]
jsonData1 = json.dumps(listObj)

# dumps() - sort_keys參數
將python資料轉成json時，可以增加sort_keys=True(python dict本是無序)
ex:
dictObj = {'b': 80, 'a': 25, 'c': 60} # 本來是b->a->c
jsonObj = json.dumps(dictObj, sort_keys=True)
>> 經過sort_keys後會變成 a->b->c

# dumps() - indent參數
使資料容易閱讀
jsonObj = json.dumps(dictObj, sort_keys=True, indent=4)

# loads()
def:將json轉成python資料
ex:
jsonObj = '{"b": 80, "a": 25, "c": 60}'
dictObj = json.loads(jsonObj)
>> <class 'dict'>

//=====如何用python應用於輸出json檔
牽涉到檔案都是用dump(), load() (no s)
ex:
dictObj = {'b': 80, 'a': 25, 'c': 60} // python字典檔  
# 寫json檔
fn = 'out24_6.json'
with open(fn, 'w') as fnObj:
    json.dump(dictObj, fnObj)

# 讀json檔
fn = 'out24_6.json'
with open(fn, 'r') as fnObj1:
    data = json.load(fnObj1)

# 如何從json抓資料
data['Year']配上for loop
*值得注意的是若字串為浮點數的樣子，要轉換成integer
必須先轉換成浮點樹再轉成整數
ex:
int(float('str'))

# pygal.maps.world 國碼資訊

列出所有國家代碼與國名:
from pygal.maps.world import COUNTRIES

for countryCode in Sorted(COUNTRIES.keys()):
    print("國家代碼:", countryCode, "國家名稱:", COUNTRIES[countryCode])

# pygal.maps.world 畫世界地圖

import pygal.maps.world

worldMap = pygal.maps.world.World()

worldMap.title = 'China in the Map'
worldMap.add('China', ['cn']) # China會出現在圖例的左上角，['cn']圖上會填色
worldMap.render_to_file('out24_14.svg') # 輸出svg檔

# 讓地圖呈現數據
範例
worldMap.add('Asia', {'cn':1262645000, 'jp':12687000, 'th': 63155029})

# 如果要從json檔讀出來 給worldMap.add()用
def getCountryCode(countryName):
    for dictCode, dictName in COUNTRIES.items():
        if dictName == countryName:
            return dictCode
    return None

*要先轉換，然後要記得 
for a, b in dict.items() 才是抓key-value
for a, b in dict 這個只有抓key的兩個字元


==========傳送電子郵件==========

# import smtplib module
使用smtplib模組

# 建立smtp物件
mySMTP = smtplib.SMTP('smtp.gmail.com', 587)
mySMTP.ehlo() # 啟動對話
mySMTP.starttls() # 啟動tls加密
mySMTP.login('email', 'password') # 登入smtp

# 簡單傳送電子郵件
status = mySMTP.sendmail('寄件者', '收件者','Subject: 標題\n內容')
print(status) # 若為空則傳送成功

# 後續
if status == {}:
    print("郵件發送成功")
mySMTP.quit() # 結束smtp連線

# sendmail()
sendmail(from_addr, to_addr_list, msg[mail_options, rcpt_options])

# 寄出txt內容
with open('out3.txt') as fn:
    mailContent = fn.read()
msg = msg + mailContent  # msg 之前要先設定好sub, from, to

# 傳送中文 by MIME
from email.mime.text import MIMEText

msg = MIMEText('傳送中文內容', 'plain', 'utf-8')
msg['Subject'] = 'Email using MIME'
msg['From'] = '我愛拍森'
msg['To'] = '..@gmail.com'
msg['Cc'] = '..@gmail.com'

以及 sendmail裡面的參數要改成
status = mySMTP.sendmail(from_addr, to_addr, msg.as_string())

# 傳送html
html_str='''
<!doctype html>
'''
msg = MIMEText(htmlstr, 'html', 'utf-8')

# 夾帶檔案
with open('out3.txt', 'rb') as fn: # 讀檔案 # 'rb'可為'r'
     mailContent = fn.read()
msg = MIMEText(mailContent, 'base64', 'utf-8') # 'base64'可為'plain'
msg['Content-Type'] = 'application/octet-stream' # 可刪
msg['Content-Disposition'] = 'attachment; filename="out3.txt"'
*註解後無法傳送二進位檔案

# 夾帶圖片
from email.mime.image import MIMEImage
with open('xxx.jpg', 'rb') as fn:
    mailPic = fn.read()
msg = MIMEImage(mailPic)
msg['Content-Type'] = 'application/octet-stream'
msg['Content-Disposition'] = 'attachment; filename="xxx.jpg"' # 副檔名要改

# smtp異常處理
try:
    ...
    print("發送成功")
except smtplib.SMTPException:
    print("發送失敗")

==========Pillow==========

# 色碼轉換
from PIL import ImageColor

# ImageColor.getrgb()
print(ImageColor.getrgb("#0000ff"))
print(ImageColor.getrgb("blue"))
print(ImageColor.getrgb("rgb(0%, 0%, 100%)"))

# ImageColor.getcolor()
print(ImageColor.getcolor("#0000ff", "RGB"))
print(ImageColor.getcolor("blue", "RGBA"))

# Box tuple
(left, top, right, bottom)
從左上到右下遞增

# 影像操作
img_obj = Image.open("rushmore.jpg") # 建立物件
width, height = img_obj.size # 獲得大小
filename = img_obj.filename # 獲得名稱
print("副檔名:", img_obj.format) # 獲得副檔名
print("詳細資訊:", img_obj.format_description) # 獲得詳細資訊
img_obj.save("rushmore_bak.png") # 存檔轉成png

# 建立新的影像物件
語法:
new(mode, size, color=0)
mode="RGBA"(.png); mode="RGB"(.jpg)
ex:
img_obj = Image.new('RGBA', (300, 100), 'aqua')

# 更改影像大小
pict = Image.open("rushmore.jpg")
width, height = pict.size

newpict1 = pict.resize((width*2, height))
newpict1.save('rushmore1.jpg')

# 影像旋轉(逆時針)
pict = Image.open("rushmore.jpg")
pict.rotate(90).save("...")
pict.rotate(180).save("...")
*圖像本身比率不變，多的部分用黑色影像替代
pict.rotate(45).save("out27_11_1.jpg") # 此圖會被切掉
pict.rotate(45, expand=True).save("out27_11_2.jpg") # 使用expand會將整個影像顯示(會變大)

# 影像翻轉
transpose(Image.FLIP_LEFT_RIGHT) # 左右翻轉
transpose(Image.FLIP_TOP_BOTTOM) # 上下翻轉

# 像素查詢
img_obj.getpixel(150, 50) # 查詢(150, 50)的座標

# 像素插入
img_obj.putpixel((x, y), (0, 255, 255, 255))
ex:
for x in range(50, 251):
    for y in range(151, 251):
        newImage.putpixel((x, y), ImageColor.getcolor('blue', 'RGBA'))
newImage.save("out27_14_2.png")

# 裁切圖片
pict.crop((80, 30, 150, 100)) # 左上右下
pict = Image.open("rushmore.jpg")
cropPict = pict.crop((80, 30, 150, 100))
cropPict.save("outcrop.jpg")

# 複製圖片
copyPict.paste(cropPict, (20, 100))
*以copyPict為底在(20, 100)貼上cropPict

# 裁切圖片填滿某區間
for x in range(20, width-20, cropWidth):
    for y in range(20, height-20, cropHeight):
        newImage.paste(cropPict, (x, y))

# 在圖片內繪製圖案
->new 圖片物件
->new 繪圖物件 (將圖片物件丟進ImageDraw.Draw(img_obj)
newImage = Image.new('RGBA', (300, 300), "Yellow")
draw_obj = ImageDraw.Draw(newImage)

->繪製點
point([(x, y)], fill='Green')

->繪製線條
line([(x1,y1), ... (xn, yn)], width, fill) # fill顏色

->繪製圓或橢圓(outline是外觀顏色)
ellipse((left, top, right, bottom), fill, outline)

->繪製矩形
rectangle((left, top, right, bottom), fill, outline)

->繪製多邊形
polygon([(x1, y1), ...(xn, yn)], fill, outline)

# 在圖片內填寫文字
from PIL import Image, ImageDraw, ImageFont

text((x, y), text, fill, font)
ex:
fontInfo = ImageFont.truetype('‪C:\\Windows\\Fonts\\OLDENGL.TTF', 36)
drawObj.text((50, 100), strText, fill='blue', font=fontInfo)

# 建立QRcode
import qrcode
qrcode.make(text)
範例:
codeText = 'www.sycomputer.org'
img = qrcode.make(codeText)
img.save("out27_22.jpg")

==========PyAUTOGUI==========

# width, height = pyautogui.size() # 螢幕寬度高度

# xloc, yloc = pyautogui.position() # 滑鼠位置

# pyautogui.moveTo(x, y, duration=0.5) # 移動到(x, y), 時間0.5s

# pyautogui.moveRel(x, y, duration=0.5) # 相對於現在的座標移動(x, y), 時間0.5s

# 利用try-except, CTRL+C 結束
try:
except KeyboardInterrupt:

# 靠右輸出(由右往左4個)
str.rjust(4)

# 讓滑鼠位置的輸出固定位置
請參閱書

# 按一下滑鼠click()
pyautogui.click(x, y, button='right')

# 按住與放開滑鼠mouseDown()和mouseUp()
pyautogui.mouseDown(button='right')
time.sleep(1)
pyautogui.mouseUp(800, 300, button='right')

# 拖曳滑鼠dragTo()/dragRel()
pyautogui.dragTo(x, y, duration=)
pyautogui.dragRel(x, y, duration=)
*使用方式同moveTo, moveRel

# 視窗滾動scroll()
pyautogui.scroll(30)
若要指定起始位置可加上
pyautogui.scroll(30, x=30, y=100)

# 螢幕截圖
pyautogui.screenshot("out.jpg")
*也可以用物件的方式(如下)

# 螢幕截圖裁切
screenImage = pyautogui.screenshot()
cropPict = screenImage.crop((960, 210, 1900, 480))
cropPict.save("out.jpg")

# 獲得影像某位置色彩
screenImage = pyautogui.screenshot()
x, y = 200, 200
print(screenImage.getpixel((x, y)))

# 色彩比對(return boolean)
tf = pyautogui.pixelMatchesColor(x, y, (79, 75, 65))
arg=(x,y)座標，rgb色碼

# 控制鍵盤傳送文字
pyautogui.typewrite("I LOVE ZHAOZHAO", 0.1) # 每隔0.1s輸入一個字元

# 單一字元輸入
pyautogui.typewrite(["i", " ",  "l", "o", "v", "e", " ", "z", "h", "a", "o", " ", "z", "h", "a", "o"], 0.2) 
* 可以搭配鍵盤的特殊功能使用
'a', '$', 'enter', 'delete', 'esc', 'up', 'down'...請詳閱文件

# 放下與放開按鍵
pyautogui.keyDown("shift") # 按下去不放開
pyautogui.press("8") # 按下並放開
pyautogui.keyUp("shift") # 就放開

# 快速組合鍵
承上例:*
pyautogui.hotkey('shift', '8')

# plt.text
# 顯現 the Sigmoid formula
plt.text(4, 0.8, r'$\sigma(x)=\frac{1}{1+e^{-x}}$', fontsize=15)


