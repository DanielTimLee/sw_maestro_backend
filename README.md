# SW마에스트로 백엔드 과제 [이호연]
### [과제 페이지 (110.35.161.182:8888)](http://110.35.161.182:8888)



**전체적인 과제에 대한 설명은 다음과 같습니다.**

1. Vectorize에 대한 이해
2. 형태소 분석
3. Stop Word
4. 태깅
5. 이미지 처리
6. 다른 모델 활용안


## 1. Vectorize에 대한 이해

### Vectorize 란?
 학습할 데이터에 있는 모든 글자를 단어 단위로 잘라 한 단어를 `(0,27390)`와 같은 형태의 벡터 차원으로 나타내는 과정을 말합니다.

SVM 모델에서는 이 모든 데이터를 벡터화 시키고,
각 카테고리(108개)가 가진 벡터들의 특징들을 기준으로 각각의 차원을 분리하고 이를 이용합니다.

이번 과제에 있어서 성적 향상에 중요하게 보았던 점은 Vectorize가 어떤식으로 작동하는지 입니다. 

```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

def vec(str):
	a = vectorizer.fit_transform([str])
	print (a)
	
vec("a.cat@굴:국밥/1 23%4_")
  (0, 2)	1
  (0, 3)	1
  (0, 0)	1
  (0, 1)	1
```

다음과 같이 `.`,`@`,`/`,`%`,`-` 등 대부분의 특수문자의 경우에는 그냥 공백(whitespace)와 동일하게 인식하고 지나가 버리는 경향이 있었습니다. 

##### 딱  한개 `_`(언더바) 한개 빼고는
거의 모든 특수문자가 글자로 인식되지 않고 띄어쓰기로 취급되었습니다.

### 글자 자르기
Vectorize에 대한 분석을 하기 전에는 데이터 중 `[인텔][인텔]i7-4770k` 이라는 데이터에 대해서 `[`,`]`,`-` 문자 등이 글자로 취급이 되어서 저 데이터는 벡터화 될 때에 '인텔엔텔i7-4770k' 라는 하나의 데이터로 인식이 될 꺼라 생각했습니다.

그러나 벡터화 될 때는 특수문자는 알아서 제거해 주기 때문에 `인텔`,`인텔`,`i7`,`4770k`로 분리가 되기 때문에 신경쓰지 않아도 되었습니다.

## 2. 형태소 분석

### KoNLPy의 태그 모델?
자연어 처리 패키지인 KoNLPy에는 총 (Kkma, Twitter 등) 5가지의 품사 태깅 클래스가 있습니다.

**[형태소 분석 및 품사 태깅](http://konlpy.org/ko/v0.4.3/morph/#pos-tagging-with-konlpy)**

위의 자료를 참고해 보면 `Kkma`클래스와 `Mecab`클래스가 가장 정교하게 잘 잘라주는 클래스입니다.

`Kkma`나 `mecab`의 경우가 더 상세히 분석을 잘 해주지만
너무 상세히 분류를 하게 되어서 한 글자 단위로까지 너무 세부하게 분류하는 경향이 있습니다.

(단어 한 글자일시 `vectorize`화 시킬때 단어를 인식 못하고 넘어가는 경우가 있습니다)

(단어를 너무 세부하게 잘라 의미 없는 단어 단위까지로 쪼개버립니다)

심지어 저 둘을 사용했을때는 오히려 점수가 떨어지는 현상도 발생했습니다.

따라서 이보다 조금 더 간단히 분류하는 `Twitter 분류 클래스`를 이용하였습니다.

**[각 클래스별 분류 태그](https://docs.google.com/spreadsheets/d/1OGAjUvalBuX-oZvZ_-9tEfYD2gQe7hTGsgUpiiBSXI8/edit#gid=0)**

위의 링크를 보면 알 수 있지만 트위터의 경우에는 다른 분류 클래스와는 다르게 `동사`,`명사`,`형옹사`,`관형사` 등 큼직큼직하게 잘라서 데이터가 너무 파편화 되는 것을 방지할 수 있습니다. 


## 3. Stop Word

### Stop Word?
만개의 상품 데이터 중에 상품과 직접적인 연관이 없는 다른 정보들을 의미합니다.

```python
import pandas as pd
train_df=pd.read_pickle("soma_goods_train.df")
train_df

cate1	cate2	cate3	name
842695	패션의류	아동의류	한복	[신한카드 5%할인, 3/19]예화-좋은아이들名品 남아 아동한복 금직배자보라 600...
842696	패션의류	아동의류	한복	[프렌치캣] [2015년 봄 기획상품]세로 ST 레깅스(Q51DKP290) [갤러리...
842697	패션의류	아동의류	한복	[BC카드5%할인][예화_좋은아이들] 名品 아동한복 남아_ 4005 이산검정 (배자...
842698	패션의류	아동의류	한복	[5% 즉시할인] 고빅스 여성 한복 소품 머리장식 주석칠보 뒷꽂이 NA027 3개
842699	패션의류	아동의류	한복	[최저가파워] 아동한복 [예화-좋은아이들] 아동한복 여아 1078 빛이나분홍
90988	디지털/가전	PC부품	CPU	[ICODA] [대리점정품]인텔 제온 E5-2630V3 하스웰-EP (2.4GHz/...
90990	디지털/가전	PC부품	CPU	인텔 intel 코어4세대 하스웰 i3-4160
90997	디지털/가전	PC부품	CPU	[해외]IntelIntel Pentium Processor G3258 4 BX806...
90998	디지털/가전	PC부품	CPU	[4%즉시할인쿠폰]인텔 제온 E3-1226V3 (하스웰) (정품)
90999	디지털/가전	PC부품	CPU	[해외]IntelIntel?? Xeon?? Processor X5570 (8M Ca...
```

다음과 같이 상품과 직접적인 연관이 없는 `5% 할인`,`BC카드`,`최저가파워`,`3/19`등 의미 없는 단어가 매우 많은 것을 알 수 있습니다.

이런 단어들을 한번에 모두 정리해 주기 위해서  `정규 표현식 (Regular Expression)`,`문자열 비교` 등을 이용하였습니다.

### 정규표현식 이용

```python
import re
r1 = re.compile("(\d{1}/|최대|)(\d{2}|\d{1})만원( |)(최대|적립|이상| |)(↑| |)(구매고객|구매시|무료배송 및 사은품| |)( |)(최대|적립|이상| |)(\d{1}개|(.*)증정| |)((.*)증정| |)")
train_df.iloc[i]['name']=r1.sub(" ",train_df.iloc[i]['name'])
 
r2 = re.compile("(시중가|기존가|정상가|최초가|)(:|)(\d{7}|\d{6}|\d{5}|((\d{3}|\d{2})( |,|)\d{3})|\d{4}|\d{3}|\d{2}|\d{1})원")
train_df.iloc[i]['name']=r2.sub(" ",train_df.iloc[i]['name'])
    
r3 = re.compile("(AK플라자|AK PLAZA|AK몰|AK유아동|AKMALL|_AK|AK수원점)")
train_df.iloc[i]['name']=r3.sub(" ",train_df.iloc[i]['name'])
 
r4 = re.compile("(\D{2}|)( |)백화점( |)((.*)관|)")
train_df.iloc[i]['name']=r4.sub(" ",train_df.iloc[i]['name'])
   
r5 = re.compile("(\d{2}|\d{1}|)(균일가|진열|시즌|%|)(마지막|한정|파격|특가|A|대박|)(%| |)세일(할인|)")
train_df.iloc[i]['name']=r5.sub(" ",train_df.iloc[i]['name'])
 
r6 = re.compile("(\d{2}.|)(\d{3}|\d{2}|\d{1})%( |)(즉시할인|할인|OFF|Sale|sale|)(쿠폰|)(~|)((.*)일까지|)")
train_df.iloc[i]['name']=r6.sub(" ",train_df.iloc[i]['name'])

r7 = re.compile("카드(\D{3,5})(\d{2}|\d{1})\/(\d{2}|\d{1})(\~|)((\d{2}|\d{1})\/(\d{2}|\d{1})|)")
train_df.iloc[i]['name']=r7.sub("카드",train_df.iloc[i]['name'])
    
r8 = re.compile("(\.\.\.\.|\.\.\.|\.\.| : | :|: |::|\((I|C|M|F|E|주)\))")
train_df.iloc[i]['name']=r8.sub(" ",train_df.iloc[i]['name'])
    
r9 = re.compile("(정성으로|유선문의|부드럽게|)( |)(작동됩|안됩|됩|팝|판매합|배송하겠습| 배송합|합|부탁드립|드립|바랍|다하겠습|입|)니다(\.|\!|)")
train_df.iloc[i]['name']=r9.sub(" ",train_df.iloc[i]['name'])
    
r10 = re.compile("(이|)( |)(가격이|샵이|고객만족을위해|)( |)최선(이다|입니다|을|)( |)(다하겠습니다|)")
train_df.iloc[i]['name']=r10.sub(" ",train_df.iloc[i]['name'])

```

###문자열 비교 이용

```python
useless_word = ["신한카드","현대카드","삼성카드","KB국민카드","KB카드","국민카드","BC카드","우리카드","롯데카드","하나카드",
                "구.하나SK",
                "비씨카드","씨티카드",
                "방문수령가능","즉시할인쿠폰","빠름배송","노마진","행사","모든구매","전구매","신제품","새제품","신속","정확","출고",
                "현대H몰","CJmal","롯데i몰","이마트몰","롯데닷컴","신세계몰","11번가","위메프","GS샵","G마켓","쿠팡","옥션","티몬",
                "관부가세","부가세","총알","출고",
                "계산서","계산","미포함","포함","상당","응모","불가","한정","수량","무조건","오케이","빠른","추천",
                "연중무휴","적립금","할인", "즉시","수량","한정","단골","우대","최대","추가","적립",
                "발행", "세금","단독", "특가", "구매", "대행","선착순" ,"사은품", "이벤트", "증정", "직배송", 
                "배송","발송","착한","가격", "쿠폰", "1주년", "판매점", "무료", "당일",
                "Ⅷ관","VII관","Ⅶ관","III관","Ⅲ관","VI관","IV관","Ⅱ관","Ⅸ관","Ⅰ관","V관",
                "최저가", "최고가", "저가", "고가", "기존가"]
special_char = ["&#39;","&frasl;","&amp;","&gt;","&quot;","col:","ㅁ","ㅇ","ㅣ",
                ':','@','▶',"!",'|','┕','Λ','Ο','Λ','◆','正','本','♥','●','※',
                '◀','┙','★','☆','*','名','品','大','＋','+','■','♣','━',
                'ㄴ','ㄱ','┏','┓','╋','?','▩','無','有','{','}','[',']','(',')']
    
"""
" x "," X ",',','/'
"""
    
for word in useless_word :
    train_df.iloc[i]['name']=train_df.iloc[i]['name'].replace(word, "")
    
for word in special_char :
    train_df.iloc[i]['name']=train_df.iloc[i]['name'].replace(word, " ")


```


## 4. 태깅 (Tagging)

### 태깅 (Tagging) ?
`Twitter` 등을 이용한 형태소 분석기를 이용하고 나면 숫자나, 한 글자씩 단어가 떨어져 나가는 현상이 발생했습니다.

이를 가지고 `Vectorize`를 돌려버리게 된다면 한 글자로 되어있는 단어들은 모두 사라지게 됩니다.

저는 이 점에 주목해서  한 글자로 떨어진 글자도 분명히 특징을 가지고 있을 것이고,
이를 모두 잃어버리게 되는 것은 매우 큰 손실이라 생각했습니다.

그래서 이 한 글자로 되어있는 글자들 뒤에 `문자열로 인식되는 태그 (ex) '_'` 등을 붙여서 한 글자가 아닌 단어로 만들어 버리게 된다면 이 부분을 해결할 수 있을 것이라 생각했습니다.

따라서 문자열의 길이를 모두 검사한 후에 뒤에  `'_'` 와 같은 태그를 붙여주는 과정을 모두 처리해 주었습니다.

```python
def tagging(result):
    for idx,data in enumerate(result):
        if len(data)==1:
            #if not data.isnumeric():
            result[idx]=data+"_"
        
    return " ".join(result)
```

**단점으로써는 데이터가 이미 너무 많이 파편화되어있기 때문에 태깅을 해도 뚜렷하게 구분되지 않든 데이터가 많습니다.**

**ex) '3_' (숫자가 그냥 떨어져 나갔습니다.),'용_' (남성용, 여성용 할 때의 용만 떨어져 나갔습니다)**


## 5. 이미지 처리

### 카페를 활용한 이미지 처리
`Caffe` 를 활용한 이미지 처리를 위해 만장의 사진을 모두 `Caffe` 도커 인스턴스 안에서 다운을 받았습니다. (wget)

`Caffe` 웹 버전에서는 이미지 한 장의 url을 입력받아 한 개의 데이터를 돌려주는 형식인데, 이런 식으로 처리하면 너무 작업이 오래걸려서 `Caffe` 웹 버전의 서버를 뜯어서 처리를 진행하였습니다.

```python
from multiprocessing import Process, Queue

from glob import glob
images =glob("soma_train/*.jpg")
File_name = images


def thread(first,last,result):
    for i in range(first,last):
        try:
            fn=File_name[i].replace("soma_train/","").replace(".jpg","")
            image = caffe.io.load_image(File_name[i])
            result.put({fn:app.clf.classify_image(image)})
        
        except:
            print ("%s error! %s")%(i,File_name[i])
    
    traceback.print_exc()
    return
------------------------------------------------------------------
									생략
------------------------------------------------------------------
def start_tornado(app, port=9000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    result=Queue()
    pr1=Process(target=thread, args=(0,1250,result))
    pr2=Process(target=thread, args=(1250,2500result))
    pr3=Process(target=thread, args=(2500,3750,result))
    pr4=Process(target=thread, args=(3750,5000,result))
    pr5=Process(target=thread, args=(5000,6750,result))
    pr6=Process(target=thread, args=(6750,7500,result))
    pr7=Process(target=thread, args=(7500,8750,result))
    pr8=Process(target=thread, args=(8750,10000,result))
    pr1.start();pr2.start();pr3.start();pr4.start();
    pr5.start();pr6.start();pr7.start();pr8.start();
    pr1.join();pr2.join();pr3.join();pr4.join();
    pr5.join();pr6.join();pr7.join();pr8.join();

    result.put("Stop")
    res=[]
    while True:
        tmp = result.get()
        if tmp == "Stop" : break
        else:
            print(tmp)
            res.append(tmp)

    file_ = open('last--result-'+str(first)+'-'+str(last)+'.txt', 'w')
    for item in res:
        file_.write("%s\n"%item)

    file_.close()

    tornado.ioloop.IOLoop.instance().start()

```


**단점으로써는 데이터가 이미 너무 많이 파편화되어있기 때문에 태깅을 해도 뚜렷하게 구분되지 않든 데이터가 많습니다.**

**ex) '3_' (숫자가 그냥 떨어져 나갔습니다.),'용_' (남성용, 여성용 할 때의 용만 떨어져 나갔습니다)**








### 순위 이미지
![순위 이미지](https://mail.google.com/mail/u/0/?ui=2&ik=a706780371&view=fimg&th=156bd0f471c9d504&attid=0.1&disp=emb&realattid=ii_156bd0f2468fa659&attbid=ANGjdJ86f6cpklAjfs5BuFPg21-mHdsWerfyQHoyPCgjl1ij7RRbTg2oOHRRoeG0K9tFaiUHSBmM9leCAxi2y3CZzc4lf9bXv28nsxch3-ApMij-R-0oiB34J6gRCkE&sz=w836-h1124&ats=1472082665624&rm=156bd0f471c9d504&zw&atsh=1)
