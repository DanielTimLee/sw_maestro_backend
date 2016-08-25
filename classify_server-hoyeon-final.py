
# coding: utf-8

# 

# # 과제에 대한 설명은 soma_classifier_hoyeon-final 파일 혹은 깃헙에 있습니다.
# 
# soma_classifier_hoyeon-final
# 
# https://github.com/DanielTimLee/sw_maestro_backend

# In[1]:

from sklearn.externals import joblib
import nltk
import re
from konlpy.tag import Twitter

twitter = Twitter()
clf = joblib.load('classify_8886j.model')
cate_dict = joblib.load('cate_dict_8886j.dat')
vectorizer = joblib.load('vectorizer_8886j.dat')
joblib.dump(clf,'n_classify.model')
joblib.dump(cate_dict,'n_cate_dict.dat')
joblib.dump(vectorizer,'n_vectorizer.dat')
cate_id_name_dict = dict([(v,k)for k,v in cate_dict.items()])


# In[2]:

def tagging(result):
    for idx,data in enumerate(result):
        if len(data)==1:
            #if not data.isnumeric():
            result[idx]=data+"_"
        
    return " ".join(result)


# In[ ]:

from flask import Flask, request,jsonify
from threading import  Condition
import time
_CONDITION = Condition()
app = Flask(__name__)

i=0   
@app.route('/classify')
def classify():
    global i
    i+=1
    
    #print ("classify called")
    
    #img = request.args.get('img')
    name = request.args.get('name')
    
    r1 = re.compile("(\d{1}/|최대|)(\d{2}|\d{1})만원( |)(최대|적립|이상| |)(↑| |)(구매고객|구매시|무료배송 및 사은품| |)( |)(최대|적립|이상| |)(\d{1}개|(.*)증정| |)((.*)증정| |)")
    name=r1.sub(" ",name)
 
    r2 = re.compile("(시중가|기존가|정상가|최초가|)(:|)(\d{7}|\d{6}|\d{5}|((\d{3}|\d{2})( |,|)\d{3})|\d{4}|\d{3}|\d{2}|\d{1})원")
    name=r2.sub(" ",name)
    
    r3 = re.compile("(AK플라자|AK PLAZA|AK몰|AK유아동|AKMALL|_AK|AK수원점)")
    name=r3.sub(" ",name)
 
    r4 = re.compile("(\D{2}|)( |)백화점( |)((.*)관|)")
    name=r4.sub(" ",name)
   
    r5 = re.compile("(\d{2}|\d{1}|)(균일가|진열|시즌|%|)(마지막|한정|파격|특가|A|대박|)(%| |)세일(할인|)")
    name=r5.sub(" ",name)
 
    r6 = re.compile("(\d{2}.|)(\d{3}|\d{2}|\d{1})%( |)(즉시할인|할인|OFF|Sale|sale|)(쿠폰|)(~|)((.*)일까지|)")
    name=r6.sub(" ",name)

    r7 = re.compile("카드(\D{3,5})(\d{2}|\d{1})\/(\d{2}|\d{1})(\~|)((\d{2}|\d{1})\/(\d{2}|\d{1})|)")
    name=r7.sub("카드",name)
    
    r8 = re.compile("(\.\.\.\.|\.\.\.|\.\.| : | :|: |::|\((I|C|M|F|E|주)\))")
    name=r8.sub(" ",name)
    
    r9 = re.compile("(정성으로|유선문의|부드럽게|)( |)(작동됩|안됩|됩|팝|판매합|배송하겠습| 배송합|합|부탁드립|드립|바랍|다하겠습|입|)니다(\.|\!|)")
    name=r9.sub(" ",name)
    
    r10 = re.compile("(이|)( |)(가격이|샵이|고객만족을위해|)( |)최선(이다|입니다|을|)( |)(다하겠습니다|)")
    name=r10.sub(" ",name)
    
    
    """
    'Free Shipping','Free shipping','Freeshipping',"new ","NEW ","New ",
    \b [MB|TB|KB|GHz|mhz]\b
    """

    brand=["Satechi","Logitech","INTEL","Intel","IBM","HP","Shure","Dell"]
    
    for word in brand :
        name=name.replace(word, " "+word+" ")    


    name=name.replace('_', " ")
    name=name.replace('.', "_")
    name=name.replace('A/S', "AS")
    name=name.replace('PS/2', "PS2")
    name=name.replace(':', "_")
    

    
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
        name=name.replace(word, "")
    
    for word in special_char :
        name=name.replace(word, " ")
    
    
    twitterPosList = twitter.pos(name)
    a=[]
    for q in twitterPosList:
        a.append(q[0])
    #q
    name+=tagging(a)
        
    name=name.replace("__"," ")
    name=name.replace("-_"," ")
    name=name.replace("&_"," ")
    name=name.replace("=_"," ")
    name=name.replace("$_"," ")
    name=name.replace("^_"," ")
    name=name.replace("#_"," ")
    name=name.replace("\_"," ")
    
    print(i)

    pred = clf.predict(vectorizer.transform([name]))[0]

    return jsonify({'cate':cate_id_name_dict[pred]})


app.run(host='0.0.0.0', port=8886)


#  * 추후 여기 docker 에서 뭔가 python package 설치할게 있으면 
#  * /opt/conda/bin/pip2 install bottle 이런식으로 설치 가능

# In[ ]:

http://somaeval.hoot.co.kr:8895/eval?url=http://110.35.161.182:8886&mode=all&name=OF30-3

