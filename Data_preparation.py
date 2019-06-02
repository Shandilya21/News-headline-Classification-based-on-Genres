import numpy as np 
import json


f = open("News_Cat_Dataset.json","r",encoding='utf8')
g = open("JSON_Extracted.txt","w+",encoding="utf8")

for i in f:
	j = json.loads(i)
	j = j["headline"]
	h = json.dumps(j)
	a = json.JSONDecoder().raw_decode(h)
	g.write(str(a[0]))
	g.write('\n')



category = open("Target_category.txt","w+",encoding="utf8")
for s in f:
	k = json.loads(s)
	k = k["category"]
	p = json.dumps(k)
	b = json.JSONDecoder().raw_decode(p)
	category.write(str(b[0]))
	category.write('\n')


