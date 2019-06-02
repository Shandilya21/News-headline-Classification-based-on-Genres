import re
import Data_preparation


k = open("JSON_Extracted.txt","r",encoding="utf8")
h = open("Extracted_Cleaned.txt","w+",encoding="utf8")

for i in k:
	j=i
	j = re.sub(r':',' ',j)
	j = re.sub(r',',' ',j)
	j = re.sub(r'!',' ',j)
	j = re.sub(r'@',' ',j)
	j = re.sub(r'/',' ',j)
	h.write(j)
	# h.write('\n')



