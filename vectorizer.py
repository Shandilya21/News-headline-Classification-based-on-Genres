import h5py
from allennlp.modules.elmo import Elmo, batch_to_ids
import tensorflow as tf 
import tensorflow_hub as hub
from torch.multiprocessing import Manager, Process, Queue, get_logger


logger = get_logger()
get_logger.server()

option_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

Data = open("Extracted_Cleaned.txt","r",encoding="utf8")
sentence=[]
for i in Data:
	if i=='\n':
		continue
	sentence.append(i.strip().split())


for j in sentence:
	elmo = Elmo(option_file,weight_file,2,dropout=0)
	character_ids = batch_to_ids(j)
	embedding = elmo(character_ids)
	


