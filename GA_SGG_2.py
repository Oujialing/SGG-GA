import numpy as np
from sko.GA import GA, GA_TSP
import json
import os
import math
import h5py
import numpy as np
import json
import pickle as cp 
import argparse
import torch
from collections import Counter
'''

[[[score,[s,p,o],[bs,bo]],[score,[s,p,o],[bs,bo]],...], #
 [[score,[s,p,o],[bs,bo]],[score,[s,p,o],[bs,bo]],...], #
 bs=[xc,yc,w,h]
 bo=[xc,yc,w,h]

GT:
[[[[s,p,o],[bs,bo]],[[s,p,o],[bs,bo]],...],
 [[[s,p,o],[bs,bo]],[[s,p,o],[bs,bo]],...],
 ...]
'''
def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser()
	# image_paths file format: [ path of image_i for image_i in images ]
	# the order of images in image_paths should be the same with obj_dets_file
	parser.add_argument('--dataset_name',dest='dataset_name',help='vrd/vg150/vg200/ ',default='vrd',type=str)
	parser.add_argument('--model_name',dest='model_name',help='VtransE/motif/RelDN/KERN ',default='VtransE',type=str)
	parser.add_argument('--iter_num',dest='iter_num',help='iteration number',default='20',type=int)
	
	args = parser.parse_args()
	return args


#tasktype = 'sgcls'
tasktype = 'predcls'

def distance(bs,bo):
	sxc = (bs[0]+bs[2])/2.0
	syc = (bs[1]+bs[3])/2.0
	oxc = (bo[0]+bo[2])/2.0
	oyc = (bo[1]+bo[3])/2.0
	return math.sqrt((sxc-oxc)**2+(syc-oyc)**2)

def getSGGRes(model_name,dataset_name,trainortest,mode):
	if model_name=="motif":
		dir = '/home/lab/zmr/motif/wby_'+mode+'_'+trainortest+'_pred_entries_graph.pkl'
		
	else:
		raise ValueError()

	print("loading SGG file: ",dir)
	with open(dir,'rb') as f:
		T=cp.load(f)
	#f = open(dir,'rb')
	#T = cp.load(f)
	#f.close()
	return T


def getGT(model_name,dataset_name,trainortest,mode):
	#GT_dir = 'GT_'+dataset_name+'.pkl'
	if model_name=="motif":
		GT_dir = '/home/lab/zmr/motif/wby_predcls_'+trainortest+'_gt_entries.pkl'
	else:
		raise ValueError()
	print("loading Ground truth file: ", GT_dir)
	f = open(GT_dir,'rb')
	R = cp.load(f)
	f.close()
	return R

def getObjPrednum(dataset_name):
	if dataset_name=="vrd":
		predicate_num = 70
		obj_num = 100
	elif dataset_name=="vg200":
		predicate_num = 100
		obj_num = 200
	elif dataset_name=="vg150":
		predicate_num = 51
		obj_num = 151	
	else:
		raise ValueError()

	return obj_num,predicate_num

def getPfromR(R,dataset_name):
	# calculate SP and N from R
	object_num,predicate_num = getObjPrednum(dataset_name)
	SP = {}
	min_n = 10000000000

	for obj1 in range(object_num):
		SP[obj1]={}
		for obj2 in range(object_num):
			#print(obj1,"-",obj2)
			SP[obj1][obj2]=[]
	
	for i in range(len(R)):
		RI = R[i]
		cn = 0
		for j in range(len(RI)):
			#print(R[j])
			rel,_ = RI[j]
			sub,pre,obj = rel
			SP[sub][obj].append(pre)
			cn += 1
		if cn<min_n:
			min_n=cn

	for obj1 in range(object_num):
		for obj2 in range(object_num):
			tmp = SP[obj1][obj2]
			pred_counter = Counter(tmp)
			all_so = sum(pred_counter.values())

			res = []
			for i in range(predicate_num):
				if all_so==0:
					res = np.zeros(predicate_num)
					break
				if pred_counter[i]==0:
					res.append(0.0)
				else:
					tmp_num = pred_counter[i]*1.0/all_so*1.0
					res.append(tmp_num)
			SP[obj1][obj2] = res
	return SP,min_n

T = getSGGRes("motif","vg150",'train',tasktype)#getSGGRes(model_name,dataset_name,trainortest,mode)
R = getGT("motif","vg150","train",tasktype)#getGT(model_name,dataset_name,trainortest,mode)
SP,N = getPfromR(R,"vg150")

def ReSet(alpha,beta):
	# update R
	#R=RUQ
	R_new = []
	for im in range(len(T)):
		cnt=0
		TI = T[im]
		RI = R[im]
		RI_new = RI
		Qpre = []
		for i in range(len(TI)):
			smi,t,tb = TI[i]	#score,[label],[bbox]
			s,p,o = t
			bs,bo = tb
			sfi = SP[s][o][p]
			shdti = 1.0/(1.0+distance(bs,bo))
			si = smi + alpha*sfi + beta*shdti
			Qpre.append([si,TI[i][1:]])

		tmp_N = N if N<len(Qpre) else len(Qpre)
		#QI = sorted(enumerate(Qpre), key=lambda x: x[0],reverse=True)[:tmp_N]
		QI = sorted(Qpre, key=lambda x: x[0],reverse=True)[:tmp_N]

		# add Q
		for i in range(tmp_N):
			_,t = QI[i]
			if t not in RI:
				RI_new.append(t)

		R_new.append(RI_new)

	return R_new


def fitness(p):
	alpha,beta = p
	
	#T = getSGGRes("motif","vg150",1)
	#R = getGT("vg150",1)
	#SP,N = getPfromR(R,dataset_name)
	#all TI[i] [score,[s,p,o],[bs,bo]]
	#RI[i]	[[s,p,o],[bs,bo]]
	#print("N: ",N)
	#print("T: ",len(T))
	cnt_all = 0
	for im in range(len(T)):
		cnt=0
		TI = T[im]
		RI = R[im]
		Qpre = []
		for i in range(len(TI)):
			smi,t,tb = TI[i]	#score,[label],[sbbox,obbox]
			s,p,o = t
			bs,bo = tb
			sfi = SP[s][o][p]
			shdti = 1.0/(1.0+distance(bs,bo))
			si = smi + alpha*sfi + beta*shdti
			Qpre.append([si,TI[i][1:]])

		tmp_N = N if N<len(Qpre) else len(Qpre)
		#QI = sorted(enumerate(Qpre), key=lambda x: x[0],reverse=True)[:tmp_N]
		QI = sorted(Qpre, key=lambda x: x[0],reverse=True)[:tmp_N]
		
		QI1 = []	# Q
		QI2 = []	# Q-R

		for i in range(tmp_N):
			_,t = QI[i]
			arg_max = -1

			if t in RI:
				QI1.append(t)
			else:
				QI2.append(t)
				cnt += 1

		cnt_all += cnt
	#return |Q-R|
	return cnt_all 



def trainab(model_name,dataset_name,iter_num):
	# init T,R,SP
	T = getSGGRes(model_name,dataset_name,"train",tasktype)
	R = getGT(model_name,dataset_name,"train",tasktype)
	SP,N = getPfromR(R,dataset_name)
	print("T info: ",len(T),len(T[0]),T[0][0])
	print("R info: ",len(R),len(R[0]),R[0][0])
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	dd = 1e23
	# start iteration

	for i in range(iter_num):
		ga = GA(func=fitness, n_dim=2, size_pop=10, max_iter=50, \
			lb=[0.0, 0.0], ub=[dd, dd], precision=1e-7)
		#ga = GA(func=fitness, n_dim=2, size_pop=10, max_iter=4, \
		#	lb=[0.0, 0.0], ub=[dd, dd], precision=1e-7)

		ga.to(device=device)	#GPU
		best_x, best_y = ga.run()# best_x={alpha,beta},best_y=Q-R
		alpha, beta = best_x
		print('iteration ',i,': best alpha and beta: ', alpha,beta)
		print('best_y:', best_y)
		if best_y==0:
			print('best {alpha,beta}:',best_x,'best_y',best_y)
			end_cnt = fitness(best_x)
			print("detect if zeros: ",end_cnt)
			if end_cnt==0:
				break
		else:
			R = ReSet(alpha,beta)
			SP,_ = getPfromR(R,dataset_name)
	return best_x
			
def getTestRes(model_name,dataset_name,p):
	# get test result with alpha and beta
	alpha,beta = p
	T = getSGGRes(model_name,dataset_name,"test",tasktype)
	R = getGT(model_name,dataset_name,"test",tasktype) #train dataset prob
	SP,_ = getPfromR(R,dataset_name)
	T_new = []
	for im in range(len(T)):
		cnt=0
		TI = T[im]
		TI_new = []
		for i in range(len(TI)):
			smi,t,tb = TI[i]	#score,[label],[bbox]
			s,p,o = t
			bs,bo= tb
			sfi = SP[s][o][p]
			shdti = 1.0/(1.0+distance(bs,bo))
			si = smi + alpha*sfi + beta*shdti
			TI_new.append([si,TI[i][1],TI[i][2]])
		T_new.append(TI_new)

	print("writing test file...")
	print("T_new info: ",T[0][0])
	testdir = 'BDR/outputs/'+model_name+'_'+dataset_name+'_'+'testRes_501020.pkl'
	f = open(testdir,'wb')
	cp.dump(T_new, f, cp.HIGHEST_PROTOCOL)
	f.close()
	return T_new



if __name__ == '__main__':
	args = parse_args()
	print('Called with args:')
	print(args)

	best_x = trainab(args.model_name,args.dataset_name,args.iter_num)

	T_new = getTestRes(args.model_name,args.dataset_name,best_x)

	recall_number=20
	#PredRecall = calPredRecall(T_new,recall_number,tasktype)

