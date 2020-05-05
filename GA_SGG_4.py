import numpy as np
from sko.GA import GA, GA_TSP
from sko.PSO import PSO
from sko.operators import ranking, selection, crossover, mutation

import json
import os
import math
from math import log
form math import exp
import h5py
import numpy as np
import json
import pickle as cp 
import argparse
import torch
import time
import tensorflow as tf
from collections import Counter
from functools import reduce
from collections import Counter

#nohup python -u GA_SGG_2.py --dataset_name vg150 --model_name motif >log_motif_predcls_10_50_20.txt 2>&1 &
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
	parser.add_argument('--GA',dest='GA',help='GA or not',default=False,type=bool)
	parser.add_argument('--PSO',dest='PSO',help='PSO or not',default=False,type=bool)
	
	args = parser.parse_args()
	return args

tasktype = 'sgdet'

#tasktype = 'sgcls'
#tasktype = 'predcls'


def calDCG(QI):
	# order : score sort  [1,1,0,1,...]
	res = 0.0
	for i in range(len(QI)):
		res += QI[i]*1.0/log(2+i,2)
	return res

def computeArea(bb):
	return max(0, bb[2] - bb[0] + 1) * max(0, bb[3] - bb[1] + 1)

def computeIoU(bb1, bb2):
	ibb = [max(bb1[0], bb2[0]), \
		max(bb1[1], bb2[1]), \
		min(bb1[2], bb2[2]), \
		min(bb1[3], bb2[3])]
	iArea = computeArea(ibb)
	uArea = computeArea(bb1) + computeArea(bb2) - iArea
	return (iArea + 0.0) / uArea

def computeOverlap(detBBs, gtBBs):
	aIoU = computeIoU(detBBs[0], gtBBs[0]) # sub
	bIoU = computeIoU(detBBs[1], gtBBs[1]) # obj
	return min(aIoU, bIoU)	

def distance(bs,bo):
	sxc = (bs[0]+bs[2])/2.0
	syc = (bs[1]+bs[3])/2.0
	oxc = (bo[0]+bo[2])/2.0
	oyc = (bo[1]+bo[3])/2.0
	return math.sqrt((sxc-oxc)**2+(syc-oyc)**2)

def getSGGRes(model_name,dataset_name,trainortest,mode):
	if model_name=="motif":
		dir = '/home/lab/zmr/motif/wby_sgg_outputs/wby_'+mode+'_'+trainortest+'_pred_entries_graph.pkl'
	elif model_name=="VtransE":
		dir = '/home/ojl/BDR_GA/SGmodelRes/vtranse/ojl_'+mode+'_'+trainortest+'_'+dataset_name+'.pkl'
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
		GT_dir = '/home/lab/zmr/motif/wby_sgg_outputs/wby_predcls_'+trainortest+'_gt_entries.pkl'
	elif model_name=="VtransE":
		GT_dir = '/home/ojl/BDR_GA/SGmodelGT/vtranse/ojl_predcls_'+trainortest+'_'+dataset_name+'.pkl'
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
		if len(RI)<min_n:
			min_n=len(RI)

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
	min_n = 2
	return SP,min_n

T = getSGGRes("VtransE","vrd",'train',tasktype)#getSGGRes(model_name,dataset_name,trainortest,mode)
R = getGT("VtransE","vrd","train",tasktype)#getGT(model_name,dataset_name,trainortest,mode)
SP,N = getPfromR(R,"vrd")
print("N:",N)
def ReSet(alpha,beta):
	# update R
	#R=RUQ
	R_new = []
	cnt_=0
	cnt1=0
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
		cnt_+=tmp_N
		# add Q
		for i in range(tmp_N):
			#cnt_+=1
			_,t = QI[i]
			if t not in RI:
				RI_new.append(t)
				cnt1+=1

		R_new.append(RI_new)
	print("add triple: ",cnt_,cnt1)
	return R_new

def addsmallNDCG(alpha,beta,R):
	# update R
	#R=RUQ
	R_new = []
	cnt_=0
	cnt1=0
	min_ndcg=1e23
	min_ndcg_trip=[]
	for im in range(len(T)):
		cnt=0
		TI = T[im]
		RI = R[im]
		RI_new = RI
		Qpre = []
		if len(TI)==0:
			continue
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
		cnt_+=tmp_N
		QI1 = []
		QI2 = []
		# add Q
		for i in range(tmp_N):
			#cnt_+=1
			_,t = QI[i]
			'''
			if t in RI:
				QI1.append(1)
				cnt += 1
			else:
				QI1.append(0)
				QI2.append(t)	
			'''

			flag_tmp=0
			for k in range(len(RI)):			
				if RI[k][0][0]==t[0][0]+1 and RI[k][0][1]==t[0][1] and RI[k][0][2]==t[0][2]+1 \
					and computeIoU(RI[k][1][0],t[1][0])>=0.5\
					and computeIoU(RI[k][1][1],t[1][1])>=0.5:
					flag_tmp=1
					break
			if flag_tmp==1:
				QI1.append(1)
				cnt += 1
			else:
				QI1.append(0)
				QI2.append(t)

		if len(RI)>tmp_N:
			IDCG = [1]*tmp_N
		else:
			IDCG = [1]*len(RI)
			IDCG.extend([0]*(tmp_N-len(RI)))
		#print("IDCG: ",tmp_N,cnt,IDCG)
		#print("calIDCG: ",calDCG(IDCG))
		if cnt==0:
			R[im].append(QI2[0])
		else:
			ndcgI = calDCG(QI1)/calDCG(IDCG)
			if min_ndcg>ndcgI and min_ndcg>0:
				min_ndcg = ndcgI
				#print("ndcg: ",calDCG(QI1),calDCG(IDCG),QI1,IDCG)
				min_ndcg_trip = [im,QI2]
	#print("min_ndcg_trip: ",min_ndcg_trip)
	if min_ndcg!=1e23 and min_ndcg!=1: 
		#1e23 no min | 1 no notR
		print("min_ndcg",min_ndcg)
		R[min_ndcg_trip[0]].append(min_ndcg_trip[1][0])
	
	return R

def selection_tournament(algorithm, tourn_size):
    FitV = algorithm.FitV
    sel_index = []
    for i in range(algorithm.size_pop):
        aspirants_index = np.random.choice(range(algorithm.size_pop), size=tourn_size)
        sel_index.append(max(aspirants_index, key=lambda i: FitV[i]))
    algorithm.Chrom = algorithm.Chrom[sel_index, :]  # next generation
    return algorithm.Chrom

class MyGA(GA):
    def selection(self, tourn_size=3):
        FitV = self.FitV
        sel_index = []
        for i in range(self.size_pop):
            aspirants_index = np.random.choice(range(self.size_pop), size=tourn_size)
            sel_index.append(max(aspirants_index, key=lambda i: FitV[i]))
        self.Chrom = self.Chrom[sel_index, :]  # next generation
        return self.Chrom

    ranking = ranking.ranking


def fitness(p):
	alpha,beta = p
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
				print("len(QI1): ",len(QI1))
			else:
				QI2.append(t)
				cnt += 1

		cnt_all += cnt
	cnt_all += alpha**2 + beta**2 
	#print("pso: |Q-R|",cnt_all)
	#return |Q-R|
	return cnt_all 


def fitness1(p):
	w,alpha,beta = p
	
	cnt_all = 0
	for im in range(len(T)):
		cnt=0
		TI = T[im]
		RI = R[im]
		Qpre = []
		for i in range(len(TI)):
			smi,t,tb = TI[i]	#score,[label],[sbbox,obbox]
			s,p,o = t
			#print("triplet: ",TI[i])
			bs,bo = tb
			sfi = SP[s][o][p]
			shdti = 1.0/(1.0+distance(bs,bo))
			si = w*smi + alpha*sfi + beta*shdti
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
	cnt_all += alpha**2 + beta**2 +w**2
	return cnt_all 

def fitnessNDCG(p):
	alpha,beta = p
	sum_ndcg = 0.0

	for im in range(len(T)):
		cnt=0
		TI = T[im]
		RI = R[im]
		Qpre = []
		if len(TI)==0:
			continue
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

		for i in range(tmp_N):
			_,t = QI[i]
			'''
			#predcls
			if t in RI:
				QI1.append(1)
				cnt += 1
				print(cnt)
			else:
				QI1.append(0)
			'''
			#sgdet
			flag_tmp=0
			for k in range(len(RI)):			
				if RI[k][0][0]==t[0][0]+1 and RI[k][0][1]==t[0][1] and RI[k][0][2]==t[0][2]+1 \
					and computeIoU(RI[k][1][0],t[1][0])>=0.5\
					and computeIoU(RI[k][1][1],t[1][1])>=0.5:
					flag_tmp=1
					break
			if flag_tmp==1:
				QI1.append(1)
				cnt += 1
			else:
				QI1.append(0)

		'''
		Example:
		N=2

		GT triplet: t1,t2,t3
		prediction: t2,t4
		IDCG:[1,1]	-> calDCG(IDCG)
		DCG:[1,0]	-> calDCG(DCG)
		---------------
		GT triplet: t1
		prediction: t2,t4
		IDCG:[1,0]	-> calDCG(IDCG)
		DCG:[0,0]	-> calDCG(DCG)
		
		'''
		# ground truth triplet -> IDCG
		if len(RI)>tmp_N:
			IDCG = [1]*tmp_N
		else:
			IDCG = [1]*len(RI)
			IDCG.extend([0]*(tmp_N-len(RI)))
		#print(tmp_N,cnt,"IDCG_list: ",IDCG)
		#print("IDCG",calDCG(IDCG))
		#IDCG =  sorted(QI1,reverse=True)

		if calDCG(IDCG)==0:
			ndcgI = 0
		else:
			ndcgI = calDCG(QI1)/calDCG(IDCG)
		sum_ndcg += ndcgI
		
	w=0.5
	sum_ndcg -= w*(alpha**2 + beta**2) #max sum_ndcg

	return -sum_ndcg 

def trainabPSO(model_name,dataset_name,iter_num):
	# init T,R,SP
	T = getSGGRes(model_name,dataset_name,"train",tasktype)
	R = getGT(model_name,dataset_name,"train",tasktype)
	SP,N = getPfromR(R,dataset_name)
	print("T info: ",len(T),len(T[0]),T[0][0])
	print("R info: ",len(R),len(R[0]),R[0][0])
	print("N: ",N)
	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# start iteration

	for i in range(iter_num):

		#pso = PSO(func=fitness, dim=2, pop=100, max_iter=50, \
		#	lb=[0.0, 0.0], ub=[1e6, 1e6],w=0.8, c1=0.5, c2=0.5)
		
		#pso = PSO(func=fitness1, dim=3, pop=100, max_iter=100, \
		#	lb=[0.0, 0.0,0.0], ub=[1e6,1e6, 1e6],w=0.8, c1=2, c2=2)
		pso = PSO(func=fitnessNDCG, dim=2, pop=50, max_iter=50, \
			lb=[0.0, 0.0], ub=[1e6, 1e6],w=0.8, c1=0.5, c2=0.5)
		

		#pso = PSO(func=fitness, dim=2, pop=10, max_iter=5, \
		#	lb=[0.0, 0.0], ub=[1e6, 1e6],w=0.8, c1=2, c2=2)
		
		#pso.to(device=device)	#GPU
		start_time = time.time()
		pso.run()# best_x={alpha,beta},best_y=Q-R
		
		print("run time: ",time.time() - start_time)
		best_x = pso.gbest_x
		alpha, beta = pso.gbest_x
		best_y = pso.gbest_y
		print('iteration ',i,': best alpha and beta: ', best_x)
		print('best_y:', best_y)
		if best_y==0:
			print('best {alpha,beta}:',best_x,'best_y',best_y)
			end_cnt = fitnessNDCG(best_x)
			print("detect if zeros: ",end_cnt)
			if end_cnt==0:
				break
		else:
			#R = ReSet(alpha,beta)
			R = addsmallNDCG(alpha,beta,R)
			SP,_ = getPfromR(R,dataset_name)
			cnt_now=fitnessNDCG(best_x)
			print(i," iter :", cnt_now)
	return best_x



def trainabGA(model_name,dataset_name,iter_num):
	# init T,R,SP
	T = getSGGRes(model_name,dataset_name,"train",tasktype)
	R = getGT(model_name,dataset_name,"train",tasktype)
	SP,N = getPfromR(R,dataset_name)
	print("T info: ",len(T),len(T[0]),T[0][0])
	print("R info: ",len(R),len(R[0]),R[0][0])
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	# start iteration
	low = [0.0,0.0]
	high = [1e6,1e6]
	
	for i in range(iter_num):
		#ga = GA(func=fitness, n_dim=2, size_pop=4, max_iter=10, \
		#	lb=[0.0, 0.0], ub=[1.0, 1.0], precision=1e-7)
		#ga = GA(func=fitness, n_dim=2, size_pop=6, max_iter=50, \
		#	lb=low, ub=high, precision=1e-7)
		
		#ga = GA(func=fitness, n_dim=2, size_pop=50, max_iter=50, \
		#	lb=low, ub=high, precision=1e-7)
		ga = GA(func=fitnessNDCG, n_dim=2, size_pop=100, max_iter=100, \
			lb=low, ub=high, precision=1e-7)

	
		ga.register(operator_name='selection', operator=selection_tournament, tourn_size=3).\
			register(operator_name='ranking', operator=ranking.ranking). \
    			register(operator_name='crossover', operator=crossover.crossover_2point). \
    			register(operator_name='mutation', operator=mutation.mutation)
		ga.to(device=device)	#GPU
		start_time = time.time()
		best_x, best_y = ga.run()# best_x={alpha,beta},best_y=Q-R
		print("run time: ",time.time() - start_time)
		
		alpha, beta = best_x
		print('iteration ',i,': best alpha and beta: ', alpha,beta)
		print('best_y:', best_y)
		print("sum_ndcg: ",-best_y-0.5*(alpha**2+beta**2))
		if best_y==0:
			print('best {alpha,beta}:',best_x,'best_y',best_y)
			end_cnt = fitnessNDCG(best_x)
			print("detect if zeros: ",end_cnt)
			if end_cnt==0:
				break
		else:
			#R = ReSet(alpha,beta)
			R = addsmallNDCG(alpha,beta,R)
			SP,_ = getPfromR(R,dataset_name)
			cnt_now=fitness(best_x)
			print(i," iter :", cnt_now)

	return best_x


			
def trainabListWise(model_name,dataset_name,iter_num):
	T = getSGGRes(model_name,dataset_name,"train",tasktype)
	R = getGT(model_name,dataset_name,"train",tasktype)
	SP,N = getPfromR(R,dataset_name)
	print("T info: ",len(T),len(T[0]),T[0][0])
	print("R info: ",len(R),len(R[0]),R[0][0])
	
	N=20
	sm = []
	sf = []
	shdt = []
	QQ =[]
	for im in range(len(T)):
		cnt=0
		TI = T[im]
		RI = R[im]
		Qpre = []
		if len(TI)==0:
			continue
		smI=[]
		sfI = []
		shdtI = []
		for i in range(len(TI)):
			smi,t,tb = TI[i]	#score,[label],[sbbox,obbox]
			s,p,o = t
			bs,bo = tb
			sfi = SP[s][o][p]
			shdti = 1.0/(1.0+distance(bs,bo))
			si = smi + alpha*sfi + beta*shdti
			smI.append(smi)
			sfI.append(sfi)
			shdtI.append(shdti)
			Qpre.append([si,TI[i][1:]])
		sm.append(smI)
		sf.append(sfI)
		shdt.append(shdtI)

		for i in range(len(TI)):
			_,t = QI[i]
			if t in RI:
				QI1.append(1)
				cnt += 1
				print(cnt)
			else:
				QI1.append(0)

		QQ.append(QI1) #GT score
	
	'''
	- 计算topK概率(K=20,50,100)
	举个例子，K=5
	前K个的模型评分[si1,si2,si3,si4,si5]
	前K个的GT得分  [label1,label2,label3,label4,label5] 前K个三元组，若在GT里，则1，否则0  label_i=0/1

	- 用交叉熵来评价差异 
	Loss=-sum_i^K (si_i)log(label_i)

	- 最小化差异（梯度下降）
	调用tensorflow的梯度下降

	'''
	image_number=len(T)
	Tsm = tf.placeholder(tf.float32, shape=(image_number, None), name='model_score')
	Tsf = tf.placeholder(tf.float32, shape=(image_number, None), name='frequence_score')
	Tshdt = tf.placeholder(tf.float32, shape=(image_number, None), name='hdt_score')
	gts = tf.placeholder(tf.float32, shape=(image_number, None), name='label')

	a = tf.Variable(tf.random_normal([1]), name="alpha")
	b = tf.Variable(tf.random_normal([1]), name="beta")

	mscore = Tsm + a*Tsf + b*Tshdt
	#train_prediction= tf.nn.softmax(fF_it, name="train_prediction")
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gts, logits=mscore), name='loss')

	learning_rate = 0.01
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

	best=[0.0,0.0]
	with tf.Session() as sess:
		min_los=1000000
		for i in range(iter_num):
			_,lossa,alpha,beta = sess.run([optimizer,loss,a,b], feed_dict={Tsm:sm, Tsf:sf, Tshdt:shdt,gts:QQ})
			print(lossa,alpha,beta)
			if min_los>lossa:
				best_x = [alpha,beta]

	return best_x

def getTestRes(model_name,dataset_name,p,flag):
	# get test result with alpha and beta
	
	if len(p)==3:
		w,alpha,beta=p
	else:
		w=1.0
		alpha,beta = p
	T = getSGGRes(model_name,dataset_name,"test",tasktype)
	R = getGT(model_name,dataset_name,"test",tasktype) 
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
			si = w*smi + alpha*sfi + beta*shdti
			#print("score: ",si,w,smi,alpha,sfi,beta,shdti,TI[i])
			#raise ValueError()
			TI_new.append([si,TI[i][1],TI[i][2]])
		T_new.append(TI_new)

	print("writing test file...")
	print("T_new info: ",T[0][0])
	if flag:
		testdir = '../outputs/'+model_name+tasktype+'_'+dataset_name+'_'+'testRes_pso.pkl'
	else:
		testdir = '../outputs/'+model_name+tasktype+'_'+dataset_name+'_'+'testRes_ga.pkl'
	print("writing output file in :",testdir)
	f = open(testdir,'wb')
	cp.dump(T_new, f, cp.HIGHEST_PROTOCOL)
	f.close()
	return T_new,T

def calPredRecall(predictions,gts):

	print("len(prediction): ",len(predictions))
	print("len(gt_boxs): ",len(gts))
	img_number = len(predictions)
	dict_recall={20: 0.0, 50: 0.0, 100: 0.0}
	for im in range(img_number):
		predi = predictions[im]
		gt = gts[im]
		pred_to_gt = [[] for x in range(len(predi))]
		num_gts = len(gt)

		if len(predi)==0:
			continue
		
		pred = sorted(predi,key=lambda x:x[0],reverse=True)
		if len(pred)>1 and pred[0][0] <= pred[-1][0]+1e-8 :
			print("Somehow the relations weren't sorted properly: \n{}".format(pred))

		for j in range(len(pred)):
			_,t,t_box = pred[j]

			for k in range(num_gts):			
				## and gt[k][1]==t_box
				if gt[k][0]==t \
					and computeIoU(gt[k][1][0],t_box[0])>=0.5\
					and computeIoU(gt[k][1][1],t_box[1])>=0.5:
				 	pred_to_gt[j].append(k)

		for k in dict_recall:
			match =reduce(np.union1d,pred_to_gt[:k])
			dict_recall[k] += float(len(match))/float(num_gts)

	for k in dict_recall:
		dict_recall[k]=float(dict_recall[k])/float(img_number)

	return dict_recall

def getUnionBB(aBB, bBB):
	return [min(aBB[0], bBB[0]), \
		min(aBB[1], bBB[1]), \
		max(aBB[2], bBB[2]), \
		max(aBB[3], bBB[3])]	


def calPhraseRecall(predictions,gts):
	print("phrase detection:----------------------------------------------------------")
	print("len(prediction): ",len(predictions))
	print("len(gt_boxs): ",len(gts))
	img_number = len(predictions)
	dict_recall={20: 0.0, 50: 0.0, 100: 0.0}
	for im in range(img_number):
		predi = predictions[im]
		gt = gts[im]
		pred_to_gt = [[] for x in range(len(predi))]
		num_gts = len(gt)

		if len(predi)==0:
			continue
		
		pred = sorted(predi,key=lambda x:x[0],reverse=True)
		if len(pred)>1 and pred[0][0] <= pred[-1][0]+1e-8 :
			print("Somehow the relations weren't sorted properly: \n{}".format(pred))

		for j in range(len(pred)):
			_,t,t_box = pred[j]

			for k in range(num_gts):			
				## and gt[k][1]==t_box
				gt_union = getUnionBB(gt[k][1][0],gt[k][1][1])
				t_union = getUnionBB(t_box[0],t_box[1])
				
				if gt[k][0]==t \
					and computeIoU(gt_union,t_union)>=0.5:
				 	pred_to_gt[j].append(k)

		for k in dict_recall:
			match =reduce(np.union1d,pred_to_gt[:k])
			dict_recall[k] += float(len(match))/float(num_gts)

	for k in dict_recall:
		dict_recall[k]=float(dict_recall[k])/float(img_number)

	return dict_recall


if __name__ == '__main__':
	args = parse_args()
	print('Called with args:')
	print(args)
	
	print("Training.......................................")
	if args.GA:
		best_x = trainabGA(args.model_name,args.dataset_name,args.iter_num)
	if args.PSO:
		best_x = trainabPSO(args.model_name,args.dataset_name,args.iter_num)
	
	#a=0.39741185 
	#b = 0.44839188
	#b=0.8555197629642345
	#best_x = [a,b]
	#best_x = [0.42584828749141085,0.15197840864026277]
	#best_x = [0.84495877,0.36265679,0.18999643]
	#best_x = [0.5206967443882901,0.5218322485234536] #best_y = 0.543434
	#best_x = [0.7151245995374163,0.2266845058329765]
	#best_x = [788594.8478815856,4739.9284724750005,149.7686200764377]
	#best_x = [6.355428809002112,0.9244783996110143,14.239028416796728]
	print("Test.......................................")
	T_new,T_old = getTestRes(args.model_name,args.dataset_name,best_x,args.PSO)

	# eval
	print("Evaluation........"+tasktype+"...............................")
	R_test = getGT(args.model_name,args.dataset_name,"test",tasktype)
	#dict_recall_old = calPhraseRecall(T_old,R_test)#
	dict_recall_old = calPredRecall(T_old,R_test)
	for k in dict_recall_old:
		print(args.model_name," R@",k,": ",dict_recall_old[k])

	#dict_recall = calPhraseRecall(T_new,R_test)
	dict_recall = calPredRecall(T_new,R_test)
	for k in dict_recall:
		print("R@",k,": ",dict_recall[k])


