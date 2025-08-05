import matplotlib.pyplot as plt
import numpy as np
import pickle
import multiprocessing
from multiprocessing.pool import Pool
from multiprocessing import get_context
import subprocess
import os
import time
import cost_abs as rc


class Evolve():

	'''
	Evolve is a an evolutionary algorithm which is designed be independent from model execution and evaluation to provide maximum flexibility in cost function design.
	
	Evolve interacts with models/cost functions through the imported method rc()
	
	Evolve expects the following parameters (required):
	
	param_ranges: list - [[min,max,increment],[min,max,increment],...]
	
	But also accepts the following parameters (optional):
	
	threshold_type: str, 'pool','individual' - this parameter designates whether it is pool or individual performances which triggers ENBO termination criteria
	score_threshold: int/float - this parameter should correspond to your desired cost function performance.
	generation_size: int - the number of individuals to be evaluated in parallel in each generation. Should not be greater than the number of processors
	mating_pool_size: int - the size of your mating pool should be adjusted according to the size/complexity of your open parameter space
	generation_limit: int - the number of generations (*generation_size) that will be evaluated before ENBO timeout
	dynamic_mutation: boolean - should mutation rate be ajusted according to mean pool score - mutation rate determined by class method update_mutation_rate() and ranges from 0.5-0.0
	mutation_rate: float - initial mutation rate. Must be a value between 1.0-0.0
	seed_generation: li - List of parameters equal in length to mating_pool_size which may be used to initialize Evolve.
	
	Evolve creates and manages two output files:
	
	'MatingPool_enbo.pickle' - stores the current mating pool
	'enbo.pickle' - stores all evaluated candidates throughout the process
	'''
	
	def __init__(self,param_ranges = [],threshold_type='pool',score_threshold=100,generation_size=4,mating_pool_size=20,generation_limit=300,dynamic_mutation=True,mutation_rate=0,seed_generation=None):
		start_time = time.time()
		self.threshold_type = threshold_type
		self.threshold=False
		self.max_pool_size = mating_pool_size
		self.score_threshold=score_threshold
		self.check_set_generation_size(generation_size)
		self.generation_limit = generation_limit
		self.generations = 0
		self.param_ranges = param_ranges
		self.f_params = []
		self.metrics_params = []
		if not self.param_ranges:
			return()
		
		self.initialize_mating_pool(seed_generation=seed_generation)
		print('Pool initialized')
		end_time = time.time()
		self.time_it(start_time)
		self.original_pool_mean = np.mean(self.mating_pool_evaluations)
		print('Original pool mean score: ',self.original_pool_mean)
		self.dynamic_mutation = dynamic_mutation
		if self.dynamic_mutation:
			self.mutation_rate = 0.5
		else:
			self.mutation_rate = mutation_rate	
		
		print('Original mutation rate: ',self.mutation_rate)	
		
	def check_set_generation_size(self,gen_size):
		if gen_size > multiprocessing.cpu_count():
			gen_size = multiprocessing.cpu_count()
			print('It is not recommended to have generation size greater than number of processors on machine. Reduced generation size to cpu count -1.')
		
		self.generation_size = gen_size
	
	def initialize_mating_pool(self,seed_generation=None):
		if seed_generation is not None:
			self.mating_pool = seed_generation
		else:
			self.mating_pool = []
			print('Initialized mating pool with '+str(self.max_pool_size)+' random entries')
		
		while len(self.mating_pool) < self.max_pool_size:
			self.mating_pool.append(self.get_random_individual())

		self.mating_pool_evaluations = []
		for i, group in enumerate(self.divide_chunks(self.mating_pool,self.generation_size)):
			self.f_params.append(group)
			metrics_total = self.evaluate_generation(group)
			self.metrics_params+=metrics_total
			scores = [item[0] for item in metrics_total]
			print('current generation: ', i+1)
			self.mating_pool_evaluations+=scores
			print('current lowerst score: ', min(self.mating_pool_evaluations))
			index_min = self.mating_pool_evaluations.index(min(self.mating_pool_evaluations))
			print('current best param: ', self.mating_pool[index_min])

	def sort_children_by_varbs(self,children,varbs,spikes):
		newspikes = []
		newvarbs = []
		for child in children:
			which = varbs.index(child)
			newvarbs.append(varbs[which])
			newspikes.append(spikes[which])
		
		return newvarbs, newspikes
	
	def update_threshold(self,scores):
		if self.threshold_type == 'pool' and self.mating_pool_evaluations != []:
			if np.mean(self.mating_pool_evaluations) < self.score_threshold:
				self.threshold=True
		
		else:
			for score in scores:
				if score < self.threshold:
					self.threshold = True
	
	def evaluate_generation(self,children):
		pool = Pool(self.generation_size)
		metrics_total = pool.map(rc.run_cost_simulation, children)
		scores = [item[0] for item in metrics_total]
		self.update_threshold(scores)
		return metrics_total
	
	def divide_chunks(self, l, n):
		for i in range(0,len(l),n):
			yield l[i:i+n]
	
	def check_add_to_mating_pool(self,children,scores):
		for c,child in enumerate(children):
			pool_max = np.max(self.mating_pool_evaluations)
			if scores[c] < pool_max:
				individual_to_remove = self.mating_pool_evaluations.index(pool_max)
				self.mating_pool_evaluations[individual_to_remove] = scores[c]
				self.mating_pool[individual_to_remove] = child
	
	def parse_results_line(self,line):
		return([float(item) for item in line.strip('[]\n').split(',')])

	def load_parallel_results(self):
		with open('parallelresults.txt','r') as f:
			results = [self.parse_results_line(item) for item in f.readlines()]
		
		varbs = results[::2]
		sptimes = results[1::2]
		return(varbs,sptimes)
	
	def mating_pool_snapshot(self):
		pool_min = np.min(self.mating_pool_evaluations)
		best_individual_index = self.mating_pool_evaluations.index(pool_min)
		best_score = self.mating_pool_evaluations[best_individual_index]
		best_individual = self.mating_pool[best_individual_index]
		print(best_individual,'best_individual',best_score,'best_score')
		
		with open('MatingPool_enbo.pickle','wb') as f:
			pickle.dump([self.mating_pool,self.mating_pool_evaluations],f)
	
	def get_random_individual(self):
		A = []
		for pr in self.param_ranges:
			#print(pr)
			A.append(np.random.choice(np.arange(pr[0],pr[1],pr[2])))
		
		return(A)
	
	def get_new_generation(self):
		children = []
		while len(children) < self.generation_size:
			mom,dad = [self.mating_pool[i] for i in np.random.choice(range(self.max_pool_size),2)]
			children+=self.crossover(mom,dad)
		
		return([self.mutate(child) for child in children[:self.generation_size]])
	
	def crossover(self,varsA,varsB):
		crossover_point = np.random.choice(range(len(varsA)))
		if np.random.random() > 0.5:
			newA = varsA[:crossover_point]+varsB[crossover_point:]
			newB = varsB[:crossover_point]+varsA[crossover_point:]
		
		else:
			newB = varsA[:crossover_point]+varsB[crossover_point:]
			newA = varsB[:crossover_point]+varsA[crossover_point:]
		
		newC = [(varsA[i]+varsB[i])/2.0 for i in range(len(varsA))]
		return(newA,newB,newC)
	
	def mutate(self,varbs):
		for v,var in enumerate(varbs):
			if np.random.random() < self.mutation_rate:
				varbs[v]=np.random.uniform(self.param_ranges[v][0],self.param_ranges[v][1])
				if varbs[v] < self.param_ranges[v][0]:
					varbs[v] = self.param_ranges[v][0]
				
				if varbs[v] > self.param_ranges[v][1]:
					varbs[v] = self.param_ranges[v][1]
		
		return(varbs)
	
	def update_mutation_rate(self):
		pool_mean = np.mean(self.mating_pool_evaluations)
		print('New pool mean score: ',pool_mean)
		self.mutation_rate = 0.5/(1+np.exp(1+(-1*pool_mean)/25))		#0.5/(1+np.exp(1+(-1*pool_mean)/25))
		print('New mutation rate: ',self.mutation_rate)
	
	def time_it(self,s):
		e = time.time()
		print('elapsed time: '+str(round((e-s)/60.0,1))+' minutes')

	
	def run(self):
		start_time = time.time()
		self.threshold=False
		while not self.threshold and self.generations < self.generation_limit:
			print('current generation: {}'.format(self.generations))
			self.mating_pool_snapshot()
			children = self.get_new_generation()
			self.f_params.append(children)
			metrics_total = self.evaluate_generation(children)
			scores_total = [item[0] for item in metrics_total]
			self.metrics_params+=metrics_total
			self.check_add_to_mating_pool(children,scores_total)
			if self.dynamic_mutation:
				self.update_mutation_rate()
			self.time_it(start_time)
			self.generations+=1
		self.mating_pool_snapshot()
		with open('enbo.pickle','wb') as f:
			pickle.dump([self.f_params, self.metrics_params],f)


if __name__ == "__main__":
	params = [
				# Soma
				[0.0005, 0.002, 0.0001],  # 0 CaN_s
				[0.0001, 0.001, 0.0001],  # 1 CaL_s
				[1e-5, 0.0005, 0.0001],  # 2 Ih
				[1e-5, 0.01, 0.001],  # 3 KDR
				[1e-5, 0.03, 0.005],  # 4 Kv31
				[1e-5, 1e-4, 1e-5],  # 5 sKCa
				[1e-7, 2e-5, 2e-6],  # 6 NaL
				[1e-5, 0.01, 0.001],  # 7 Na
				[1e-7, 100e-6, 10e-6],  # 8 gpas
				# scaling factor for dend CaT, KDR, Kv31, sKCa, NaL, Na, CaN, CaL
				[1e-5, 0.015, 0.0015], [0, 0.01, 0.001], [0, 0.03, 0.005], [0, 1e-4, 1e-5], [0, 2e-5, 2e-6], [0, 0.01, 0.001],
				[0.0001, 0.005, 0.0005], [0.0001, 0.005, 0.0005],  # 9, 10, 11, 12, 13, 14, 15, 16
				[1e-5, 0.015, 0.0015], [0, 0.01, 0.001], [0, 0.03, 0.005], [0, 1e-4, 1e-5], [0, 2e-5, 2e-6], [0, 0.01, 0.001],
				[0.0001, 0.005, 0.0005], [0.0001, 0.005, 0.0005], # 17, 18, 19, 20, 21, 22, 23, 24
				[0.1, 1, 0.1],  # 25 r_kdr could potentially be even slower, but unlikely to be authentic
				[0.3, 2, 0.1],  # 26 r_na
				[1, 10, 0.1],  # 27 r_kv31 could potentially be even quicker
				[0.1, 1, 0.1],  # 28 r_cat
				[1, 8, 0.5],  # 29 r_cacum soma
				[0.8, 5, 0.5],  # 30 r_cacum dend
			]
	
	use_seed = False
	if use_seed:
		with open('MatingPool_enbo.pickle','rb') as f:
			seed = pickle.load(f)
		evol = Evolve(param_ranges=params,threshold_type='pool',score_threshold=30,generation_limit=0, generation_size=30,mating_pool_size=120,dynamic_mutation=True,mutation_rate=0.5, seed_generation=seed[0])
	else:
		evol = Evolve(param_ranges=params,threshold_type='pool',score_threshold=30,generation_limit=3000, generation_size=10,mating_pool_size=120,dynamic_mutation=False,mutation_rate=1)
	evol.run()
