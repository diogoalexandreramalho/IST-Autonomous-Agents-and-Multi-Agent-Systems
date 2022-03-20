import sys
import math
import itertools
from collections import Counter

#########################
### A: AGENT BEHAVIOR ###
#########################

class Agent:
	def __init__(self, options):
		options[-1] = options[-1][:-1]

		for opt in options:
			opt = opt.split('=')
			if opt[0] == "cycle":
				self.nr_steps = int(opt[1])
			elif opt[0] == "decision":
				self.decision = opt[1]
			elif opt[0] == "restart":
				self.restart = int(opt[1])
			elif opt[0] == "memory-factor":
				self.memory_factor = float(opt[1])
			elif opt[0] == "agents":
				self.agents = opt[1].strip('][').split(',')
			elif opt[0] == "concurrency-penalty" or opt[0] == "concurrentCost":
				self.cc_penalty = int(opt[1])
			else:
				pass

		if self.decision != "homogeneous-society" and self.decision != "heterogeneous-society":
			self.cc_penalty = 0

		if self.decision == "rationale" or self.decision == "flexible":
			self.agents = 'A'

		if self.decision == "flexible":
			self.hasNegative = []
			self.split_action = {}
			self.split_string = ""

		self.utilities = {}
		self.possible_tasks = []
		self.step = 0
		self.gain = 0
		
		self.tasks = {}  # {'Agent':{"Ti": [utility, ground_truth, observed_utilities, steps]}}
		self.actions = {}  # {'Agent': action}
		self.counters = {}  # {'Agent': counter}
		for agent in self.agents:
			self.tasks[agent] = {} 
			self.actions[agent] = ""
			self.counters[agent] = self.restart



	def update_agent_utility(self, perception_type, utility):
		observed_utilities = self.tasks[perception_type][self.actions[perception_type]][2] + [utility]
		steps = self.tasks[perception_type][self.actions[perception_type]][3] + [self.step]

		# calculation of new utility
		steps_total = sum([i**self.memory_factor for i in steps])
		probs = [(step**self.memory_factor)/steps_total for step in steps]
		new_utility = 0
		for i in range(len(probs)):
			new_utility += probs[i] * observed_utilities[i]
		new_utility = round(new_utility,2)
		self.tasks[perception_type][self.actions[perception_type]] = [new_utility, 1, observed_utilities, steps]

	def update_agent_utility_with_task(self, agent, task, utility):
		observed_utilities = self.tasks[agent][task][2] + [utility]
		steps = self.tasks[agent][task][3] + [self.step]

		# calculation of new utility
		steps_total = sum([i**self.memory_factor for i in steps])
		probs = [(step**self.memory_factor)/steps_total for step in steps]
		new_utility = 0
		for i in range(len(probs)):
			new_utility += probs[i] * observed_utilities[i]
		new_utility = round(new_utility,2)
		self.tasks[agent][task] = [new_utility, 1, observed_utilities, steps]

	
	def update_flexible_utility(self, task, utility):
		observed_utilities = self.tasks['A'][task][2] + [utility]
		steps = self.tasks['A'][task][3] + [self.step]

		# calculation of new utility
		steps_total = sum([i**self.memory_factor for i in steps])
		probs = [(step**self.memory_factor)/steps_total for step in steps]
		new_utility = 0
		for i in range(len(probs)):
			new_utility += probs[i] * observed_utilities[i]
		new_utility = round(new_utility,2)
		self.tasks['A'][task] = [new_utility, 1, observed_utilities, steps]


	def perceive(self, input):

		perception = input[:-1].split(' ')
		perception_type = perception[0]
		

		if self.decision == "flexible" and len(self.split_action) > 0:
			#gets perceptions
			first_task, u_first = perception[1].split(',')[0].split('{')[1].split("=")
			second_task, u_second = perception[1].split(',')[1].split('}')[0].split("=")

			u_first = int(u_first)
			u_second = int(u_second)
			p_first_task = self.split_action[first_task]
			p_second_task = self.split_action[second_task]

			self.update_flexible_utility(first_task, u_first)
			self.update_flexible_utility(second_task, u_second)
			
			self.gain = self.gain + u_first * p_first_task + u_second * p_second_task
			self.split_action = {}
		else:
			utility = int(perception[1].split("=")[1])

			if self.decision == "flexible" and utility < 0 and perception_type[0] != "T":
				self.hasNegative += [self.actions[perception_type]]

			if perception_type[0] == "T":
				self.possible_tasks += [perception_type]
				self.utilities[perception_type] = []
				for agent in self.tasks:
					self.tasks[agent][perception_type] = [utility, 0, [], [], 0]
			elif self.decision == "homogeneous-society":
				self.gain += utility
				self.utilities[self.actions[perception_type]] += [utility]
			else:
				self.gain += utility
				self.update_agent_utility(perception_type, utility)




	def decide_act(self):
		self.step += 1
		#print("Step {}".format(self.step))

		flag = 0

		for task in self.utilities:
			if len(self.utilities[task]) > 0:
				flag = 1
				break

		if self.decision == "homogeneous-society" and flag:
			#compute utility per task
			for task in self.possible_tasks:
				if len(self.utilities[task]) > 0:
					self.utilities[task] = [sum(self.utilities[task]) / len(self.utilities[task])]
			
			for task in self.utilities:
				if len(self.utilities[task]) > 0: 
					for agent in self.tasks:
						utility = self.utilities[task][0]
						self.update_agent_utility_with_task(agent, task, utility)

			for task in self.possible_tasks:
				self.utilities[task] = []

		#print(self.tasks)
		if self.decision == "homogeneous-society" or self.decision == "heterogeneous-society":
			# current action for each agent in order to compare with possible future tasks
			current_actions = []
			agents = []
			for agent in self.actions:
				agents += [agent]
				current_actions += [self.actions[agent]]

			# generate all possible combination of tasks 
			tasks_combinations = list(itertools.product(self.possible_tasks,repeat=len(self.agents)))
			
			best_expected_utility = -math.inf
			best_combination = ()

			for combination in tasks_combinations:
				# subtract concurrency penalty
				task_counter = dict(Counter(combination))
				discounted_tasks = []
				for task in task_counter:
					if task_counter[task] >= 2:
						discounted_tasks += [task]

				# compute expected utility for the combination without concurrency penalty
				total_expected_utility = 0
				for i in range(len(agents)):
					if current_actions[i] == combination[i]:
						executed_steps = self.nr_steps - self.counters[agents[i]]
					else:
						executed_steps = self.nr_steps - self.restart

					if executed_steps < 0:
						executed_steps = 0

					#print("agent: {}".format(agents[i]))
					#print("combination: {}".format(combination[i]))

					task_utility = self.tasks[agents[i]][combination[i]][0]
					agent_expected_utility = task_utility * executed_steps

					if combination[i] in discounted_tasks:
						agent_expected_utility -= self.cc_penalty * executed_steps 

					total_expected_utility += agent_expected_utility



				#print("combination: {}".format(combination))
				#print("utility: {}".format(total_expected_utility))

				if total_expected_utility > best_expected_utility:
					best_expected_utility = total_expected_utility
					best_combination = combination

			for i in range(len(agents)):
				if self.actions[agents[i]] != best_combination[i]:
					self.counters[agents[i]] = self.restart

				if self.counters[agents[i]] > 0:
					self.counters[agents[i]] -= 1

				self.actions[agents[i]] = best_combination[i]
			
			#print("Future actions: {}".format(self.actions))

				
		else:
			# choose best task for each agent
			for agent in self.tasks:
				best_task = ""
				best_utility = -math.inf
				current_task_total_utility = 0
				best_task_total_utility = 0

				# select task with higher utility
				for task in self.tasks[agent]:

					# check task with higher utility
					if self.tasks[agent][task][0] > best_utility:
						best_task = task
						best_utility = self.tasks[agent][task][0]

				#print("Best Task is {}".format(best_task))

				if self.decision == "flexible" and best_task in self.hasNegative:
					tasks = self.tasks['A']
					tasks_lst = []
					best_expected_u = -math.inf
					best_first_task = ""
					best_second_task = ""
					p_best_first_task = 0
					p_best_second_task = 0
					
					#get all tasks
					for task in tasks:
						tasks_lst += [task]

					#generate all possible combinations between tasks
					combinations = list(itertools.combinations(tasks_lst,2))
					
					for combination in combinations:
						first_task = combination[0]
						second_task = combination[1]

						#get minimum utilities for each task
						if len(tasks[first_task][2]) > 0:
							min_u_first_task = min(tasks[first_task][2])
						else:
							min_u_first_task = tasks[first_task][0]

						if len(tasks[second_task][2]) > 0:
							min_u_second_task = min(tasks[second_task][2])
						else:
							min_u_second_task = tasks[second_task][0]

						if min_u_second_task != min_u_first_task:
							# get percentage of efforts for each task					
							p_first_task = min_u_second_task / (min_u_second_task - min_u_first_task)
							p_second_task = 1 - p_first_task

							if p_first_task >= 0 and p_first_task <= 1:
								# get expected utility given effort for each task
								expected_u = p_first_task * tasks[first_task][0] + p_second_task * tasks[second_task][0]

								if expected_u > best_expected_u:
									best_expected_u = expected_u
									best_first_task = first_task
									best_second_task = second_task
									p_best_first_task = p_first_task
									p_best_second_task = p_second_task

					self.split_action = {best_first_task: p_best_first_task, best_second_task: p_best_second_task}
					self.split_string += "{{{}={:.2f},{}={:.2f}}}\n".format(best_first_task, p_best_first_task, best_second_task, p_best_second_task)
				else:
					# if there is a task with higher utility, check if is worth changing task
					if best_task != self.actions[agent] and self.actions[agent] != "":

						# calculates total gain for current chosen task
						executed_steps = self.nr_steps - self.counters[agent]
						if executed_steps < 0:
							executed_steps = 0
						current_task_utility = self.tasks[agent][self.actions[agent]][0]
						current_task_total_utility = current_task_utility * executed_steps

						# calculates total gain for task with higher utility
						executed_steps = self.nr_steps - self.restart
						if executed_steps < 0:
							executed_steps = 0
						best_task_total_utility = best_utility * executed_steps

					# change task if another task is worth it or if it is the first choice
					if best_task_total_utility > current_task_total_utility or self.actions[agent] == "" or best_task_total_utility == current_task_total_utility and best_task < self.actions[agent]:
						self.actions[agent] = best_task
						self.counters[agent] = self.restart

					if self.counters[agent] > 0:
						self.counters[agent] -= 1

		self.nr_steps -= 1
		#print(self.split_action)




	def recharge(self):
		if self.decision == "homogeneous-society" and len(self.utilities) > 0:
			self.decide_act()

		state = ''

		if self.decision == "rationale" or self.decision == "flexible":
			tasks = self.tasks['A']
			for task in tasks:
				if tasks[task][1] == 1:
					state += "{}={:.2f},".format(task, tasks[task][0])
				else:
					state += "{}=NA,".format(task)
			state = state[:-1]
			final = "state=" + '{' + state + '}' + ' ' + "gain={:.2f}".format(self.gain)
		else:
			for agent in self.tasks:
				agent_state = ''
				for task in self.tasks[agent]:
					if self.tasks[agent][task][1] == 1:
						agent_state += "{}={:.2f},".format(task, self.tasks[agent][task][0])
					else:
						agent_state += "{}=NA,".format(task)
				agent_state = agent_state[:-1]
				state += agent + '={' + agent_state + '},'

			state = state[:-1]
			final = "state=" + '{' + state + '}' + ' ' + "gain={:.2f}".format(self.gain)

		if self.decision == "flexible":
			final = self.split_string + final

		return final





#####################
### B: MAIN UTILS ###
#####################

line = sys.stdin.readline()
agent = Agent(line.split(' '))
for line in sys.stdin:
	if line.startswith("end"): break
	elif line.startswith("TIK"): agent.decide_act()
	else: agent.perceive(line)
sys.stdout.write(agent.recharge()+'\n');

