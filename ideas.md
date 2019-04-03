Ideas
=====



Bayesian modelling of the state of the enemy
	E.g how many units of a particular type the enemy has
	also how this state would evolve over time by using different strategies

	State possibilities
		1.
			Divide map into a number of key points (e.g base locations and some other points on the map, maybe about 1 screen width apart)..
			Keep track of (probabilistic) unit count at each position
			Keep track of number of buildings of each type?
				Or just has at least one/none?
			Keep track of upgrades
			Base locations

			Possibly: keep track of total number of units + normalized distribution of positions
				Might be easier to keep track of and update

			Log decomposition of unit groups (e.g. allow N units to be ordered somewhere, where N \in {1,2,4,8,16,32,64, etc.})



	Actions
		Don't use actions like 'produce units', instead use actions that determine the policy (e.g. focus on economy, focus on countering the enemy)
		Possibly with occational 'scout'
Analyze replays
	1. Bayesian causality
		Which states lead to other states
	2. 

ML states
	ML to map game states (possibly probabilistic) to an N dimensional space, where similarities between states can be done using a distance function (or maybe a dot product)
	ML to predict what states lead to what other states.
	Interpolate between states?
	States are similar if they predict/lead to similar outcomes?

Bayesian
	Extract large scale unit movements
		If unit moved more than say 1 screen distance, consider it important.
			First try to predict which group of units will move
				Score every unit?
				Output GMM that queries the units?
			Then given a unit group, where should it move to

MCTS
	Make a NN learn a representation for what reasonable actions there are in a given state, let the possible actions in a given MCTS state be samples from this NN instead of all possible actions (might be overwhelmingly many).

	Separate economic and military MCTS
		run a MCTS for military first and then assume that policy is static when evaluating the economic one
		finally re-run the military MCTS again.

		Reduces branching factor

	Sample actions from
		NN that predicts which region units should come from (branch ≈25)
		NN that predicts which type of units from that region should move (branch 150), given unit counts in the region
		NN that predicts how many units of them should move (branch log2 200. Alternatives are 1, 2, 4, 8, 16, etc. given the total count as 1, 2, 4, ... one hot encoded)
		NN that predicts movement type (attack, move)

	Learn NN that generates priors for states based on the simulation results so far ("every simulation that ends up attacking base B seems to turn out badly, maybe stop exploring attacks on base B?")
	Use GAN to sample actions using a latent space
		Generator: (latent space, region x unit counts, mining speed etc.) -> (one hot from region, one hot to region, soft unit mask [0...1]^N, one hot log2 unit count, one hot attack/move)
		Discriminator -> (input space, (one hot from region, one hot to region, soft unit mask [0...1]^N, one hot log2 unit count, one hot attack/move))

		Can even be used for policies like how many units to produce.
			Extract from replays state at T and T + dt and check how many units were produced of each type in between, set that as a policy goal.
			Use GAN to sample this space and MCTS to evaluate the different actions
	

	Action space
		Move to
			Unit subsets
				All army units
				All ground units
				All flying units
				3
			Possible regions are limited:
				bases: 3
				enemy bases: 3
				outside enemy bases: 3
				enemy mineral lines: 3
				enemy position
				12

		Action groups "Transpositions and Move Groups in Monte Carlo Tree Search"
			E.g split up the action "attack base 3" to two actions attack -> base 3.
			This means the bot can discover that 'attack' is good before it has a good idea about which base to attack
			Can this be generalized? Similarity between actions.
			E.g. all actions with base 1 as target are similar, all actions that attack are similar.
	
	Optimization note
		Since most groups are probably not going to change every step, one way to reduce memory usage and reduce amount of memory that is copied is to make groups
		represent a state as a function of time (e.g. position is as a function of time, etc.)
		When creating a new simulation state, a pointer to the old group is kept. If the group needs to be changed in any way a copy is made however.

NN for refining the destinations
	Input
		Small region around the destination ± some noise
	Output
		Actual destination

State examples
	1. Beginning of game, 12 workers and one command center
	2. Early game, a single barracks, no military units
	3. Mid game, a mid sized army and some production buildings
	4. In the middle of a fight
	5. Late game with multiple bases. A small army because it was recently lost, but high production capacity
	6. Late game with multiple bases. A large army but crippled economy

Microing
	State:
		Sorted list of enemy units by distance with all unit stats (up to say 8 units, then an additional counter saying that there are say N more units nearby)
			Positions given relative to current unit.
		Sorted list of ally units by distance with all unit stats.
			Positions given relative to current unit.
		Low res walkability map?
		Some kind of phase perhaps? To coordinate different units.

		Evaluate a few positions (say 8) around the agent and train a part of the network to give a score for how reasonable they are to move towards
	Actions:
		Move(Point)
		Attackmove(Point)
		Attack(unit type?)


Countering bot
	Predict what the enemy has.
		Build counter

Citing
	Hjax: https://www.youtube.com/watch?v=-4zdI8p943Q&feature=youtu.be&t=3730
		Comparison of making random moves in chess vs starcraft

Chatting
	To be or not to be, that is the question
		In your case however, the answer is simple: it is 'not to be'

	Comment on microing skills after combats

	Glory to Arztotzka!
	Ha! I can check-mate you in 3... no wait, this isn't chess! What is this game??
		Maybe print random chess moves during the match (Knight to G-4)
		


Reinforcement Learning Build Order
	State:
		Unit counts
			Current units
			Units In Progress
			Units almost ready / Time to next unit finished

		Goal
			Normalized unit counts that we want to make
	Rewards
		+1 when creating a desired unit
		+ e^(-old_count/(desired_proportion * (1 + total_unit_count)))
		+ value corresponding to how well the current unit distribution corresponds to the target distribution
		+ 1 when creating a desired upgrade(/building?)
		
		Decay over time, not over number of actions


RNN
	Output army composition to RL build order