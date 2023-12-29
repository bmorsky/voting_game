using Plots, Random, Statistics

# Parameters
κ = 100 # payoff differential sensitivity
ℓⁱ = 0.1 # rate of individual learning
ℓˢ = 0.1 # rate of social learning
M = 6 # memory length
N = 50 # number of agents
num_bots = 0
num_agents = N - num_bots # number of agents
num_turns = 5000 # number of turns
S = 2 # number of strategy tables per individual
threshold = 55/100

rng = MersenneTwister()

# Variables
action = Array{Int,1}(undef,N) # actions taken: buy=1, sell=0
history = Array{Int,1}(undef,M) # history of the winning strategy for the last M turns

# Outputs
vote = Array{Int,1}(undef,num_turns)
phasespace = zeros(num_turns,2)

# Initialize game
history = rand(rng,1:2^M)
strategy_tables = rand(rng,0:1,S*num_agents,2^M) # S strategy tables for the N players
virtual_points = zeros(Int64,num_agents,S) # virtual points for all players' strategy tables
team = [zeros(Int64,Int(N/2)-num_bots); ones(Int64,Int(N/2))]

# Run simulation
for turn=1:num_turns
    # Actions taken
    for i=1:num_agents
        best_strat = 2*(i-1) + findmax(virtual_points[i,:])[2]
        action[i] = strategy_tables[best_strat,history]
    end
    cur_vote = sum(action)
    if cur_vote >= N/2
        majority = 1
    else
        majority = 0
    end
    if cur_vote >= threshold*N
        super_majority = 1
    elseif cur_vote <= (1-threshold)*N
        super_majority = 0
    else
        super_majority = -1
    end
    vote[turn] = sum(action)-(N-sum(action))
    phasespace[turn,:] = [sum(2*action[1:Int(N/2)])/N,2*sum(action[Int(N/2):end])/N]

    # Determine virtual payoffs and win rate
    if super_majority != -1
        for i=1:num_agents
            cur_team = team[i]
            virtual_points[i,1] += 1+(-1)^(super_majority+team[i])/2+(-1)^(super_majority+strategy_tables[2*i-1,history])/2
            virtual_points[i,2] += 1+(-1)^(super_majority+team[i])/2+(-1)^(super_majority+strategy_tables[2*i,history])/2
        end
    else
        virtual_points .-= 1
    end
    history = Int(mod(2*history,2^M) + majority + 1)

    # # Individual learning
    # for i=1:num_agents
    #     if ℓⁱ > rand(rng)
    #         new_strat = rand(rng,0:1)
    #         strategy_tables[i+new_strat,:] = rand(rng,0:1,2^M)
    #         virtual_points[i,new_strat+1] = 0
    #     end
    # end

    # # Social learning
    # update_strategy_tables = strategy_tables
    # update_virtual_points = virtual_points
    # for i=1:num_agents
    #     if ℓˢ >= rand(rng)
    #         # Find worst strategy and its points of focal player
    #         worst_points,worst_strat = findmin(virtual_points[i,:])
    #         # Select random other player and find its best strat and points
    #         player = rand(filter(x -> x ∉ [i], 1:num_agents))
    #         best_points,best_strat = findmax(virtual_points[player,:])
    #         if 1/(1+exp(κ*(worst_points-best_points))) > rand(rng)
    #             update_strategy_tables[2*(i-1)+worst_strat,:] = strategy_tables[2*(player-1)+best_strat,:]
    #             update_virtual_points[i,worst_strat] = virtual_points[player,best_strat]
    #         end
    #     end
    # end
    # strategy_tables = update_strategy_tables
    # virtual_points = update_virtual_points
end

plot(vote[1:500],size = (500, 200),xlabel = "Time",ylabel="A",
legend=false,thickness_scaling = 1.5)
# savefig("ts_M6_N1001_S2.pdf")
# scatter(phasespace[:,1],phasespace[:,2])
