using Plots, Random, Statistics, StatsPlots

# Outputs
max_M = 12
avg_vote_volatility = zeros(max_M*5,3) # [, , num_bots]

# Parameters
κ = 100 # payoff differential sensitivity
ℓⁱ = 0.1 # rate of individual learning
ℓˢ = 0.1 # rate of social learning
N = 100 # total number of agents + bots, X = [51,101,251,501,1001]
num_games = 100 # number of games to average over
num_turns = 500 # number of turns
S = 2 # number of strategy tables per individual
threshold = 60/100

# Variables
vote = Array{Int,1}(undef,num_turns)

B = [0,1,5,10,20] #[0,5,10,15,20] # [0,5,10,15,20] #
rng = MersenneTwister()

global count = 1
for b = 1:5
    num_bots = B[b] # number of agents
    team = [zeros(Int64,Int(N/2)-num_bots); ones(Int64,Int(N/2))]

    num_agents = N - num_bots # number of agents
    action = Array{Int,1}(undef,N) # actions taken: buy=1, sell=0
    for M = 1:max_M
        for game=1:num_games
            # Initialize game
            history = rand(rng,1:2^M)
            strategy_tables = rand(rng,0:1,S*num_agents,2^M) # S strategy tables for the N players
            virtual_points = zeros(Int64,num_agents,S) # virtual points for all players' strategy tables

            # Run simulation
            for turn=1:num_turns
                # Actions taken
                for i=1:num_agents
                    best_strat = 2*(i-1) + findmax(virtual_points[i,:])[2]
                    action[i] = strategy_tables[best_strat,history]
                end
                cur_vote = sum(action)
                if cur_vote >= (N-1)/2
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

                # Individual learning
                for i=1:num_agents
                    if ℓⁱ > rand(rng)
                        new_strat = rand(rng,0:1)
                        strategy_tables[i+new_strat,:] = rand(rng,0:1,2^M)
                        virtual_points[i,new_strat+1] = 0
                    end
                end

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
            avg_vote_volatility[count,1] += var(vote)/(num_games*N)
        end
        avg_vote_volatility[count,2] = (2^M)/N
        avg_vote_volatility[count,3] = b
        global count += 1
    end
end

avg_vote_volatility[:,3] = Int.(avg_vote_volatility[:,3])

x = 2 .^ collect(1:M)/N

y1 = avg_vote_volatility[1:12,1]
y2 = avg_vote_volatility[13:24,1]
y3 = avg_vote_volatility[25:36,1]
y4 = avg_vote_volatility[37:48,1]
y5 = avg_vote_volatility[49:60,1]

z1 = Int.(avg_vote_volatility[1:12,3])
z2 = Int.(avg_vote_volatility[13:24,3])
z3 = Int.(avg_vote_volatility[25:36,3])
z4 = Int.(avg_vote_volatility[37:48,3])
z5 = Int.(avg_vote_volatility[49:60,3])

scatter([x1 x2 x3 x4 x5], [y1 y2 y3 y4 y5], markercolor=[1 2 3 4 5],
xlims=(0.01,100), ylims=(0.01,10), xscale=:log10, yscale=:log10,
label=["no bots" "1 bot" "5 bots" "10 bots" "20 bots"],
xlabel = "\\alpha", ylabel="\\sigma ²/N", legend=:bottomright,
thickness_scaling = 1.5)
savefig("voter_thresh_55.pdf")
