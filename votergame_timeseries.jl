using Plots, Random, Statistics, StatsPlots

# Parameters
κ = 100 # payoff differential sensitivity
ℓⁱ = 0.1 # rate of individual learning
ℓˢ = 0.1 # rate of social learning
max_M = 12
N = 50 # total number of agents + bots, X = [51,101,251,501,1001]
num_games = 50 # number of games to average over
num_turns = 500 # number of turns
S = 2 # number of strategy tables per individual
threshold = 65/100

# Variables
vote = Array{Int,1}(undef,num_turns)
history = Array{Int,1}(undef,max_M) # history of the winning strategy for the last M turns

# Output
vote = Array{Int,1}(undef,num_turns)
vote_volatility = zeros(max_M*num_games*5,2)

B = [0,1,5,10,20] #[0,5,10,15,20] # [0,5,10,15,20] #
rng = MersenneTwister()

global count1 = 1
global count2 = 1
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
            vote_volatility[count1,:] = [var(vote)/N, (2^M)/N]
            global count1 += 1
            avg_vote_volatility[count2,1] += var(vote)/(num_games*N)
        end
        avg_vote_volatility[count2,2] = (2^M)/N
        global count2 += 1
    end
end

x1 = vote_volatility[1:num_games*max_M,2]
x2 = vote_volatility[num_games*max_M+1:2*num_games*max_M,2]
x3 = vote_volatility[2*num_games*max_M+1:3*num_games*max_M,2]
x4 = vote_volatility[3*num_games*max_M+1:4*num_games*max_M,2]
x5 = vote_volatility[4*num_games*max_M+1:5*num_games*max_M,2]

y1 = vote_volatility[1:num_games*max_M,1]
y2 = vote_volatility[num_games*max_M+1:2*num_games*max_M,1]
y3 = vote_volatility[2*num_games*max_M+1:3*num_games*max_M,1]
y4 = vote_volatility[3*num_games*max_M+1:4*num_games*max_M,1]
y5 = vote_volatility[4*num_games*max_M+1:5*num_games*max_M,1]

scatter([x1 x2 x3 x4 x5], [y1 y2 y3 y4 y5], markercolor=[1 2 3 4 5],
xlims=(0.01,100), ylims=(0.01,100), xscale=:log10, yscale=:log10,
label=["no bots" "1 bot" "5 bots" "10 bots" "20 bots"],
xlabel = "\\alpha", ylabel="\\sigma ²/N", legend=:bottomright,
thickness_scaling = 1.5, alpha=0.5)
savefig("voter_thresh_65_N50.pdf")

violin(reshape(y1,(num_games,max_M)), yscale=:log10)
savefig("y1.pdf")
violin(reshape(y5,(num_games,max_M)), yscale=:log10)
savefig("y2.pdf")

#violin([y2[1:num_games] y2[num_games+1:2*num_games] y2[2*num_games+1:3*num_games] y2[3*num_games+1:4*num_games] y2[4*num_games+1:5*num_games]],yscale=:identity)

x1 = avg_vote_volatility[1:max_M,2]
x2 = avg_vote_volatility[max_M+1:2*max_M,2]
x3 = avg_vote_volatility[2*max_M+1:3*max_M,2]
x4 = avg_vote_volatility[3*max_M+1:4*max_M,2]
x5 = avg_vote_volatility[4*max_M+1:5*max_M,2]

y1 = avg_vote_volatility[1:max_M,1]
y2 = avg_vote_volatility[max_M+1:2*max_M,1]
y3 = avg_vote_volatility[2*max_M+1:3*max_M,1]
y4 = avg_vote_volatility[3*max_M+1:4*max_M,1]
y5 = avg_vote_volatility[4*max_M+1:5*max_M,1]

scatter([x1 x2 x3 x4 x5], [y1 y2 y3 y4 y5], markercolor=[1 2 3 4 5],
xlims=(0.01,100), ylims=(0.01,100), xscale=:log10, yscale=:log10,
label=["no bots" "1 bot" "5 bots" "10 bots" "20 bots"],
xlabel = "\\alpha", ylabel="\\sigma ²/N", legend=:bottomright,
thickness_scaling = 1.5)
savefig("avg_voter_thresh_65_N50.pdf")