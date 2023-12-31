using Plots
using Random
using Statistics

# Outputs
max_M = 12
avg_outcome = zeros(max_M*4,3)

# Parameters
κ = 100 # payoff differential sensitivity
ℓⁱ = 0.1 # rate of individual learning
ℓˢ = 0.1 # rate of social learning
N = 120 # number of players
num_games = 100 # number of games to average over
num_turns = 500 # number of turns
S = 2 # number of strategy tables per player
threshold = 80/120

# Variables
outcome = Array{Int,1}(undef,num_turns)

initial_strats = [40 40 40; 20 20 80; 20 80 20; 80 20 20]
rng = MersenneTwister()

global count = 1
for i = 1:4
    for M = 1:max_M
        for game=1:num_games
            # Initialize game
            num_consensus_makers = initial_strats[i,1] # number of consensus makers
            num_strategic_voters = initial_strats[i,2] # number of strategic voters
            num_zealots = initial_strats[i,3] # number of zealots
            consensus_votes = Int(num_consensus_makers/2)
            strategic_voters_votes = Array{Int,1}(undef,num_strategic_voters) # votes taken: vote=1, vote=0
            history = rand(rng,1:2^M)
            strategy_tables = rand(rng,0:1,S*N,2^M) # S strategy tables for the N players
            virtual_points = zeros(Int64,N,S) # virtual points for all players' strategy tables
            team = rand(rng,0:1,N)
                # Votes
                for j=1:num_strategic_voters
                    best_strat = 2*(j-1) + findmax(virtual_points[j,:])[2]
                    strategic_voters_votes[j] = strategy_tables[best_strat,history]
                end
                cur_vote = Int(num_zealots/2)+consensus_votes*num_consensus_makers + sum(strategic_voters_votes)
                if cur_vote >= N/2
                    majority = 1
                else
                    majority = 0
                end
                if cur_vote >= threshold*N
                    super_majority = 1
                    vote[t] = 1
                elseif cur_vote <= (1-threshold)*N
                    super_majority = 0
                    vote[t] = 1
                else
                    super_majority = -1
                    vote[t] = 0
                end

                consensus_votes = majority

                # Stategic voters: determine virtual payoffs
                if super_majority != -1
                    for j=1:num_strategic_voters
                        cur_team = team[j]
                        virtual_points[j,1] += 1+(-1)^(super_majority+team[j])/2+(-1)^(super_majority+strategy_tables[2*j-1,history])/2
                        virtual_points[j,2] += 1+(-1)^(super_majority+team[j])/2+(-1)^(super_majority+strategy_tables[2*j,history])/2
                    end
                else
                    virtual_points .-= 1
                end
                history = Int(mod(2*history,2^M) + majority + 1)

                # Individual learning: zealots
                for j = 1:num_zealots
                    if ℓⁱ > rand(rng)
                        r = rand(rng,1:2)
                        if r == 1
                            strategy_tables = vcat(strategy_tables,rand(rng,0:1,2,2^M))
                            virtual_points = vcat(virtual_points,[0,0])
                            team = vcat(team,zealots[j])
                            k -= 1
                            num_zealots -= 1
                        else
                            strategy_tables = strategy_tables[1:end .!= k, :]
                            virtual_points = virtual_points[1:end .!= k, :]
                            team  = team[1:end .!= k, :]
                            k -= 1
                            num_zealots -= 1
                            num_consensus_makers += 1
                        end
                    end
                    k += 1
                end
                # Individual learning: strategic voters
                k = 1
                while k <= num_strategic_voters
                    if ℓⁱ > rand(rng)
                        r = rand(rng,1:3)
                        if r == 1
                            strategy_tables = strategy_tables
                            virtual_points = virtual_points[1:end .!= k, :]
                            if team[k] == 1
                                num_zealots += 2
                            end
                            team  = team[1:end .!= k, :]
                            k -= 1
                            num_strategic_voters -= 1
                        elseif r == 2
                            new_strat = rand(rng,0:1)
                            strategy_tables[k+new_strat,:] = rand(rng,0:1,2^M)
                            virtual_points[k,new_strat+1] = 0
                        else
                            strategy_tables = strategy_tables[1:end .!= k, :]
                            virtual_points = virtual_points[1:end .!= k, :]
                            team  = team[1:end .!= k, :]
                            k -= 1
                            num_consensus_makers += 1
                            num_strategic_voters -= 1
                        end
                    end
                    k += 1
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
            avg_attendance_volatility[count,1] += mean(vote)/(num_games)
        end
        avg_attendance_volatility[count,2] = (2^M)/N
        avg_attendance_volatility[count,3] = i
        global count += 1
    end
end

avg_attendance_volatility[:,3] = Int.(avg_attendance_volatility[:,3])

x1 = avg_attendance_volatility[1:12,2]
x2 = avg_attendance_volatility[13:24,2]
x3 = avg_attendance_volatility[25:36,2]
x4 = avg_attendance_volatility[37:48,2]

y1 = avg_attendance_volatility[1:12,1]
y2 = avg_attendance_volatility[13:24,1]
y3 = avg_attendance_volatility[25:36,1]
y4 = avg_attendance_volatility[37:48,1]

z1 = Int.(avg_attendance_volatility[1:12,3])
z2 = Int.(avg_attendance_volatility[13:24,3])
z3 = Int.(avg_attendance_volatility[25:36,3])
z4 = Int.(avg_attendance_volatility[37:48,3])

scatter([x1 x2 x3 x4], [y1 y2 y3 y4], markercolor=[z1 z2 z3 z4],
xlims=(0.01,1000), ylims=(0,1.1), xscale=:log10,
label=["1/3, 1/3, 1/3" "1/6, 1/6, 2/3" "1/6, 2/3, 1/6" "2/3, 1/6, 1/6"],
xlabel = "\\alpha", ylabel="Vote", legend=:topright,
thickness_scaling = 1.5)
savefig("vote_imitate_theta66.pdf")
