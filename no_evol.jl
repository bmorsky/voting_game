using Distributions, Plots, Random, Statistics

# Outputs
avg_output = Array{Float32,2}(undef,10,10)
var_output = Array{Float32,2}(undef,10,10)

# Parameters
M = 4 # memory length
N = 121 # number of players
num_games = 100 # number of games to average over
num_turns = 500 # number of turns
S = 2 # number of strategy tables per player

# Variables
rng = MersenneTwister()
d = Binomial(1,0.8)

for num_consensus_makers = 10:10:N-20
    for num_strategists = 10:10:N-20-num_consensus_makers
        num_zealots = N-num_consensus_makers-num_strategists
        avg_vote = zeros(num_games)
        for game=1:num_games
            # Initialize game
            consensus_votes = round(sum(rand(d,num_consensus_makers))/num_consensus_makers) # initial consensus votes
            history = rand(rng,1:2^M) # initial history
            payoffs = zeros(Float32,num_strategists,S) # payoffs for strategy tables
            strategists_parties = rand(d,num_strategists)
            strategists_votes = Array{Int,1}(undef,num_strategists) # votes taken: vote=1, vote=0
            strategy_tables = rand(rng,0:1,S*num_strategists,2^M) # S strategy tables for the N players
            vote = Array{Float32,1}(undef,num_turns)
            zealots_votes = sum(rand(d,num_zealots)) #sum(rand(0:1,num_zealots))
            for turn=1:num_turns
                # Votes
                for j=1:num_strategists
                    best_strat = 2*(j-1) + findmax(payoffs[j,:])[2]
                    strategists_votes[j] = strategy_tables[best_strat,history]
                end
                cur_vote = consensus_votes*num_consensus_makers + sum(strategists_votes) + zealots_votes
                if cur_vote > N/2
                    majority = 1
                else
                    majority = 0
                end
                consensus_votes = majority
                vote[turn] = maximum([cur_vote,N-cur_vote])/N

                # Stategic voters: determine payoffs
                for j=1:num_strategists
                    cur_party = strategists_parties[j]
                    for k=1:S
                        if majority == cur_party == strategy_tables[2*j-1,history]
                            payoffs[j,k] += 1
                        elseif majority == strategy_tables[2*j-1,history] != cur_party
                            payoffs[j,k] += 1/2
                        elseif majority != cur_party == strategy_tables[2*j-1,history]
                            payoffs[j,k] -= 1/2
                        else
                            payoffs[j,k] -= 1
                        end
                    end
                end
                history = Int(mod(2*history,2^M) + majority + 1)
            end
            avg_vote[game] = mean(vote)
        end
        avg_output[Int(num_strategists/10),Int(num_consensus_makers/10)] = mean(avg_vote)
        var_output[Int(num_strategists/10),Int(num_consensus_makers/10)] = var(avg_vote)
    end
end

pyplot()

heatmap(1:10, 1:10, avg_output, xlabel="Consensus makers", ylabel="Strategists", 
colorbar_title="Votes for majority", thickness_scaling = 1.5)
savefig("static_avg_mem_$M.pdf")

heatmap(1:Int(10), 1:Int(10), var_output, xlabel="Consensus makers", ylabel="Strategists", 
colorbar_title="Votes for majority", thickness_scaling = 1.5)
savefig("static_var_mem_$M.pdf")
