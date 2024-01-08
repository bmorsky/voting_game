using Distributions, Plots, Random, Statistics

# Outputs
avg_output = fill(NaN,(10,10))#10*ones(10,10) #Array{Float32,2}(10,10,10)
var_output = fill(NaN,(10,10))#10*ones(10,10) #Array{Float32,2}(10,10,10)

# Parameters
M = 3 # memory length
N = 120 # number of players
num_games = 20 # number of games to average over
num_turns = 300 # number of turns
S = 2 # number of strategy tables per player
p = 10/N # probability δ/N of joining two nodes with mean degree δ
ϕ = 0.5

# Variables
rng = MersenneTwister()
d = Binomial(1,0.5)

for num_consensus_makers = 10:10:N-20
    for num_strategists = 10:10:N-10-num_consensus_makers
        num_zealots = N-num_consensus_makers-num_strategists
        avg_vote = zeros(num_games)
        for game=1:num_games
            # Initialize game
            history = rand(rng,1:2^M,N-num_zealots) # initial history
            payoffs = zeros(Float32,num_strategists,S) # payoffs for strategy tables
            strategists_parties = rand(d,num_strategists)
            strategy_tables = rand(rng,0:1,S*num_strategists,2^M) # S strategy tables for the N players
            vote = rand(d,N)
            majority_vote = Array{Float32,1}(undef,num_turns)
            party = vcat(strategists_parties,vote[num_strategists+1:N])
            # Generate random graph
            adjacency_matrix = [[] for i = 1:N]
            for i=1:N-1
                for j=i+1:N
                    if party[i]==party[j]
                        imitate=p
                    else
                        imitate=ϕ*p
                    end
                    if imitate >= rand(rng)
                        push!(adjacency_matrix[i],j)
                        push!(adjacency_matrix[j],i)
                    end
                end
            end
            # Run a single realization
            for turn=1:num_turns
                # Votes
                for j=1:num_strategists
                    best_strat = 2*(j-1) + findmax(payoffs[j,:])[2]
                    vote[j] = strategy_tables[best_strat,history[j]]
                end
                cur_vote = sum(vote)
                if cur_vote > N/2
                    majority = 1
                else
                    majority = 0
                end
                majority_vote[turn] = maximum([cur_vote,N-cur_vote])/N

                # Stategic voters: determine payoffs
                for j=1:num_strategists
                    cur_party = strategists_parties[j]
                    for k=1:S
                        if majority == cur_party == strategy_tables[2*j-1,history[j]]
                            payoffs[j,k] += 1
                        elseif majority == strategy_tables[2*j-1,history[j]] != cur_party
                            payoffs[j,k] += 1/2
                        elseif majority != cur_party == strategy_tables[2*j-1,history[j]]
                            payoffs[j,k] -= 1/2
                        else
                            payoffs[j,k] -= 1
                        end
                    end
                end

                # Determine local history and consensus makers' votes
                for j=1:N-num_zealots
                    if sum(vote[adjacency_matrix[j]]) > length(adjacency_matrix[j])/2
                        local_majority = 1
                    else
                        local_majority = 0
                    end
                    history[j] = Int(mod(2*history[j],2^M) + majority + 1)
                    if j > num_strategists
                        vote[j] = local_majority
                    end
                end
            end
            avg_vote[game] = mean(majority_vote)
        end
        avg_output[Int(num_strategists/10),Int(num_consensus_makers/10)] = mean(avg_vote)
        var_output[Int(num_strategists/10),Int(num_consensus_makers/10)] = var(avg_vote)
    end
end

pyplot()

heatmap(1:10, 1:10, avg_output, xlabel="Consensus makers", ylabel="Strategists", 
colorbar_title="Votes for majority", thickness_scaling = 1.5, clim=(0.5,1))
savefig("avg.pdf")

# heatmap(1:Int(10), 1:Int(10), var_output, xlabel="Consensus makers", ylabel="Strategists", 
# colorbar_title="Votes for majority", thickness_scaling = 1.5)
# savefig("dynamic_var_mem_bias_$M.pdf")
