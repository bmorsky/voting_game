using Distributions, Plots, Random, Statistics

# Outputs
avg_output = fill(NaN,(10,10))#10*ones(10,10) #Array{Float32,2}(10,10,10)
var_output = fill(NaN,(10,10))#10*ones(10,10) #Array{Float32,2}(10,10,10)
consensus_makers_output = fill(NaN,(10,10))
strategists_output = fill(NaN,(10,10))

# Parameters
M = 3 # memory length
N = 120 # number of players
num_games = 20 # number of games to average over
num_turns = 300 # number of turns
S = 2 # number of strategy tables per player
p = 5/N # probability δ/N of joining two nodes with mean degree δ
μ = 0.2
ϕ = 1
function χ(i)
    if i == 0
        return 5
    elseif i==1
        return 1
    else
        return 1
    end
end
bias = 0.5

# Variable
rng = MersenneTwister()
d = Binomial(1,bias)

for num_consensus_makers = 10:10:N-20
    for num_strategists = 10:10:N-10-num_consensus_makers
        num_zealots = N-num_consensus_makers-num_strategists
        avg_vote = zeros(num_games)
        avg_consensus_makers = zeros(num_games)
        avg_strategists = zeros(num_games)
        for game=1:num_games
            # Initialize game
            history = rand(rng,1:2^M,N) # initial history
            strategy_table_payoffs = zeros(Float32,N,S) # payoffs for strategy tables
            payoffs = zeros(N)
            party = rand(d,N)
            strategy_tables = rand(rng,0:1,S*N,2^M) # S strategy tables for the N players
            vote = rand(d,N)
            majority_vote = Array{Float32,1}(undef,num_turns)
            strategy = vcat(zeros(Int,num_consensus_makers),ones(Int,num_strategists),2*ones(Int,num_zealots))
            # Zealot's votes
            for j=1:N
                if strategy[j] == 2
                    vote[j] = party[j]
                end
            end
            # Generate random graph
            adjacency_matrix = [[] for i = 1:N]
            for i=1:N-1
                for j=i+1:N
                    if party[i]==party[j]
                        imitate=p*χ(strategy[i])χ(strategy[j])
                    else
                        imitate=ϕ*p*χ(strategy[i])χ(strategy[j])
                    end
                    if imitate >= rand(rng)
                        push!(adjacency_matrix[i],j)
                        push!(adjacency_matrix[j],i)
                    end
                end
            end
            # Run a single realization
            for turn=1:num_turns
                # Strategists' votes
                for j=1:N
                    if strategy[j] == 1
                        best_strat = 2*(j-1) + findmax(strategy_table_payoffs[j,:])[2]
                        vote[j] = strategy_tables[best_strat,history[j]]
                    end
                end
                cur_vote = sum(vote)
                if cur_vote > N/2
                    majority = 1
                else
                    majority = 0
                end
                majority_vote[turn] = maximum([cur_vote,N-cur_vote])/N

                # Stategic voters: determine payoffs
                for j=1:N
                    if strategy[j] == 1
                        cur_party = party[j]
                        for k=1:S
                            if majority == cur_party == strategy_tables[2*j-1,history[j]]
                                strategy_table_payoffs[j,k] += 1
                            elseif majority == strategy_tables[2*j-1,history[j]] != cur_party
                                strategy_table_payoffs[j,k] += 1/2
                            elseif majority != cur_party == strategy_tables[2*j-1,history[j]]
                                strategy_table_payoffs[j,k] -= 1/2
                            else
                                strategy_table_payoffs[j,k] -= 1
                            end
                        end
                    end
                end

                # Determine payoffs
                for j=1:N
                    if sum(vote[adjacency_matrix[j]]) > length(adjacency_matrix[j])/2
                        local_majority = 1
                    else
                        local_majority = 0
                    end
                    cur_party = party[j]
                    cur_vote = vote[j]
                    if local_majority == cur_party == cur_vote
                        payoffs[j] += 1
                    elseif local_majority == cur_vote != cur_party
                        payoffs[j] += 1/2
                    elseif local_majority != cur_party == cur_vote
                        payoffs[j] -= 1/2
                    else
                        payoffs[j] -= 1
                    end
                end

                # Determine local history and consensus makers' votes
                for j=1:N
                    if sum(vote[adjacency_matrix[j]]) > length(adjacency_matrix[j])/2
                        local_majority = 1
                    else
                        local_majority = 0
                    end
                    history[j] = Int(mod(2*history[j],2^M) + majority + 1)
                    if strategy[j] == 0
                        vote[j] = local_majority
                    end
                end

                # Imitate others
                for i=1:N
                    j = rand(rng,1:N)
                    if rand(rng) < μ && adjacency_matrix[j] != []
                        neighbour = rand(adjacency_matrix[j])
                        if party[j]==party[neighbour]
                            imitate=(2+mean(payoffs[neighbour])-mean(payoffs[j]))/4
                        else
                            imitate=ϕ*(2+mean(payoffs[neighbour])-mean(payoffs[j]))/4
                        end
                        if imitate >= rand(rng)
                            payoffs[j] = payoffs[neighbour]
                            strategy[j] = strategy[neighbour]
                            if strategy[j] == 2
                                vote[j] = party[j]
                            else
                                vote[j] = vote[neighbour]
                            end
                            # party[j] = party[neighbour]
                            strategy_tables[j] = strategy_tables[neighbour]
                            strategy_table_payoffs[j] = strategy_table_payoffs[neighbour]
                        end
                    end
                end

            end
            avg_vote[game] = mean(majority_vote)
            avg_consensus_makers[game] = sum(x->x==0,strategy)
            avg_strategists[game] = sum(x->x==1,strategy)
        end
        avg_output[Int(num_strategists/10),Int(num_consensus_makers/10)] = mean(avg_vote)
        var_output[Int(num_strategists/10),Int(num_consensus_makers/10)] = var(avg_vote)
        consensus_makers_output[Int(num_strategists/10),Int(num_consensus_makers/10)] = mean(avg_consensus_makers)
        strategists_output[Int(num_strategists/10),Int(num_consensus_makers/10)] = mean(avg_strategists)
    end
end

pyplot()

heatmap(1:10, 1:10, avg_output, xlabel="Consensus makers", ylabel="Strategists", 
colorbar_title="Votes for majority", thickness_scaling = 1.5, clim=(0.5,1))
savefig("evol_avg_phi$(ϕ)_bias$bias.pdf")

# heatmap(1:Int(10), 1:Int(10), var_output, xlabel="Consensus makers", ylabel="Strategists", 
# colorbar_title="Votes for majority", thickness_scaling = 1.5)
# savefig("evol_var_mem_bias_$M.pdf")

heatmap(1:Int(10), 1:Int(10), consensus_makers_output, xlabel="Consensus makers", ylabel="Strategists", 
colorbar_title="Votes for majority", thickness_scaling = 1.5)
savefig("evol_cons_phi$(ϕ)_bias$bias.pdf")

heatmap(1:Int(10), 1:Int(10), strategists_output, xlabel="Consensus makers", ylabel="Strategists", 
colorbar_title="Votes for majority", thickness_scaling = 1.5)
savefig("evol_strat_phi$(ϕ)_bias$bias.pdf")

heatmap(1:Int(10), 1:Int(10), N .- consensus_makers_output .- strategists_output, xlabel="Consensus makers", ylabel="Strategists", 
colorbar_title="Votes for majority", thickness_scaling = 1.5)
savefig("evol_zeal_phi$(ϕ)_bias$bias.pdf")