using Distributions, Plots, Random, Statistics

# Arrays for simulation outputs
avg_output = fill(NaN,(10,10)) # average of the average majority vote
var_output = fill(NaN,(10,10)) # variance of the average majority vote
consensus_makers_output = fill(NaN,(10,10)) # propotion of consensus makers
strategists_output = fill(NaN,(10,10)) # proportion of strategists

# Parameters
M = 3 # memory length
N = 120 # number of players
num_games = 20 # number of games to average over
num_turns = 300 # number of turns
S = 2 # number of strategy tables per player
p = 5/N # base probability δ/N of joining two nodes with mean degree δ
μ = 0.2 # rate of immitation of others' strategies
μᵢ = 0.01 # individual learning
ϕ = 1 # weight of imitating the strategy of a player of the opposing party

# Functions
function χ(i) # imitation weight for each strategy
    if i == 0 # if player is a consensus-maker
        return 1
    elseif i==1 # if player is a strategist
        return 1
    else # if player is a zealot
        return 1
    end
end

# Random numbers
rng = MersenneTwister() # pseudorandom number generator
bias = 0.5 # probability that a player is of party 1
d = Binomial(1,bias) # binomial distribution

# Run simulations for different initial conditions of consensus-makers, strategists, and zealots
for num_consensus_makers = 10:10:N-20
    for num_strategists = 10:10:N-10-num_consensus_makers
        num_zealots = N-num_consensus_makers-num_strategists
        avg_vote = zeros(num_games) # vector of average votes for each realization
        avg_consensus_makers = zeros(num_games) # vector of average proportion of consensus-makers for each realization
        avg_strategists = zeros(num_games) # vector of average proportion of strategiest for each realization
        for game=1:num_games # run realizations of the game
            # Initialize game
            history = rand(rng,1:2^M,N) # initial history of votes
            strategy_table_payoffs = zeros(Float32,N,S) # payoffs for strategy tables, initialized to zero
            payoffs = zeros(N) # vector of payoffs for each player
            party = rand(d,N) # vector of party affiliation for each player
            strategy_tables = rand(rng,0:1,S*N,2^M) # S strategy tables for the N players (note that only strategists will use these)
            vote = rand(d,N) # vector of the votes each player makes
            majority_vote = Array{Float32,1}(undef,num_turns) # vector of majority votes each turn
            # strategy is the vector of the initial strategies for each player: 0=consensus-makers, 1=strategists, 2=zealots
            strategy = vcat(zeros(Int,num_consensus_makers),ones(Int,num_strategists),2*ones(Int,num_zealots)) 
            # Zealot's votes
            for j=1:N
                if strategy[j] == 2
                    vote[j] = party[j]
                end
            end
            # Generate Erdos-Renyi random graph
            adjacency_matrix = [[] for i = 1:N]
            for i=1:N-1
                for j=i+1:N
                    if party[i]==party[j] # if the parties of the players are the same
                        imitate=p*χ(strategy[i])χ(strategy[j])
                    else # if the parties of the players are different
                        imitate=ϕ*p*χ(strategy[i])χ(strategy[j])
                    end
                    if imitate >= rand(rng) # join the two players if this condition is true
                        push!(adjacency_matrix[i],j)
                        push!(adjacency_matrix[j],i)
                    end
                end
            end
            # Run a single realization
            for turn=1:num_turns
                # Determine strategists' votes
                for j=1:N
                    if strategy[j] == 1
                        best_strat = 2*(j-1) + findmax(strategy_table_payoffs[j,:])[2]
                        vote[j] = strategy_tables[best_strat,history[j]]
                    end
                end

                # Determine majority
                cur_vote = sum(vote) # sum of the current votes
                if cur_vote > N/2
                    majority = 1
                else
                    majority = 0
                end
                majority_vote[turn] = maximum([cur_vote,N-cur_vote])/N

                # Stategic voters: determine payoffs for their strategy tables
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

                # Determine payoffs for all players
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

                # Determine local history for each player and thus consensus makers' votes
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
                    if rand(rng) < μ && adjacency_matrix[j] != [] # imitate a randomly selected neighbour with probability μ
                        neighbour = rand(adjacency_matrix[j])
                        if party[j]==party[neighbour]
                            imitate=(2+mean(payoffs[neighbour])-mean(payoffs[j]))/4 # probability of imitating a neighbour of the same party
                        else
                            imitate=ϕ*(2+mean(payoffs[neighbour])-mean(payoffs[j]))/4 # probability of imitating a neighbour of a different party
                        end
                        if imitate >= rand(rng)
                            payoffs[j] = payoffs[neighbour] # copy the nieghbour's payoffs
                            strategy[j] = strategy[neighbour] # copy the neighbour's strategy
                            if strategy[j] == 2 # if neighbour is a zealot, adjust vote to be in line with party
                                vote[j] = party[j]
                            else # if neighbour is not a zealot, adjust vote to be that of neighbour
                                vote[j] = vote[neighbour]
                            end
                            # party[j] = party[neighbour] # this is commented out, since we assume that players don't change parties
                            strategy_tables[j] = strategy_tables[neighbour] # copy the neighbour's strategy tables (only has an impact if neighbour is a strategist)
                            strategy_table_payoffs[j] = strategy_table_payoffs[neighbour] # copy the neighbour's strategy tables' payoffs
                        end
                    end
                end

                # Individual learning
                for i=1:N
                    j = rand(rng,1:N)
                    if rand(rng) < μᵢ
                        strategy[j]=rand(0:2)
                    if strategy[j] == 2 # if neighbour is a zealot, adjust vote to be in line with party
                        vote[j] = party[j]
                    end # if neighbour is not a zealot, adjust vote to be that of neighbour
                    # strategy_tables[j] = strategy_tables[neighbour] # copy the neighbour's strategy tables (only has an impact if neighbour is a strategist)
                    # strategy_table_payoffs[j] = strategy_table_payoffs[neighbour] # copy the neighbour's strategy tables' payoffs
                    end
                end

            end
            avg_vote[game] = mean(majority_vote)
            avg_consensus_makers[game] = sum(x->x==0,strategy)/N
            avg_strategists[game] = sum(x->x==1,strategy)/N
        end
        avg_output[Int(num_strategists/10),Int(num_consensus_makers/10)] = mean(avg_vote)
        var_output[Int(num_strategists/10),Int(num_consensus_makers/10)] = var(avg_vote)
        consensus_makers_output[Int(num_strategists/10),Int(num_consensus_makers/10)] = mean(avg_consensus_makers)
        strategists_output[Int(num_strategists/10),Int(num_consensus_makers/10)] = mean(avg_strategists)
    end
end

pyplot()

heatmap(0.1:0.1:1, 0.1:0.1:1, avg_output, xlabel="Consensus makers", ylabel="Strategists", 
colorbar_title="Votes for majority", thickness_scaling = 1.5, clim=(0.5,1))
savefig("evol_avg_phi$(ϕ)_bias$bias.pdf")

# heatmap(1:Int(10), 1:Int(10), var_output, xlabel="Consensus makers", ylabel="Strategists", 
# colorbar_title="Votes for majority", thickness_scaling = 1.5)
# savefig("evol_var_mem_bias_$M.pdf")

heatmap(0.1:0.1:1, 0.1:0.1:1, consensus_makers_output, xlabel="Consensus makers", ylabel="Strategists", 
colorbar_title="Votes for majority", thickness_scaling = 1.5)
savefig("evol_cons_phi$(ϕ)_bias$bias.pdf")

heatmap(0.1:0.1:1, 0.1:0.1:1, strategists_output, xlabel="Consensus makers", ylabel="Strategists", 
colorbar_title="Votes for majority", thickness_scaling = 1.5)
savefig("evol_strat_phi$(ϕ)_bias$bias.pdf")

heatmap(0.1:0.1:1, 0.1:0.1:1, 1 .- consensus_makers_output .- strategists_output, xlabel="Consensus makers", ylabel="Strategists", 
colorbar_title="Votes for majority", thickness_scaling = 1.5)
savefig("evol_zeal_phi$(ϕ)_bias$bias.pdf")