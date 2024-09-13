using Distributions, Plots, Random, Statistics

# Parameters
M = 3 # memory length
N = 350 # number of players
T = 250 # number of turns
p = 0.01 # probability of joining two players of the same party
q = 0.01 # probability of joining two players of the different party
S = 2 # number of strategy tables per player
β = 0.8 # party affiliation bias
μ = 0.01 # individual learning
ϕ = 0.1 # weight of imitating the strategy of a player of the opposing party

# Random numbers
rng = MersenneTwister() # pseudorandom number generator
dist = Binomial(1,β) # binomial distribution

# Arrays for simulation outputs: [consensus Chartist, gridlock Chartist, Consensus-maker, Gridlocker, consensus Zealot, gridlock Zealot, party Zealot, majority vote]
ts_output_blue = zeros(T,7) # Blue affiliated
ts_output_red = zeros(T,7) # Red affiliated
ts_output_majority = zeros(T)

# Run simulations for different initial distributions of player types
consensus_Chartists = 50
gridlock_Chartists = 50
Consensus_makers = 50
Gridlockers = 50
consensus_Zealots = 50
gridlock_Zealots = 50
party_Zealots = 50

# Initialize game
local_history = rand(rng,1:2^M,N) # initial local history of votes
strategy_table_payoffs = zeros(Float32,N,S) # payoffs for strategy tables, initialized to zero
payoffs = zeros(N) # vector of payoffs for each player
party = zeros(N)
for i=1:N
    party[i] = rand(dist)#mod(i,2)
end
strategy_tables = rand(rng,0:1,S*N,2^M) # S strategy tables for the N players (note that only strategists will use these)
vote = copy(party) # vector of the votes each player makes
# strategy is the vector of the initial strategies for each player
# 1=consensus_Chartists, 2=gridlock_Chartists, 3=Consensus_makers, 4=Gridlockers,
# 5=consensus_Zealots, 6=gridlock_Zealots, 7=party_Zealots
strategy = vcat(ones(Int,consensus_Chartists),2*ones(Int,gridlock_Chartists),3*ones(Int,Consensus_makers),4*ones(Int,Gridlockers),5*ones(Int,consensus_Zealots),6*ones(Int,gridlock_Zealots),7*ones(Int,party_Zealots),) 

# Generate Erdos-Renyi random graph
adjacency_matrix = [[] for i = 1:N]
for i=1:N-1
    for j=i+1:N
        if party[i] == party[j] && rand(rng) ≤ p
            push!(adjacency_matrix[i],j)
            push!(adjacency_matrix[j],i)
        elseif party[i] != party[j] && rand(rng) ≤ q
            push!(adjacency_matrix[i],j)
            push!(adjacency_matrix[j],i)
        end
    end
end

# Run a single realization
for t=1:T
    # Determine Chartists' votes
    for i=1:N
        if strategy[i] == 1 # if consensus-pref
            best_strat = S*(i-1) + findmax(strategy_table_payoffs[i,:])[2]
            vote[i] = strategy_tables[best_strat,local_history[i]]
        elseif strategy[i] == 2 # if gridlock-pref
            best_strat = S*(i-1) + findmin(strategy_table_payoffs[i,:])[2]
            vote[i] = strategy_tables[best_strat,local_history[i]]
        end
    end

    # Determine majority
    cur_vote = sum(vote) # sum of the current votes
    if cur_vote < N/2
        majority = 0 # blue is majority
    elseif cur_vote > N/2
        majority = 1 # red is majority
    elseif rand() < 1/2
        majority = 0 # blue is majority
    else
        majority = 1 # red is majority
    end

    # Record strategy distributions
    ts_output_red[t,:] = [sum(x->x==1,strategy.*party)/N,sum(x->x==2,strategy.*party)/N,sum(x->x==3,strategy.*party)/N,sum(x->x==4,strategy.*party)/N,sum(x->x==5,strategy.*party)/N,sum(x->x==6,strategy.*party)/N,sum(x->x==7,strategy.*party)/N]
    ts_output_blue[t,:] = [sum(x->x==1,strategy)/N,sum(x->x==2,strategy)/N,sum(x->x==3,strategy)/N,sum(x->x==4,strategy)/N,sum(x->x==5,strategy)/N,sum(x->x==6,strategy)/N,sum(x->x==7,strategy)/N] .- ts_output_red[t,:]
    ts_output_majority[t] = maximum([cur_vote,N-cur_vote])/N

    # Determine payoffs for strategy tables
    for i=1:N
        cur_party = party[i]
        for j=1:S
            if majority == cur_party == strategy_tables[2*i-1,local_history[i]]
                strategy_table_payoffs[i,j] += 1
            elseif majority == strategy_tables[2*i-1,local_history[i]] != cur_party
                strategy_table_payoffs[i,j] += 1/2
            elseif majority != cur_party == strategy_tables[2*i-1,local_history[i]]
                strategy_table_payoffs[i,j] -= 1/2
            else
                strategy_table_payoffs[i,j] -= 1
            end
        end
    end

    # Determine payoffs for all consensus preferred players
    for j=1:N
        if sum(vote[adjacency_matrix[j]]) > length(adjacency_matrix[j])/2
            local_majority = 1
        else
            local_majority = 0
        end
        cur_party = party[j]
        cur_vote = vote[j]
        if strategy[j] ∈ [1 3 5] # for those who value consensus (consensus-pref Chartists, Consensus-makers, consensus-pref Zealots)
            if local_majority == cur_party == cur_vote
                payoffs[j] += local_majority - 1/2 # 1
            elseif local_majority == cur_vote != cur_party
                payoffs[j] += (local_majority - 1/2)/2 # 1/2
            elseif local_majority != cur_party == cur_vote
                payoffs[j] -= (local_majority - 1/2)/2 # 1/2
            else
                payoffs[j] -= local_majority - 1/2 # 1
            end
        elseif strategy[j] ∈ [2 4 6]  # for those who value gridlock (gridlock-pref Chartists, Gridlockers, gridlock-pref Zealots)
            if local_majority != cur_party == cur_vote
                payoffs[j] += 1 - local_majority # 1
            elseif local_majority != cur_vote != cur_party
                payoffs[j] += (1 - local_majority)/2 # 1/2
            elseif local_majority == cur_party == cur_vote
                payoffs[j] -= (1 - local_majority)/2 # 1/2
            else
                payoffs[j] -= 1 - local_majority # 1
            end
        else # party-pref Zealots
            if local_majority == cur_party == cur_vote
                payoffs[j] += local_majority - 1/2 # 1
            elseif local_majority !== cur_vote == cur_party
                payoffs[j] += (1 - local_majority)/2 # 1/2
            elseif local_majority != cur_party != cur_vote
                payoffs[j] -= (1 - local_majority)/2 # 1/2
            else
                payoffs[j] -= local_majority - 1/2 # 1
            end
        end
    end

    # Determine local history for each player and thus Consensus-makers' and Gridlockers' votes
    for i=1:N
        if sum(vote[adjacency_matrix[i]]) > length(adjacency_matrix[i])/2
            local_majority = 1
        else
            local_majority = 0
        end
        local_history[i] = Int(mod(2*local_history[i],2^M) + local_majority + 1)
        if strategy[i] == 3 # if player is a Consensus-maker, vote the same as local majority
            vote[i] = local_majority
        end
        if strategy[i] == 4 # if player is a Gridlocker, vote the opposite as local majority
            vote[i] = mod(local_majority+1,2)
        end
    end

    # Imitate others
    shadow_payoffs = deepcopy(payoffs)
    shadow_strategy = deepcopy(strategy)
    shadow_strategy_tables = deepcopy(strategy_tables)
    shadow_strategy_table_payoffs = deepcopy(strategy_table_payoffs)
    for i=1:N
        if adjacency_matrix[i] != [] # imitate a randomly selected neighbour
            neighbour = rand(adjacency_matrix[i])
            if party[i]==party[neighbour]
                imitate=(2+mean(payoffs[neighbour])-mean(payoffs[i]))/4 # probability of imitating a neighbour of the same party
            else
                imitate=ϕ*(2+mean(payoffs[neighbour])-mean(payoffs[i]))/4 # probability of imitating a neighbour of a different party
            end
            if rand(rng) ≤ imitate
                shadow_payoffs[i] = payoffs[neighbour] # copy the nieghbour's payoffs
                shadow_strategy[i] = strategy[neighbour] # copy the neighbour's strategy
                if strategy[neighbour] ∈ [5 6 7] # if neighbour is a zealot, adjust vote to be in line with party
                    vote[i] = party[i]
                end
            end
        end
    end
    payoffs = deepcopy(shadow_payoffs)
    strategy = deepcopy(shadow_strategy)
    strategy_tables = deepcopy(shadow_strategy_tables)
    strategy_table_payoffs = deepcopy(shadow_strategy_table_payoffs)

    # Mutation
    for i=1:N
        if rand() ≤ μ
            strategy[i]=rand(1:7)
            if strategy[i] ∈ [5 6 7] # if neighbour is a zealot, adjust vote to be in line with party
                vote[i] = party[i]
            end
        end
    end

end

pyplot()
ts_output = ts_output_blue .+ ts_output_red
pl_blue = plot(1:T,hcat(ts_output_blue,ts_output_majority),label=["consensus-pref Chartists" "gridlock-pref Chartists" "Consensus-makers" "Gridlockers" "consensus-pref Zealots" "gridlock-pref Zealots" "party-pref Zealots" "majority vote"],position=:outerright)
pl_red = plot(1:T,hcat(ts_output_red,ts_output_majority),label=["consensus-pref Chartists" "gridlock-pref Chartists" "Consensus-makers" "Gridlockers" "consensus-pref Zealots" "gridlock-pref Zealots" "party-pref Zealots" "majority vote"],position=:outerright)
pl = plot(1:T,hcat(ts_output,ts_output_majority),label=["consensus-pref Chartists" "gridlock-pref Chartists" "Consensus-makers" "Gridlockers" "consensus-pref Zealots" "gridlock-pref Zealots" "party-pref Zealots" "majority vote"],position=:outerright)
plot(pl_blue, pl_red, pl, layout=(3, 1))