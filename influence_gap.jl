using Distributions, Plots, Random, Statistics

# Parameters
G = 50 # number of games to average over
M = 3 # memory length
N = 350 # number of players
T = 250 # number of turns
p = 0.02 # probability of joining two players of the same party
q = 0.02 # probability of joining two players of the different party
S = 2 # number of strategy tables per player
β = 0.5 # party affiliation bias
μ = 0.01 # individual learning
ϕ = 1 # weight of imitating the strategy of a player of the opposing party

# Random numbers
rng = MersenneTwister() # pseudorandom number generator
dist = Binomial(1, β) # binomial distribution

# Output
output = zeros(G, 2)

# for x = 1:50
#     party = zeros(N)
#     for i = 1:N
#         party[i] = rand(dist)
#     end

#     adjacency_matrix = [[] for i = 1:N]
#     for i = 1:N-1
#         for j = i+1:N
#             if party[i] == party[j] && rand(rng) ≤ p
#                 # if party[i] == 0 && rand(rng) ≤ p
#                 push!(adjacency_matrix[i], j)
#                 push!(adjacency_matrix[j], i)
#             elseif party[i] != party[j] && rand(rng) ≤ q
#                 # elseif party[i] == 1 && rand(rng) ≤ q
#                 push!(adjacency_matrix[i], j)
#                 push!(adjacency_matrix[j], i)
#             end
#         end
#     end

#     influence = zeros(2)
#     for i = 1:N
#         a = 0
#         for j ∈ adjacency_matrix[i]
#             if party[j] == party[i]
#                 a += 1
#             end
#         end
#         influence[Int(party[i] + 1)] = a / length(adjacency_matrix[i])
#     end

#     GB[x] = influence[1] - influence[2]
# end


# consensus_Chartists = 50
# gridlock_Chartists = 50
# Consensus_makers = 50
# Gridlockers = 50
# consensus_Zealots = 50
# gridlock_Zealots = 50
# party_Zealots = 50

consensus_Chartists = 0
gridlock_Chartists = 0
Consensus_makers = 175
Gridlockers = 175
consensus_Zealots = 0
gridlock_Zealots = 0
party_Zealots = 0

for g = 1:G # run G games
    # Initialize game
    local_history = rand(rng, 1:2^M, N) # initial local history of votes
    strategy_table_payoffs = zeros(Float32, N, S) # payoffs for strategy tables, initialized to zero
    payoffs = zeros(N) # vector of payoffs for each player
    party = zeros(N)
    for i = 1:N
        party[i] = rand(dist)#mod(i,2)
    end
    strategy_tables = rand(rng, 0:1, S * N, 2^M) # S strategy tables for the N players (note that only strategists will use these)
    vote = copy(party) # vector of the votes each player makes
    # strategy is the vector of the initial strategies for each player
    # 1=consensus_Chartists, 2=gridlock_Chartists, 3=Consensus_makers, 4=Gridlockers,
    # 5=consensus_Zealots, 6=gridlock_Zealots, 7=party_Zealots
    strategy = vcat(ones(Int, consensus_Chartists), 2 * ones(Int, gridlock_Chartists), 3 * ones(Int, Consensus_makers), 4 * ones(Int, Gridlockers), 5 * ones(Int, consensus_Zealots), 6 * ones(Int, gridlock_Zealots), 7 * ones(Int, party_Zealots))
    # Generate Erdos-Renyi random graph
    adjacency_matrix = [[] for i = 1:N]
    for i = 1:N-1
        for j = i+1:N
            # if party[i] == party[j] && rand(rng) ≤ p
            if party[i] == 0 && rand(rng) ≤ p
                push!(adjacency_matrix[i], j)
                push!(adjacency_matrix[j], i)
                # elseif party[i] != party[j] && rand(rng) ≤ q
            elseif party[i] == 1 && rand(rng) ≤ q
                push!(adjacency_matrix[i], j)
                push!(adjacency_matrix[j], i)
            end
        end
    end
    #####################################

    influence = zeros(2)
    for i = 1:N
        if length(adjacency_matrix[i]) != 0
            a = 0
            for j ∈ adjacency_matrix[i]
                if party[j] == party[i]
                    a += 1
                end
            end
            a = a / length(adjacency_matrix[i])
            if a < 1/2
                a = a-1
            end
            influence[Int(party[i] + 1)] += a
        end
    end

    output[g, 1] = influence[1]/(N-sum(party)) - influence[2]/sum(party)

    for t = 1:T
        # Determine Chartists' votes
        for i = 1:N
            if strategy[i] ∈ [1 2] # if Chartist
                best_strat = S * (i - 1) + findmax(strategy_table_payoffs[i, :])[2]
                vote[i] = strategy_tables[best_strat, local_history[i]]
            end
        end

        # Determine majority
        cur_vote = sum(vote) # sum of the current votes
        if cur_vote < N / 2
            majority = 0 # blue is majority
        elseif cur_vote > N / 2
            majority = 1 # red is majority
        elseif rand() < 1 / 2
            majority = 0 # blue is majority
        else
            majority = 1 # red is majority
        end

        # Determine payoffs for strategy tables
        for i = 1:N
            if strategy[i] == 1 # if consensus-pref Chartist
                cur_party = party[i]
                for j = 1:S
                    if majority == cur_party == strategy_tables[2*i-1, local_history[i]]
                        strategy_table_payoffs[i, j] += 1
                    elseif majority == strategy_tables[2*i-1, local_history[i]] != cur_party
                        strategy_table_payoffs[i, j] += 1 / 2
                    elseif majority != cur_party == strategy_tables[2*i-1, local_history[i]]
                        strategy_table_payoffs[i, j] -= 1 / 2
                    else
                        strategy_table_payoffs[i, j] -= 1
                    end
                end
            elseif strategy[i] == 2 # if gridlock-pref Chartist
                cur_party = party[i]
                for j = 1:S
                    if majority == cur_party == strategy_tables[2*i-1, local_history[i]]
                        strategy_table_payoffs[i, j] -= 1 / 2
                    elseif majority == strategy_tables[2*i-1, local_history[i]] != cur_party
                        strategy_table_payoffs[i, j] -= 1
                    elseif majority != cur_party == strategy_tables[2*i-1, local_history[i]]
                        strategy_table_payoffs[i, j] += 1 / 2
                    else
                        strategy_table_payoffs[i, j] += 1
                    end
                end
            end
        end

        # Determine payoffs
        for j = 1:N
            if sum(vote[adjacency_matrix[j]]) > length(adjacency_matrix[j]) / 2
                local_majority = 1
            else
                local_majority = 0
            end
            cur_party = party[j]
            cur_vote = vote[j]
            if strategy[j] ∈ [1 3 5] # for those who value consensus (consensus-pref Chartists, Consensus-makers, consensus-pref Zealots)
                if local_majority == cur_party == cur_vote
                    payoffs[j] = 1 #local_majority - 1/2 # 1
                elseif local_majority == cur_vote != cur_party
                    payoffs[j] = 1 / 2 #(local_majority - 1/2)/2 # 1/2
                elseif local_majority != cur_party == cur_vote
                    payoffs[j] = -1 / 2 #(local_majority - 1/2)/2 # 1/2
                else
                    payoffs[j] = -1 #local_majority - 1/2 # 1
                end
            elseif strategy[j] ∈ [2 4 6]  # for those who value gridlock (gridlock-pref Chartists, Gridlockers, gridlock-pref Zealots)
                if local_majority == cur_party == cur_vote
                    payoffs[j] = -1 / 2 #1 - local_majority # 1
                elseif local_majority == cur_vote != cur_party
                    payoffs[j] = -1 #(1 - local_majority)/2 # 1/2
                elseif local_majority != cur_party == cur_vote
                    payoffs[j] = 1 / 2 #(1 - local_majority)/2 # 1/2
                else
                    payoffs[j] = 1 #1 - local_majority # 1
                end
            else # party-pref Zealots
                if local_majority == cur_party
                    payoffs[j] = 1 #local_majority - 1/2 # 1
                else
                    payoffs[j] = -1 #local_majority - 1/2 # 1
                end
            end
        end

        # Determine local history for each player and thus Consensus-makers' and Gridlockers' votes
        for i = 1:N
            if sum(vote[adjacency_matrix[i]]) > length(adjacency_matrix[i]) / 2
                local_majority = 1
            else
                local_majority = 0
            end
            local_history[i] = Int(mod(2 * local_history[i], 2^M) + local_majority + 1)
            if strategy[i] == 3 # if player is a Consensus-maker, vote the same as local majority
                vote[i] = local_majority
            elseif strategy[i] == 4 # if player is a Gridlocker, vote the opposite as local majority
                vote[i] = mod(local_majority + 1, 2)
            end
        end

        # # Imitate others
        # shadow_payoffs = deepcopy(payoffs)
        # shadow_strategy = deepcopy(strategy)
        # shadow_strategy_tables = deepcopy(strategy_tables)
        # shadow_strategy_table_payoffs = deepcopy(strategy_table_payoffs)
        # for i = 1:N
        #     if adjacency_matrix[i] != [] # imitate a randomly selected neighbour
        #         neighbour = rand(adjacency_matrix[i])
        #         if party[i] == party[neighbour]
        #             #imitate = ι/(1+exp(κ*(payoffs[i]-payoffs[neighbour])))
        #             imitate = (2 + payoffs[neighbour] - payoffs[i]) / 4 # probability of imitating a neighbour of the same party
        #         else
        #             if strategy[neighbour] ∈ [1 3 5]
        #                 if payoffs[neighbour] == 1
        #                     neighbour_payoff = 1 / 2
        #                 elseif payoffs[neighbour] == -1
        #                     neighbour_payoff = -1 / 2
        #                 elseif payoffs[neighbour] == -1 / 2
        #                     neighbour_payoff = -1
        #                 else
        #                     payoffs[neighbour] == 1 / 2
        #                     neighbour_payoff = 1
        #                 end
        #                 #imitate = ι*ϕ/(1+exp(κ*(payoffs[i]-neighbour_payoff)))
        #                 imitate = ϕ * (2 + neighbour_payoff - payoffs[i]) / 4 # probability of imitating a neighbour of a different party
        #             elseif strategy[neighbour] ∈ [2 4 6]
        #                 if payoffs[neighbour] == -1 / 2
        #                     neighbour_payoff = -1
        #                 elseif payoffs[neighbour] == 1 / 2
        #                     neighbour_payoff = 1
        #                 elseif payoffs[neighbour] == 1
        #                     neighbour_payoff = 1 / 2
        #                 else
        #                     payoffs[neighbour] == -1
        #                     neighbour_payoff = -1 / 2
        #                 end
        #                 #imitate = ι*ϕ/(1+exp(κ*(payoffs[i]-neighbour_payoff)))
        #                 imitate = ϕ * (2 + neighbour_payoff - payoffs[i]) / 4 # probability of imitating a neighbour of a different party
        #             else
        #                 neighbour_payoff = -payoffs[neighbour]
        #                 #imitate = ι*ϕ/(1+exp(κ*(payoffs[i]-neighbour_payoff)))
        #                 imitate = ϕ * (2 + neighbour_payoff - payoffs[i]) / 4 # probability of imitating a neighbour of a different party
        #             end
        #         end
        #         if rand(rng) ≤ imitate
        #             shadow_payoffs[i] = payoffs[neighbour] # copy the nieghbour's payoffs
        #             shadow_strategy[i] = strategy[neighbour] # copy the neighbour's strategy
        #             if strategy[neighbour] ∈ [5 6 7] # if neighbour is a zealot, adjust vote to be in line with party
        #                 vote[i] = party[i]
        #             end
        #         end
        #     end
        # end
        # payoffs = deepcopy(shadow_payoffs)
        # strategy = deepcopy(shadow_strategy)
        # strategy_tables = deepcopy(shadow_strategy_tables)
        # strategy_table_payoffs = deepcopy(shadow_strategy_table_payoffs)

        # # Mutation
        # for i = 1:N
        #     if rand() ≤ μ
        #         strategy[i] = rand(1:7)
        #         if strategy[i] ∈ [1 2]
        #             strategy_tables[S*i-1, :] = rand(rng, 0:1, 2^M)
        #             strategy_tables[S*i, :] = rand(rng, 0:1, 2^M)
        #         elseif strategy[i] ∈ [5 6 7] # if neighbour is a zealot, adjust vote to be in line with party
        #             vote[i] = party[i]
        #         end
        #     end
        # end
    end
    output[g, 2] = maximum([mean(vote), 1 - mean(vote)])
end

pyplot()

plot(output[:,1],output[:,2],seriestype=:scatter)

# heatmap(0:7, 0:7, output, xlabel="Consensus-preferring non-Zealots", ylabel="Gridlock-preferring non-Zealots", 
# colorbar_title="Votes for majority", thickness_scaling = 1.5, clim=(0.5,1),
# xticks=([0,3.5,7],["0", "0.5", "1"]),yticks=([0,3.5,7],["0", "0.5", "1"]))
# savefig("heatmap_beta_$(β)_phi_$(ϕ)_pblue_$(p)_qred_$(q).pdf")