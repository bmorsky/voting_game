using Distributions, Plots, Random, Statistics

# Parameters
M = 2 # memory length
N = 1000 # number of players
num_turns = 5000 # number of turns
S = 2 # number of strategy tables per player
p = 1 #N/N # base probability δ/N of joining two nodes with mean degree δ
μ = 0.1 # rate of immitation of others' strategies
μᵢ = 0.01 # individual learning
ϕ = 1 # weight of imitating the strategy of a player of the opposing party

# Arrays for simulation outputs
ts_output = zeros(num_turns,6) # [num_consensus_makers, num_strategists, num_zealots, num_gridlockers, majority vote]
ts_output0 = zeros(num_turns,6) # [num_consensus_makers, num_strategists, num_zealots, num_gridlockers, majority vote]
ts_output1 = zeros(num_turns,6) # [num_consensus_makers, num_strategists, num_zealots, num_gridlockers, majority vote]

# Functions
function χ(i) # imitation weight for each strategy
    if i == 1 # if player is a consensus-maker
        return 1
    elseif i==2 # if player is a strategist
        return 1
    else # if player is a zealot
        return 1
    end
end

# Random numbers
rng = MersenneTwister() # pseudorandom number generator
bias = 0.7 # probability that a player is of party 1
d = Binomial(1,bias) # binomial distribution

# Run simulations for different initial conditions of consensus-makers, strategists, and zealots
# num_consensus_makers = rand(0:N)
# num_strategists = rand(0:N-num_consensus_makers)
# num_zealots = N-num_consensus_makers-num_strategists

num_consensus_makers = 200
num_strategists = 200
num_zealots = 200
num_gridlockers = 200
num_zealot_gridlockers = 200

# Initialize game
history = rand(rng,1:2^M,N) # initial history of votes
strategy_table_payoffs = zeros(Float32,N,S) # payoffs for strategy tables, initialized to zero
payoffs = zeros(N) # vector of payoffs for each player
party = rand(d,N) # vector of party affiliation for each player
# party = zeros(N)
# for i=1:N
#     party[i] = mod(i,2)
# end
strategy_tables = rand(rng,0:1,S*N,2^M) # S strategy tables for the N players (note that only strategists will use these)
vote = rand(d,N) # vector of the votes each player makes
# strategy is the vector of the initial strategies for each player: 1=consensus-makers, 2=strategists, 3=zealots, 4=gridlockers, 5=zealotgridlockers
strategy = vcat(ones(Int,num_consensus_makers),2*ones(Int,num_strategists),3*ones(Int,num_zealots),4*ones(Int,num_gridlockers),5*ones(Int,num_zealot_gridlockers)) 
# Zealot's votes
for j=1:N
    if strategy[j] == 3 || strategy[j] == 5
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
        if strategy[j] == 2
            best_strat = S*(j-1) + findmax(strategy_table_payoffs[j,:])[2]
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
    # ts_output[turn,:] = [sum(x->x==0,strategy)/N,sum(x->x==1,strategy)/N,sum(x->x==2,strategy)/N,maximum([cur_vote,N-cur_vote])/N]
    ts_output[turn,:] = [sum(x->x==1,strategy)/N,sum(x->x==2,strategy)/N,sum(x->x==3,strategy)/N,sum(x->x==4,strategy)/N,sum(x->x==5,strategy)/N,maximum([cur_vote,N-cur_vote])/N]
    ts_output1[turn,:] = [sum(x->x==1,strategy.*party)/N,sum(x->x==2,strategy.*party)/N,sum(x->x==3,strategy.*party)/N,sum(x->x==4,strategy.*party)/N,sum(x->x==5,strategy.*party)/N,cur_vote/N]
    ts_output0[turn,:] = [sum(x->x==1,strategy)/N,sum(x->x==2,strategy)/N,sum(x->x==3,strategy)/N,sum(x->x==4,strategy)/N,sum(x->x==5,strategy)/N,1] .- ts_output1[turn,:]

    # Stategic voters: determine payoffs for their strategy tables
    for j=1:N
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

    # Determine payoffs for all players
    for j=1:N
        if sum(vote[adjacency_matrix[j]]) > length(adjacency_matrix[j])/2
            local_majority = 1
        else
            local_majority = 0
        end
        cur_party = party[j]
        cur_vote = vote[j]
        if strategy[j] < 4 # for those who value consensus (note this includes zealots)
            if local_majority == cur_party == cur_vote
                payoffs[j] += local_majority - 1/2 # 1
            elseif local_majority == cur_vote != cur_party
                payoffs[j] += (local_majority - 1/2)/2 # 1/2
            elseif local_majority != cur_party == cur_vote
                payoffs[j] -= (local_majority - 1/2)/2 # 1/2
            else
                payoffs[j] -= local_majority - 1/2 # 1
            end
        elseif strategy[j] == 4 # for those who value consensus (note this includes zealots)
            if local_majority != cur_party == cur_vote
                payoffs[j] += 1 - local_majority # 1
            elseif local_majority != cur_vote != cur_party
                payoffs[j] += (1 - local_majority)/2 # 1/2
            elseif local_majority == cur_party == cur_vote
                payoffs[j] -= (1 - local_majority)/2 # 1/2
            else
                payoffs[j] -= 1 - local_majority # 1
            end
        else # for those who value consensus (note this includes zealots)
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

                # Determine local history for each player and thus consensus-makers' and gridlockers' votes
                for j=1:N
                    if sum(vote[adjacency_matrix[j]]) > length(adjacency_matrix[j])/2
                        local_majority = 1
                    else
                        local_majority = 0
                    end
                    history[j] = Int(mod(2*history[j],2^M) + majority + 1)
                    if strategy[j] == 1 # if player is a consensus-maker, vote the same as local majority
                        vote[j] = local_majority
                    end
                    if strategy[j] == 4 # if player is a gridlocker, vote the opposite as local majority
                        vote[j] = mod(local_majority+1,2)
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
                            if strategy[j] == 3 # if neighbour is a zealot, adjust vote to be in line with party
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
                            strategy[j]=rand(1:5) #rand(1:3)
                            if strategy[j] == 3 # if neighbour is a zealot, adjust vote to be in line with party
                                vote[j] = party[j]
                            end # if neighbour is not a zealot, adjust vote to be that of neighbour
                            # strategy_tables[i] = strategy_tables[j] # copy the neighbour's strategy tables (only has an impact if neighbour is a strategist)
                            # strategy_table_payoffs[i] = strategy_table_payoffs[j] # copy the neighbour's strategy tables' payoffs
                        end
                    end

            end

pyplot()


out0 = [ts_output0[:,1] ts_output0[:,5]]
out1 = [ts_output1[:,1] ts_output1[:,5]]
out = [ts_output[:,1] ts_output[:,5]]

pl0 = plot(1:num_turns,out0,label=["Consensus-makers" "Strategists" "Zealots" "Gridlockers" "Zealot Gridlockers"],legend=false)
pl1 = plot(1:num_turns,out1,label=["Consensus-makers" "Strategists" "Zealots" "Gridlockers" "Zealot Gridlockers"],legend=false)
pl = plot(1:num_turns,out,label=["Consensus-makers" "Strategists" "Zealots" "Gridlockers" "Zealot Gridlockers"],legend=false)


plot(pl0, pl1, pl, layout=(3, 1))
plot(1:num_turns,ts_output,label=["Consensus-makers" "Strategists" "Zealots" "Gridlockers" "Zealot Gridlockers" "Majority vote"])


