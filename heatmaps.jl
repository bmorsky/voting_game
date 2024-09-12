using Distributions, Plots, Random, Statistics

# Output
output = fill(NaN,(11,11))

# Parameters
G = 50 # number of games to average over
M = 3 # memory length
N = 500 # number of players
T = 250 # number of turns
p = 0.01 # probability of joining two players of the same party
q = 0.01 # probability of joining two players of the different party
S = 2 # number of strategy tables per player
β = 0.5 # party affiliation bias
μ = 0.01 # individual learning
ϕ = 0 # weight of imitating the strategy of a player of the opposing party

# Random numbers
rng = MersenneTwister() # pseudorandom number generator
dist = Binomial(1,β) # binomial distribution

global count = 1

for consensus_pref = 0:50:N
    for gridlock_pref = 0:50:N-consensus_pref
        avg_vote = 0
        for g = 1:G # run G games
            ########## Initialize game ##########
            party = rand(dist,N) # party affiliation for each player

            consensus_votes = round(sum(rand(d,num_consensus_makers))/num_consensus_makers) # initial consensus votes
            history = rand(rng,1:2^M) # outcome of past elections
            
            payoffs = zeros(Float32,num_strategists,S) # payoffs for strategy tables
            strategists_parties = rand(d,num_strategists)
            strategists_votes = Array{Int,1}(undef,num_strategists) # votes taken: vote=1, vote=0
            strategy_tables = rand(rng,0:1,S*N,2^M) # S strategy tables for the N players
            strategy_tables_payoffs = zeros(N,S) # strategy tables' payoffs

            vote = Array{Float32,1}(undef,num_turns)
            zealots_votes = sum(rand(d,num_zealots)) #sum(rand(0:1,num_zealots))

            # Generate Erdos-Renyi random network
            adjacency_matrix = [[] for i = 1:N]
            for i=1:N-1
                for j=i+1:N
                    if party[i]==party[j] && rand(rng) ≤ p
                        push!(adjacency_matrix[i],j)
                        push!(adjacency_matrix[j],i)
                    elseif rand(rng) ≤ q
                        push!(adjacency_matrix[i],j)
                        push!(adjacency_matrix[j],i)
                    end
                end
            end
            #####################################

            for t=1:T
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
            avg_vote += mean(vote)
        end
        output[Int(consensus_pref/50)+1,Int(gridlock_pref/50)+1] = avg_vote/G
        global count = count + 1
    end
end

pyplot()

heatmap(0:10, 0:10, output, xlabel="Consensus-prefering non-Zealots", ylabel="Gridlock-prefering non-Zealots", 
colorbar_title="Votes for majority", thickness_scaling = 1.5, clim=(0.5,1))
savefig("heatmap_bias_($β)_homophily_($ϕ).pdf")