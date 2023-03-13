## Remember to run Pkg> activate .
using Meshes, Plots, StatsBase, GLM, Combinatorics, Distributions
using Statistics, LinearAlgebra
import Random
include("../src/PointProcessLearning.jl")
using .PointProcessLearning

import ColorSchemes.okabe_ito
cscheme = okabe_ito

#########################################
# Set up window
#########################################
W = Box((0.,0.),(1.,1.))
Wplus = enlarge(W,0.25)


#########################################
# Set up Geyer process
#########################################
# Parameters
# Distance between points
R = 0.05; sat=7;
t(u::Point,s::PointSet) = sum(dist(u,x)<R for x in s.items)
# Parameters
λ₀ = 100; g₀ = 1.25;
θ₀ = [log(λ₀),log(g₀)]
S = [(u,s)->1, (u,s)->min(sat,t(u,s))]
#deleteat!(S,1)
# For fixed finite process
#θ₀ = [log(g₀)]
#S = [(u,s)->min(sat,t(u,s))]
# Generate patter
logλ(u,s,θ) = sum(θ[i]*S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
p = GibbsProcess(logλ)
#p = FiniteGibbsProcess2(logλ,200)
s = sample_pp(p,Wplus)
s = sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=80_000)
#s = sample_pp2(Random.GLOBAL_RNG,p,Wplus; niter=50_000)
plot(s,W)
StoyanGrabarnik(s,logλ,W)

#########################################
# Generate original, and reconstructed patterns
#########################################
# Resolutions to be used in reconstruction
resolutions = [1,2,4,8,16,32,64]

noriginals = 20
nresolutions = length(resolutions)

patterns = [sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=80_000) for _ in 1:noriginals]
L = mean(N(s,W) for s in patterns)
poispatterns = [sample_pp(PoissonProcess(L),W) for _ in 1:noriginals]
#poispatterns = [sample_pp(PoissonProcess(L),W) for _ in 1:200]

#########################################
# Analyze Ni distributions
#########################################

Nvectors = Array{Vector{Int64}}(undef, (noriginals,nresolutions))
for i in 1:noriginals, j in 1:nresolutions
    s = patterns[i]
    boxes = PointProcessLearning.partitions(W,resolutions[j])
    Nvectors[i,j] = [N(s,box) for box in boxes]
end

poisNvectors = Array{Vector{Int64}}(undef, (length(poispatterns),nresolutions))
for i in eachindex(poispatterns), j in 1:nresolutions
    s = poispatterns[i]
    boxes = PointProcessLearning.partitions(W,resolutions[j])
    poisNvectors[i,j] = [N(s,box) for box in boxes]
end

### Distribution of distances for Nvectors
countdistances = Array{Float64}(undef, (nresolutions, Int(noriginals*(noriginals-1)/2)))
for k in 1:nresolutions
    dists = []
    for i in 1:noriginals-1
        for j in i+1:noriginals
            distij = sqrt(sum((Nvectors[i,k] .- Nvectors[j,k]).^2))
            push!(dists,distij)
        end
    end
    countdistances[k,:] = dists
end
plts = [histogram(countdistances[k,:]) for k in 1:nresolutions]
plot(plts...)
qtls = [quantile(countdistances[k,:], 0.95) for k in 1:nresolutions]



#prior_distr = product_distribution([Uniform(log(50),log(150)),Uniform(log(1.0),log(1.7))])

prior_distr = product_distribution([Normal(log(100),1),truncated(Normal(log(1),0.5),0,nothing)])
rand(prior_distr)
#rand(prior_distr)

function ParamEstimationABC(counts::Vector{Int64}, boxes::Vector{Box{2, Float64}}, W::Box, logλ::Function, prior_distr, qlimit; nproposals=500)
    #prior_distr = product_distribution([Uniform(log(50),log(150)),Uniform(log(1.0),log(1.75))])
    #prior_distr = prior
    proposals = rand(prior_distr,nproposals)
    accept = Array{Bool}(undef,nproposals)
    #Threads.@threads while length(posterior_samples) < 100
    Threads.@threads for i in 1:nproposals
        #if (length(posterior_samples)%10 == 0)&(length(posterior_samples)!=0)
        #    println("Finished $(length(posterior_samples)) %")
        #end
        if (i%floor(Int,nproposals/4)==0)
            println("i=$i")
        end
        Theta_prop = proposals[:,i]
        logλ_prop(u,s) = logλ(u,s,Theta_prop)
        p = GibbsProcess(logλ_prop)
        #s = sample_pp(p,W)
        s_prop = sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=30_000)
        counts_prop = [N(s_prop,box) for box in boxes]
        count_dist = sqrt(sum((counts_prop .- counts).^2))
        accept[i] = (count_dist < qlimit)
        #if count_dist < qlimit
            #push!(posterior_samples, Theta_prop)
        #    accept[i] = true
        #end
    end 
    posterior_samples = proposals[:, accept]
    return proposals, accept
end


function ParamEstimationABC(intensities::Vector{Int64}, boxes::Vector{Box{2, Float64}}, W::Box, logλ::Function, prior_distr, qlimit; nproposals=500)
    #prior_distr = product_distribution([Uniform(log(50),log(150)),Uniform(log(1.0),log(1.75))])
    #prior_distr = prior
    proposals = rand(prior_distr,nproposals)
    accept = Array{Bool}(undef,nproposals)
    #Threads.@threads while length(posterior_samples) < 100
    Threads.@threads for i in 1:nproposals
        #if (length(posterior_samples)%10 == 0)&(length(posterior_samples)!=0)
        #    println("Finished $(length(posterior_samples)) %")
        #end
        if (i%floor(Int,nproposals/4)==0)
            println("i=$i")
        end
        Theta_prop = proposals[:,i]
        logλ_prop(u,s) = logλ(u,s,Theta_prop)
        p = GibbsProcess(logλ_prop)
        #s = sample_pp(p,W)
        s_prop = sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=30_000)
        intensities_prop = [N(s_prop,box) for box in boxes]*length(boxes)
        intensity_dist = sqrt(sum((intensities_prop .- intensities).^2))
        accept[i] = (intensity_dist < qlimit)
        #if count_dist < qlimit
            #push!(posterior_samples, Theta_prop)
        #    accept[i] = true
        #end
    end 
    posterior_samples = proposals[:, accept]
    return proposals, accept
end


rand(prior_distr,10)[1,:]

jval = 3
counts = Nvectors[1,jval]
intensities = Nvectors[1,jval].*(resolutions[jval]^2)
qtls = [quantile(countdistances[k,:].*resolutions[jval]^2, 0.95) for k in 1:nresolutions]


boxes = PointProcessLearning.partitions(W,resolutions[jval])
qtls[jval]
proposals, accept = ParamEstimationABC(intensities,boxes,W,logλ,prior_distr,qtls[3]; nproposals=500)
#proposals, accept = ParamEstimationABC(counts,boxes,W,logλ,prior_distr,qtls[3]; nproposals=500)
results = proposals[:,accept]
scatter(results[1,:],results[2,:])
scatter!([θ₀[1]],[θ₀[2]], mc=:red, ms=6, labels="true")
scatter!([mean(results[1,:])],[mean(results[2,:])], mc=:black, ms=6, labels="mean")

# both accepted and rejected
scatter(proposals[1,:],proposals[2,:], mc= [accept[i] ? :blue : :gray for i in eachindex(accept)], opacity=0.5)
scatter!([θ₀[1]],[θ₀[2]], mc=:red, ms=6, labels="true")
scatter!([mean(results[1,:])],[mean(results[2,:])], mc=:black, ms=6, labels="mean")


results_all = Array{Matrix}(undef, nresolutions)
#results_all = Array{Matrix}(undef, 4)
Threads.@threads for k in 1:nresolutions #1:nresolutions
    #jval = k+1
    counts = Nvectors[1,k]
    intensities = counts.*resolutions[k]^2
    boxes = PointProcessLearning.partitions(W,resolutions[k])
    proposals, accept = ParamEstimationABC(intensities,boxes,W,logλ,prior_distr,qtls[k]; nproposals=500)
    results_all[k] = proposals[:,accept]
end

kk = 2
rslts = results_all[k]
scatter(rslts[1,:],rslts[2,:], opacity=0.3, labels=false)
scatter!([θ₀[1]],[θ₀[2]], mc=:red, ms=6, labels="true")
scatter!([mean(rslts[1,:])],[mean(rslts[2,:])], mc=:black, ms=6, labels="mean")

plts = [plot(title="n=$(resolutions[k]) ($(round(Int,100*length(results_all[k][1,:])/500))% accept)",ylims=(0,1),xlims=(1,5)) for k in 1:nresolutions]
for k in 1:nresolutions
    rslts = results_all[k]
    scatter!(plts[k],rslts[1,:],rslts[2,:], opacity=0.3, labels=false)
    scatter!(plts[k],[θ₀[1]],[θ₀[2]], mc=:red, ms=6, labels="true")
    scatter!(plts[k],[mean(rslts[1,:])],[mean(rslts[2,:])], mc=:black, ms=6, labels="mean")
end
plot(plts..., layout=(4,2))
plot!(size=(500,900))
#savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/GeyerProcess_ABC_onesample.pdf")


# Store and load data
using BSON
bson("abc_results_all.bson",Dict(:res=>results_all))
load_data = BSON.load("abc_results.bson")
load_data[:res]






# Do ABC with summary Statistics
# can we use stuff like mean directly? It depends of course on resolution...as does variance.
# But we could normalize mean(N)*|A| where |A|=1/k^2
jval = 5
plts = [plot() for k in 1:nresolutions]
for k in 1:nresolutions
    histogram!(plts[k],mean.(Nvectors[:,k].*(resolutions[k]^2)))
end
plot(plts...)
plts = [plot() for k in 1:nresolutions]
for k in 2:nresolutions
    histogram!(plts[k],10*sqrt.(var.(Nvectors[:,k].*(resolutions[k]))))
end
plot(plts...)





function ParamEstimationABC_stats(stats::Vector{Float64}, boxes::Vector{Box{2, Float64}}, W::Box, logλ::Function, prior_distr, qlimit; nproposals=500)
    #prior_distr = product_distribution([Uniform(log(50),log(150)),Uniform(log(1.0),log(1.75))])
    #prior_distr = prior
    res = length(boxes)
    proposals = rand(prior_distr,nproposals)
    accept = Array{Bool}(undef,nproposals)
    #Threads.@threads while length(posterior_samples) < 100
    for i in 1:nproposals
        #if (length(posterior_samples)%10 == 0)&(length(posterior_samples)!=0)
        #    println("Finished $(length(posterior_samples)) %")
        #end
        if (i%floor(Int,nproposals/4)==0)
            println("i=$i")
        end
        Theta_prop = proposals[:,i]
        logλ_prop(u,s) = logλ(u,s,Theta_prop)
        p = GibbsProcess(logλ_prop)
        #s = sample_pp(p,W)
        s_prop = sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=30_000)
        nvector = [N(s_prop,box) for box in boxes]
        stats_prop = [mean(nvector*res^2), 10*sqrt(var(nvector*res))]
        stats_dist = sqrt(sum((stats_prop .- stats).^2))
        accept[i] = (stats_dist < qlimit)
        #if count_dist < qlimit
            #push!(posterior_samples, Theta_prop)
        #    accept[i] = true
        #end
    end 
    #posterior_samples = proposals[:, accept]
    return proposals, accept
end

jval = 3
nvector = Nvectors[1,jval]
stats = [mean(nvector*resolutions[jval]^2), 10*sqrt(var(nvector*resolutions[jval]))]

boxes = PointProcessLearning.partitions(W,resolutions[jval])
proposals, accept = ParamEstimationABC_stats(stats,boxes,W,logλ,prior_distr,50; nproposals=200)
#proposals, accept = ParamEstimationABC(counts,boxes,W,logλ,prior_distr,qtls[3]; nproposals=500)
results = proposals[:,accept]
scatter(results[1,:],results[2,:])
scatter!([θ₀[1]],[θ₀[2]], mc=:red, ms=6, labels="true")
scatter!([mean(results[1,:])],[mean(results[2,:])], mc=:black, ms=6, labels="mean")


# both accepted and rejected
scatter(proposals[1,:],proposals[2,:], mc= [accept[i] ? :blue : :gray for i in eachindex(accept)], opacity=0.5)
scatter!([θ₀[1]],[θ₀[2]], mc=:red, ms=6, labels="true")
scatter!([mean(results[1,:])],[mean(results[2,:])], mc=:black, ms=6, labels="mean")



μ = vec(mean(hcat(Nvectors[:,3]...), dims=2))
Σ = cov(hcat(Nvectors[:,3]...)')

like = MvNormal(μ,Σ)
rand(like)
logpdf(like,Nvectors[1,3])

using Statistics
function syntheticlikelihood(θ::Vector, logλ::Function, boxes::Vector{Box{2, Float64}},W::Box; nsamples = 15)
    pp = GibbsProcess((u,s) -> logλ(u,s,θ))
    counts = Matrix{Float64}(undef,(length(boxes),nsamples))
    for n in 1:nsamples
        s = sample_pp(Random.GLOBAL_RNG,pp,W; niter=30_000)
        counts[:,n] = [N(s,box) for box in boxes]
    end
    μ = vec(mean(counts, dims=2))
    #Σ = cov(hcat([counts[:,k] for k in 1:nsamples]...)')
    Σ = Statistics.cov(counts')
    #Σ = cov(counts, vardim=2)
    # correct if semidefinite
    Σ = Symmetric(Σ) + max(eps(),-minimum(eigvals(Symmetric(Σ))))*I
    #alt1
    #Σ = Symmetric(Σ) + max(0,abs(minimum(eigvals(Symmetric(Σ)))))*I
    #alt2
    #lambda = minimum(filter(x->(x<0),eigvals(Σ)))
    #Σ = Σ + abs(lambda)*I
    
    #eigvals(Σ)
    #return counts
    #return (μ,Σ)
    return MvNormal(μ,Σ)
end


function ParamEstimationSynthetic(counts,prior,logλ,boxes,W; postsamples = 20)
    pr(θ) = logpdf(prior,θ)
    Q(θ) = product_distribution([Normal(θ[1],0.25),Normal(θ[2],0.125)])
    q(θnew,θold) = logpdf(Q(θold),θnew)
    L(θ) = syntheticlikelihood(θ,logλ,boxes,W)
    l(N,θ) = logpdf(L(θ),N)
    A(θnew,θold) = min(1,exp(l(counts,θnew)+pr(θnew)+q(θold,θnew) - l(counts,θold)-pr(θold)-q(θnew,θold)))
    # Initialize
    #θold = rand(prior)
    θold = θ₀
    θs = Array{Float64}(undef, (postsamples,length(θold)))
    θs[1,:] = θold
    for i in 2:postsamples
        println("i=$i")
        θold = θs[i-1,:]
        θnew = rand(Q(θold))
        while rand() < A(θnew,θold)
            θnew = rand(Q(θold))
        end
        θs[i,:] = θnew
    end
    return θs
end


jval = 2
counts = Nvectors[2,jval]
boxes = PointProcessLearning.partitions(W,resolutions[jval])

# Note: Try also with lambda = counts.*measure(boxes)
thetavals = ParamEstimationSynthetic(counts,prior_distr,logλ,boxes,Wplus; postsamples=400)

scatter(thetavals[200:400,1],thetavals[200:400,2], mc=:blue, alpha=0.5)
scatter!([θ₀[1]],[θ₀[2]])

plot(thetavals[:,1])
hline!([ θ₀[1] ])

plot(thetavals[:,2])
hline!([ θ₀[2] ])








function syntheticlikelihood_ints(θ::Vector, logλ::Function, boxes::Vector{Box{2, Float64}},W::Box; nsamples = 10)
    pp = GibbsProcess((u,s) -> logλ(u,s,θ))
    λs = Matrix{Float64}(undef,(length(boxes),nsamples))
    Threads.@threads for n in 1:nsamples
        s = sample_pp(Random.GLOBAL_RNG,pp,W; niter=30_000)
        #λs[:,n] = [N(s,box) for box in boxes]
        λs[:,n] = [N(s,box)*measure(box) for box in boxes]
    end
    μ = vec(mean(λs, dims=2))
    #Σ = cov(hcat([counts[:,k] for k in 1:nsamples]...)')
    Σ = Statistics.cov(λs')
    #Σ = cov(counts, vardim=2)
    # correct if semidefinite
    Σ = Symmetric(Σ) + max(eps(),-minimum(eigvals(Symmetric(Σ))))*I
    #alt1
    #Σ = Symmetric(Σ) + max(0,abs(minimum(eigvals(Symmetric(Σ)))))*I
    #alt2
    #lambda = minimum(filter(x->(x<0),eigvals(Σ)))
    #Σ = Σ + abs(lambda)*I
    
    #eigvals(Σ)
    #return counts
    #return (μ,Σ)
    return MvNormal(μ,Σ)
end

function syntheticlikelihood_safe(θ::Vector, logλ::Function, boxes::Vector{Box{2, Float64}},W::Box; nsamples = 10)
    try 
        return syntheticlikelihood_ints(θ,logλ,boxes,W; nsamples=nsamples)
    catch e
        println(e)
        println("trying again")
        return syntheticlikelihood_safe(θ,logλ,boxes,W; nsamples=nsamples)
    end
end


function ParamEstimationSynthetic_ints(λ,prior,logλ,boxes,W; postsamples = 20)
    pr(θ) = logpdf(prior,θ)
    Q(θ) = product_distribution([Normal(θ[1],0.28),Normal(θ[2],0.13)])
    q(θnew,θold) = logpdf(Q(θold),θnew)
    L(θ) = syntheticlikelihood_safe(θ,logλ,boxes,W; nsamples=10)
    l(N,θ) = logpdf(L(θ),N)
    A(θnew,θold) = min(1,exp(l(λ,θnew)+pr(θnew)+q(θold,θnew) - l(λ,θold)-pr(θold)-q(θnew,θold)))
    # Initialize
    #θold = rand(prior)
    θold = θ₀
    θs = Array{Float64}(undef, (postsamples,length(θold)))
    θs[1,:] = θold
    for i in 2:postsamples
        rejects = 0
        θold = θs[i-1,:]
        θnew = rand(Q(θold))
        while rand() > A(θnew,θold)
            θnew = rand(Q(θold))
            rejects = rejects + 1
        end
        θs[i,:] = θnew
        acceptrate = (rejects>0) ? 1/rejects : 1
        println("i=$i , acceptance rate = $(round(acceptrate,digits=3))")
    end
    return θs
end

jval = 3
counts = Nvectors[5,jval]
boxes = PointProcessLearning.partitions(W,resolutions[jval])
λ = counts.*measure.(boxes)

# Note: Try also with lambda = counts.*measure(boxes)
thetavals = ParamEstimationSynthetic_ints(λ,prior_distr,logλ,boxes,Wplus; postsamples=300)

scatter(thetavals[:,1],thetavals[:,2], mc=:blue, alpha=0.5)
plot(thetavals[100:300,1],thetavals[100:300,2], mc=:blue, alpha=0.5)
scatter!([θ₀[1]],[θ₀[2]])
scatter!([mean(thetavals[:,1])],[mean(thetavals[:,2])])

plot(thetavals[:,1])
hline!([ θ₀[1], mean(thetavals[:,1]) ])

plot(thetavals[:,2])
hline!([ θ₀[2], mean(thetavals[:,2]) ])


thetavec = Vector{Matrix{Float64}}(undef,nresolutions)
Threads.@threads for j in 1:nresolutions
    counts = Nvectors[5,j]
    boxes = PointProcessLearning.partitions(W,resolutions[j])
    λ = counts.*measure.(boxes)
    thetavec[j] = ParamEstimationSynthetic_ints(λ,prior_distr,logλ,boxes,Wplus; postsamples=300)
end




#### Compare 
mean(thetavals, dims=1)
θ₀
EstimateParamsPL_Logit(S,patterns[5],W)




###### Bös

size(thetavals)

logliks = Array{Float64}(undef,400)
Threads.@threads for i in 1:400
    println("i=$i")
    logliks[i] = logpdf(syntheticlikelihood_ints(thetavals[i,:],logλ,boxes,Wplus),λ)
end

plot([plot(logliks[1:100]),plot(thetavals[1:100,1]),plot(thetavals[1:100,2])]..., layout=(3,1))

plot(logliks[1:150])
logliks



0.2 < logpdf(prior_distr,[log(100),log(0.9)])





struct ReconstructedPointProcess{Counts<:Vector{Int64},Boxes<:Vector{Box{2, Float64}},P<:GibbsProcess} <: PointProcess
    counts :: Counts
    boxes :: Boxes
    p :: P
end

# Birth death move
function sample_pp2(rng::Random.AbstractRNG,
    prec::ReconstructedPointProcess,
    b::Box; niter=10_000, progress=false)

    # Number of points
    N(S) = length(S.items)

    # probabilities
    logλ(u,S) = N(S)>0 ? prec.p.logλ(u,S) : 0
    rmove(u,ξ,S₋) = exp(logλ(u,S₋)-logλ(ξ,S₋))/(N(S₋)+1)

    # Initial sample
    Svec = [prec.counts[i]>0 ? sample_pp(BinomialProcess(prec.counts[i]),prec.boxes[i]) : EmptyPointSet() for i in eachindex(prec.counts)]
    S = PointSet(union(Svec...))
    #println("Performing ",niter," steps")
    for m in 1:niter
        if progress
            if mod(m,100)==0
                println("step ",m)
            end
        end
        Svecprop = deepcopy(Svec)
        Sprop = deepcopy(S)
        # Randomly select box
        bind = rand(1:length(boxes))
        # region configuration
        lo, up = coordinates.(extrema(boxes[bind]))
        V = measure(boxes[bind])
        U = product_distribution([Uniform(lo[i], up[i]) for i in 1:embeddim(boxes[bind])])
        if  counts[bind]>0 #rand() < pmove
            # Randomly select and move point
            i = rand(1:counts[bind])
            ξ = Svecprop[bind].items[i]
            deleteat!(Svecprop[bind].items,i)
            Sprop = PointSet(union(Svecprop...))
            u = Point(rand(U))
            # accept/reject move
            if rand()<rmove(u,ξ,Sprop)
                Svec[bind] = Svecprop[bind] ∪ u
                S = PointSet(union(Svec...))
            end
        end
    end
    return S
end


jval = 3
counts = Nvectors[1,jval]
boxes = PointProcessLearning.partitions(W,resolutions[jval])
prec = ReconstructedPointProcess(counts,boxes,p)
#prec.p
#logλ2(u::Point,s::PointSet) = logλ(u,s,θ₀)
srec = sample_pp2(Random.GLOBAL_RNG,prec,W)
counts
[N(srec,box) for box in boxes]

plt1 = plot(patterns[1],W)
plt2 = plot(srec,W)
plot(plt1,plt2)


# For parallellization
#using Base.Threads
#nthreads()








function syntheticlikelihood_simple(θ::Vector, logλ::Function, boxes::Vector{Box{2, Float64}},W::Box; nsamples = 10, δ = 0.5)
    pp = GibbsProcess((u,s) -> logλ(u,s,θ))
    counts = Matrix{Float64}(undef,(length(boxes),nsamples))
    for n in 1:nsamples
        s = sample_pp(Random.GLOBAL_RNG,pp,W; niter=30_000)
        counts[:,n] = [N(s,box) for box in boxes]
    end
    μ = vec(mean(counts, dims=2))
    #Σ = cov(hcat([counts[:,k] for k in 1:nsamples]...)')
    Σ₀ = diagm(vec(var(counts, dims=2)))
    Σ₁ = Statistics.cov(counts')
    # correct if semidefinite
    Σ₁ = Symmetric(Σ₁) + max(eps(),-minimum(eigvals(Symmetric(Σ₁))))*I
    #alt1
    #Σ = Symmetric(Σ) + max(0,abs(minimum(eigvals(Symmetric(Σ)))))*I
    #alt2
    #lambda = minimum(filter(x->(x<0),eigvals(Σ)))
    #Σ = Σ + abs(lambda)*I
    
    Σ = δ.*Σ₁ .+ (1-δ).*Σ₀
    #eigvals(Σ)
    #return counts
    #return (μ,Σ)
    return MvNormal(μ,Σ)
end


function ParamEstimationSynthetic_simple(counts,prior,logλ,boxes,W; postsamples = 20)
    pr(θ) = logpdf(prior,θ)
    Q(θ) = product_distribution([Normal(θ[1],0.275),Normal(θ[2],0.1275)])
    q(θnew,θold) = logpdf(Q(θold),θnew)
    L(θ) = syntheticlikelihood_simple(θ,logλ,boxes,W)
    l(N,θ) = logpdf(L(θ),N)
    A(θnew,θold) = min(1,exp(l(counts,θnew)+pr(θnew)+q(θold,θnew) - l(counts,θold)-pr(θold)-q(θnew,θold)))
    # Initialize
    #θold = rand(prior)
    θold = θ₀
    θs = Array{Float64}(undef, (postsamples,length(θold)))
    θs[1,:] = θold
    for i in 2:postsamples
        println("i=$i")
        θold = θs[i-1,:]
        θnew = rand(Q(θold))
        θs[i,:] = (rand() < A(θnew,θold)) ? θnew : θold
        #while rand() < A(θnew,θold)
        #    θnew = rand(Q(θold))
        #end
        #θs[i,:] = θnew
    end
    return θs
end


jval = 3
counts = Nvectors[2,jval]
boxes = PointProcessLearning.partitions(W,resolutions[jval])

# Note: Try also with lambda = counts.*measure(boxes)
thetavals = ParamEstimationSynthetic_simple(counts,prior_distr,logλ,boxes,Wplus; postsamples=400)

scatter(thetavals[:,1],thetavals[:,2], mc=:blue, alpha=0.5)
scatter!([θ₀[1]],[θ₀[2]])

plot(thetavals[:,1])
hline!([ θ₀[1] ])

plot(thetavals[:,2])
hline!([ θ₀[2] ])