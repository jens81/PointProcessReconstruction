## Remember to run Pkg> activate .
using Pkg;
Pkg.activate(".")
using Meshes, Plots, StatsBase, GLM, Combinatorics, Distributions
using Statistics, LinearAlgebra
using BSON
import Random
include("../src/PointProcessLearning.jl")
using .PointProcessLearning

#########################################
# Set up window
#########################################
W = Box((0.,0.),(1.,1.))
Wplus = enlarge(W,0.1)


#########################################
# Set up Geyer process
#########################################
# Parameters
# Distance between points
R = 0.05; sat=4;
t(u::Point,s::PointSet) = sum(dist(u,x)<R for x in s.items)
# Parameters
λ₀ = 100; g₀ = 1.25;
θ₀ = [log(λ₀),log(g₀)]
S = [(u,s)->1, (u,s)->min(sat,t(u,s))]
# Generate patter
logλ(u,s,θ) = sum(θ[i]*S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
p = GibbsProcess(logλ)
s = sample_pp(p,Wplus)

#########################################
# Generate original, and reconstructed patterns
#########################################
# Resolutions to be used in reconstruction
resolutions = [1,2,4,8,16,32,64]
noriginals = 50
nresolutions = length(resolutions)
# the patterns
patterns = [sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=30_000) for _ in 1:noriginals]
bson("patterns.bson", Dict(:patterns => patterns))
# countdata
Nvectors = Array{Vector{Int64}}(undef, (noriginals,nresolutions))
for i in 1:noriginals, j in 1:nresolutions
    s = patterns[i]
    boxes = PointProcessLearning.partitions(W,resolutions[j])
    Nvectors[i,j] = [N(s,box) for box in boxes]
end

# 
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
#
function syntheticlikelihood_safe(θ::Vector, logλ::Function, boxes::Vector{Box{2, Float64}},W::Box; nsamples = 10)
    try 
        return syntheticlikelihood_ints(θ,logλ,boxes,W; nsamples=nsamples)
    catch e
        println(e)
        println("trying again")
        return syntheticlikelihood_safe(θ,logλ,boxes,W; nsamples=nsamples)
    end
end
#
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

prior_distr = product_distribution([Normal(log(100),1),truncated(Normal(log(1),0.5),0,nothing)])
thetavec = Matrix{Matrix{Float64}}(undef,(10,nresolutions))
Threads.@threads for i in 1:1
    Threads.@threads for j in 1:nresolutions
        counts = Nvectors[i,j]
        boxes = PointProcessLearning.partitions(W,resolutions[j])
        λ = counts.*measure.(boxes)
        thetavec[i,j] = ParamEstimationSynthetic_ints(λ,prior_distr,logλ,boxes,Wplus; postsamples=200)
    end
end
bson("thetas_ABC.bson", Dict(:thetavec => thetavec[1,:]))

plts = [plot() for _ in 1:nresolutions]
means = Vector{Vector}(undef,nresolutions)
for j in 1:nresolutions
    scatter!(plts[j],thetavec[1,j][:,1],thetavec[1,j][:,2])
    means[j] = [mean(thetavec[1,j][:,1]),mean(thetavec[1,j][:,2])]
end
plot(plts...)
means
plt1 = plot(resolutions,map(x->x[1],means))
plt2 = plot(resolutions,map(x->x[2],means))
plot(plt1,plt2)

plt1 = plot([1/res for res in resolutions],map(x->x[1],means))
plt2 = plot([1/res for res in resolutions],map(x->x[2],means))
hline!(plt1,[θ₀[1]])
hline!(plt2,[θ₀[2]])
plot(plt1,plt2)
