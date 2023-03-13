## Remember to run Pkg> activate .
using Meshes
using Plots
#import Random
using StatsBase
include("../src/PointProcessLearning.jl")
using .PointProcessLearning

# Set up window
W = Box((0.,0.),(1.,1.))
Wplus = Box((-0.1,-0.1),(1.1,1.1))


##### Exponential Gibbs Process

# Strauss
R = 0.05;
λ₀ = 1000; g₀ = 0.5;
θ₀ = [log(λ₀),log(g₀)]
S = [(u,s)->1, (u,s)->(NNdist(s ∪ u,R)-NNdist(s,R))]

# Generate patter
logλ(u,s,θ) = sum(θ[i]S[i](u,s) for i in eachindex(θ))
p = GibbsProcess((u,s) -> logλ(u,s,θ₀))
s = sample_pp(p,Wplus)
plot(s,W)


res = EstimateParamsPL_Logit(S,s,W)
exp.(coef(res))

# Do it many times
nruns = 20
θmat = zeros(nruns,2)
for i in 1:nruns
    s = sample_pp(p,Wplus)
    res = EstimateParamsPL_Logit(S,s,W)
    θmat[i,:] = exp.(coef(res))
end

θ̂ = reshape(mean(θmat, dims=1),2)
plt1 = scatter(1:nruns,θmat[:,1], label="estimates", title="β = exp(θ₁)")
hline!(plt1,[exp(θ₀[1])], labels="true value")
hline!(plt1, [θ̂[1]], label="est. mean")

plt2 = scatter(1:nruns,θmat[:,2], label="estimates", title="γ = exp(θ₂)")
hline!(plt2,[exp(θ₀[2])])
hline!(plt2, [θ̂[2]], label="est. mean")

plot(plt1,plt2)


# Gibbs Point Process (Geyer)
# Distance between points
R = 0.05; sat=5;
t(u::Point,s::PointSet) = sum(dist(u,x)<R for x in s.items)
# Parameters
λ₀ = 100; g₀ = 1.4;
θ₀ = [log(λ₀),log(g₀)]
S = [(u,s)->1, (u,s)->min(sat,t(u,s))]

# Generate patter
logλ(u,s,θ) = sum(θ[i]S[i](u,s) for i in eachindex(θ))
p = GibbsProcess((u,s) -> logλ(u,s,θ₀))
s = sample_pp(p,Wplus)
plot(s,W)


res = EstimateParamsPL_Logit(S,s,W)
exp.(coef(res))

# Do it many times
nruns = 20
θmat = zeros(nruns,2)
for i in 1:nruns
    s = sample_pp(p,Wplus)
    res = EstimateParamsPL_Logit(S,s,W)
    θmat[i,:] = exp.(coef(res))
end

θ̂ = reshape(mean(θmat, dims=1),2)
plt1 = scatter(1:nruns,θmat[:,1], label="estimates", title="β = exp(θ₁)")
hline!(plt1,[exp(θ₀[1])], labels="true value")
hline!(plt1, [θ̂[1]], label="est. mean")

plt2 = scatter(1:nruns,θmat[:,2], label="estimates", title="γ = exp(θ₂)")
hline!(plt2,[exp(θ₀[2])])
hline!(plt2, [θ̂[2]], label="est. mean")

plot(plt1,plt2)












########

function logλtest(u::Point,s::PointSet,θ::Vector)
    return θ[1] + θ[2]*coordinates(u)[1]
end

# Poisson model
poissonmodel = model_pp([2.5,3.5],[-Inf,-Inf],[Inf,Inf],logλtest,true)
p1 = GibbsProcess((u,s) -> poissonmodel.logλ(u,s,poissonmodel.θ))
s1 = sample_pp(p1,W)
plot(s1,W)

# Generate differently
p2 = PoissonProcess(u->exp(logλtest(u,EmptyPointSet(),[2.5,3.5])))
s2 = sample_pp(p2,W)
plot(s2,W)


####################
# Simple test
sT, sV = TVsplit(s2,0.8)
plot(sT,W,mc=:black)
plot!(sV,W,mc=:red)
PredictionError(sT,sV,(u,s)->logλtest(u,s,poissonmodel.θ),W;p=0.3,independent=true)
PredictionErrorPlot(sT,sV,(u,s)->logλtest(u,s,poissonmodel.θ),W,4;p=0.3,independent=true)

##############

# Using point process learning
results = zeros(30,2)
for i in 1:30
    results[i,:] = EstimateParams(poissonmodel, s2, W; p=0.5,k=20)
end
scatter(results)
hline!([mean(results[:,1]),mean(results[:,2])])

# Using pseudolikelihood
EstimateParams(poissonmodel, s2, W)

