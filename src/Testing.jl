## Remember to run Pkg> activate .
using Meshes
using Plots
#import Random
using StatsBase
include("PointProcessLearning.jl")
using .PointProcessLearning

# Set up window
W = Box((0.,0.),(1.,1.))
Wextra = Box((-0.5,-0.5),(1.5,1.5))

3+4

#########################################
###### PointProcessLearning Stuff



# With model specification
t(u::Point,s::PointSet,R::Real) = sum(dist(u,x)<R for x in s.items)
logλgeyer(u,s,θ) = log(θ[1]) + log(θ[2])*min(θ[4],t(u,s,θ[3]))
geyermodel = model_pp([100,1.5,0.05,12],[0,1,0,1],[Inf,Inf,Inf,Inf],logλgeyer,false)

pgeyer = GibbsProcess((u,s) -> geyermodel.logλ(u,s,geyermodel.θ))
sgeyer = sample_pp(pgeyer,W)
plot(sgeyer,W)
CondIntPlot((u,s)->geyermodel.logλ(u,s,geyermodel.θ),sgeyer,W)

#Ur(sgeyer,0.05,W)


function PredictionErrorDep2(sT::PointSet,sV::PointSet,logλ::Function,b::Box;p=0.5)
    #println("Running dependent")
    λ(u,s) = exp(logλ(u,s))
    h(u,s) = (1/(1-p))*(1/λ(u,s))
    #h(u,s) = (p)*(1/λ(u,s))
    #score = sum( ( x∈b ? h(x,sT) : 0) for x in sV.items) -measure(b)
    score = sum( ( x∈b ? h(x, (sT ∪ sV) \ PointSet([x])) : 0) for x in sV.items) -measure(b)
    return score
end
################ Important !!!! 
#### rerun several times for different samples, should average to 0

ps = repeat([0.2,0.4,0.6,0.8], inner=25) 
k=20;
m = geyermodel
Nsamples = 100
pe = zeros(k,Nsamples)
st = zeros(Nsamples)
for n in eachindex(ps)
    p = ps[n]
    sgeyer = sample_pp(pgeyer,W)
    CVfolds = [TVsplit(sgeyer,p) for i in 1:k]
    for i in 1:k
        (sT,sV) = CVfolds[i]
        pe[i,n] = PredictionError(sT,sV,(u,s)->m.logλ(u,s,m.θ),W;p=p)
        #pe[i,n] = PredictionError(sT,sV,(u,s)->m.logλ(u,s,m.θ),W;p=p,independent=false)
        st[n] = StoyanGrabarnik(sgeyer,(u,s)->m.logλ(u,s,m.θ),W)
    end
end
scatter(mean(pe,dims=1)',label="mean over CV for 100 samples")
hline!([mean(pe)], label="mean over realizations")
#### Compare stoyan StoyanGrabarnik
scatter!(st,label="S-G over 100 samples")
hline!([mean(st)],label="S-G")
### Make predictionerrorplot
TVset = TVsplit(sgeyer,0.8)
PredictionErrorPlot(TVset[1],TVset[2],(u,s)->m.logλ(u,s,m.θ),W,4)



# Gibbs Point Process (Strauss)
λ₀ = 200
γ = 0.8
R = 0.1
logλ4(u::Point,s::PointSet) = log(λ₀) + log(γ)*(NNdist(s ∪ u,R) - NNdist(s,R))
p4 = GibbsProcess(logλ4)
s4 = sample_pp(p4,W)
plot(s4,W)

ps = repeat([0.2,0.4,0.6,0.8], inner=25) 
k=20;
Nsamples = 100
pe = zeros(k,Nsamples)
st = zeros(Nsamples)
for n in eachindex(ps)
    p = ps[n]
    s4 = sample_pp(p4,W)
    CVfolds = [TVsplit(s4,p) for i in 1:k]
    for i in 1:k
        (sT,sV) = CVfolds[i]
        pe[i,n] = PredictionError(sT,sV,logλ4,W;p=p)
        #pe[i,n] = PredictionError(sT,sV,(u,s)->m.logλ(u,s,m.θ),W;p=p,independent=false)
        st[n] = StoyanGrabarnik(s4,logλ4,W)
    end
end
scatter(mean(pe,dims=1)',label="mean over CV for 100 samples")
hline!([mean(pe)], label="mean over realizations")
#### Compare stoyan StoyanGrabarnik
scatter!(st,label="S-G over 100 samples")
hline!([mean(st)],label="S-G")
TVset = TVsplit(s4,0.8)
PredictionErrorPlot(TVset[1],TVset[2],logλ4,W,4)




########
## Testing Optimizers
#using Optim
#CVfolds = [TVsplit(sgeyer,0.5) for i in 1:10]
# Loss function
#m = geyermodel
#θ₀ = m.θ
#L(θ) = Loss(CVfolds, (u,s)->m.logλ(u,s,θ),W,0.5; independent=m.independent)
# Find minimum
#L(θ₀)

# res = optimize(L, m.lo, m.up, m.θ, Fminbox(NelderMead()),
#         Optim.Options(x_tol = 1e-4,
#         iterations = 1000,
#         store_trace = false,
#         show_trace = true,
#         time_limit = 15))
# Optim.converged(res)
# Optim.minimizer(res)



#############
## Automatic optimizers
res = EstimateParams(geyermodel, sgeyer, W; p=0.8,k=50)
resPL = EstimateParamsPL(geyermodel, sgeyer, W)

results = zeros(30,4)
for i in 1:30
    results[i,:] = EstimateParams(geyermodel, sgeyer, W; p=0.5,k=20)
end
PLresults = EstimateParamsPL(geyermodel, sgeyer, W)
θlabels = ["λ₀","γ","R","sat"]
plts = [scatter(results[:,j],label=θlabels[j]) for j in 1:4]
for j in 1:4
    hline!(plts[j],[mean(results[:,j]),PLresults[j]],labels=(θlabels[j]*"mean","PL"))
end
plot(plts...)


hline!(PLresults)


struct model_pp_scaled{T<:Vector,logL<:Function,ind<:Bool}
    θ :: T
    logλ :: logL
    independent :: ind
end



function StraussModel(λ₀,γ,R)
    η = Vector([λ₀,γ,R])
    f = [x->log(x), x->log(x/(1-x)),x->log(x)]
    finv = [x->exp(x), x->1/(1+exp(-x)), x->exp(x)]
    θ = Vector([f[i](η[i]) for i in eachindex(η)])
    #θ = Vector([log(λ₀),log(γ/(1-γ)),log(R)])
    logλ(u::Point,s::PointSet,θ::Vector) = θ[1] + log(1/(1+exp(-θ[2])))*(NNdist(s ∪ u,exp(θ[3])) - NNdist(s,exp(θ[3])))
    independent = false
    return model_pp_scaled(θ,logλ,independent)
end

# Gibbs Point Process (Strauss)
λ₀ = 200
γ = 0.6
R = 0.1
straussmodel = StraussModel(λ₀,γ,R)
pstrauss = GibbsProcess((u,s) -> straussmodel.logλ(u,s,straussmodel.θ))
sstrauss = sample_pp(pstrauss,W)
plot(sstrauss,W)


#### Berman Turner

function partitions(W::Box,n::Int)
    lo, up = coordinates.(extrema(W))
    dx = (up[1]-lo[1])/n
    dy = (up[2]-lo[2])/n
    boxmat = [Box((lo[1]+k*dx,lo[2]+j*dy),(lo[1]+(k+1)*dx,lo[2]+(j+1)*dy)) for k=0:(n-1), j=0:(n-1)]
    return reshape(boxmat,n*n)
end


function BermanTurner(s::PointSet, f::Function , b::Box; nbox=10)
    boxes = partitions(b,nbox)
    #areas = zeros(nbox) # improve by just computing directly:  measure(b)/(nbox^2)
    boxarea = measure(b)/(nbox*nbox)
    S = PointSet(filter(x->x∈b, s.items))
    Ns = length(S.items)
    #BoxPoint = zeros(nbox*nbox, Ns + nbox*nbox)
    intval = 0.0
    for (i,box) in enumerate(boxes)
        lo, up = coordinates.(extrema(box))
        S = S ∪ Point((up+lo)./2)
        # now take all points in the box and compute their contribution to intval
        pointsinbox = filter(x -> x∈box,S.items)
        n = length(pointsinbox)
        w = boxarea/n
        intval = intval + w*sum(f(u,s) for u in pointsinbox)
    end
    return intval
end

plot(stest,W)
BT = BermanTurner(stest,logλ,W)

function BermanTurnerPL(s::PointSet,logλ::Function,b::Box;nbox=10)
    λ = (u,s) -> logλ(u,s)
    return sum(logλ(x,s\PointSet(x)) for x in s.items) - BermanTurner(s,λ,b; nbox=nbox)
    #return BermanTurner(s,λ,b; nbox=nbox)
end

BT_PL = BermanTurnerPL(stest,logλ,W)

using Optim
function EstimateParamsPL2(m::model_pp, s::PointSet, b::Box)
    loglik(θ::Vector) = BermanTurnerPL(s,(u,s)->m.logλ(u,s,θ),W)
    res = optimize(θ->-loglik(θ), m.lo, m.up, m.θ, Fminbox(NelderMead()),
            Optim.Options(x_tol = 1e-8,
            iterations = 1000,
            store_trace = false,
            show_trace = true,
            time_limit = 15))
    if Optim.converged(res)==true
        return Optim.minimizer(res)
    else
        return println("Optimization did not converge.")
    end
end

# With model specification
t(u::Point,s::PointSet,R::Real) = sum(dist(u,x)<R for x in s.items)
R = 0.05; sat = 5;
logλgeyer(u,s,θ) = log(θ[1]) + log(θ[2])*min(sat,t(u,s,R))
geyermodel = model_pp([100,1.3],[20,1],[500,2],logλgeyer,false)
# Produce pattern
pgeyer = GibbsProcess((u,s) -> geyermodel.logλ(u,s,geyermodel.θ))
Wplus = Box((-.5,-.5),(1.5,1.5))
sgeyer = sample_pp(pgeyer,Wplus)
plot(sgeyer,W)
# Estimate params
est = EstimateParamsPL2(geyermodel,sgeyer,W)

# With model specification
R = 0.05;
logλstrauss(u,s,θ) = log(θ[1]) + log(θ[2])*(NNdist(s ∪ u,R) - NNdist(s,R))
straussmodel = model_pp([100,0.8],[20,0.2],[500,1.0],logλstrauss,false)
# Produce pattern
pstrauss = GibbsProcess((u,s) -> straussmodel.logλ(u,s,straussmodel.θ))
Wplus = Box((-.5,-.5),(1.5,1.5))
sstrauss = sample_pp(pstrauss,Wplus)
plot(sstrauss,W)
# Estimate params
est = EstimateParamsPL2(straussmodel,sstrauss,W)
straussmodel.θ
3+5


##### Exponential Gibbs Process

R = 0.05;
λ₀ = 1000; g₀ = 0.5;
θ₀ = [log(λ₀),log(g₀)]
S = [(u,s)->1, (u,s)->(NNdist(s ∪ u,R)-NNdist(s,R))]

# Generate patter
logλ(u,s,θ) = sum(θ[i]S[i](u,s) for i in eachindex(θ))
p = GibbsProcess((u,s) -> logλ(u,s,θ₀))
s = sample_pp(p,Wplus)
plot(s,W)

using GLM

# Requires 100+ for nd to work properly
function EstimateParamsPl2(S::Vector{Function}, s::PointSet, W::Box; nd = 80)
    boxes = partitions(W,nd)
    u = PointSet([centroid(box) for box in boxes])
    sw = PointSet(filter(x->x∈W,s.items))
    v = u ∪ sw
    Nv = length(v.items)
    K = length(S)
    boxind = [findfirst(box -> in(x,box),boxes) for x in v.items] 
    a = measure(W)/(nd*nd)
    # Now define
    w = zeros(Nv)
    y = zeros(Nv)
    X = zeros(Nv,K)
    for i in 1:Nv
        box = boxes[boxind[i]]
        vi = v.items[i]
        w[i] = a / (N(v,box))
        y[i] = (vi ∈ sw.items) ? 1/w[i] : 0
        for j in 1:K
            X[i,j] = S[j](vi,s\PointSet(vi))
            #X[i,j] = S[j](vi,s)
        end
    end
    return (y,X,w)
end
(y,X,w) = EstimateParamsPl2(S,s,W; nd=150)
res = glm(X,y,Poisson(),LogLink(); wts=w)
θest = coef(res)
exp.(θest)



# Ok. Now we do the logistic one !!!!
function EstimateParamsPl3(S::Vector{Function}, s::PointSet, W::Box; nd = 80)
    boxes = partitions(W,nd)
    u = PointSet([centroid(box) for box in boxes])
    sw = PointSet(filter(x->x∈W,s.items))
    v = u ∪ sw
    Nv = length(v.items)
    K = length(S)
    boxind = [findfirst(box -> in(x,box),boxes) for x in v.items] 
    r = (nd*nd)/measure(W)
    ofs = fill(log(1/r),Nv)
    # Now define
    #w = zeros(Nv)
    y = zeros(Nv)
    X = zeros(Nv,K)
    for i in 1:Nv
        box = boxes[boxind[i]]
        vi = v.items[i]
        #w[i] = a / (N(v,box))
        y[i] = (vi ∈ sw.items) ? 1 : 0
        for j in 1:K
            X[i,j] = S[j](vi,s\PointSet(vi))
            #X[i,j] = S[j](vi,s)
        end
    end
    return (y,X,ofs)
end
(y,X,offset) = EstimateParamsPl3(S,s,W; nd=10)
res = glm(X,y,Bernoulli(),LogitLink(); offset=offset)
exp.(coef(res))


# Running models
struct model_pp{T<:Vector,Lo<:Vector,Up<:Vector,logL<:Function,ind<:Bool}
    θ :: T
    lo :: Lo
    up :: Up
    logλ :: logL
    independent :: ind
end








# Create subtype GibbsProcess to PointProcess
# Introduce GibbsProcesBare or something for the struct now called GibbsProcess
# Make ExponentialGibbsModel subtype of GibbsProcess
mutable struct ExponentialGibbsModel3{T<:Vector,S<:Vector{Function},logL<:Function}
    θ :: T
    S :: S
    logλ :: logL # remove this later and introduce a method logλ for this object instead
end 

function ExponentialGibbsModel(θ::Vector, S::Vector{Function})
    logλ = (u,s) -> sum(θ[i]*S[i](u,s) for i in eachindex(θ))
    return ExponentialGibbsModel3(θ,S,logλ)
end 

m = ExponentialGibbsModel(θ₀,S)
m.θ
m.S
m.logλ







# Define function to estimate

l(θ::Vector) = sum(sum(θ[i]) for x in s.items) - BermanTurner(s,(u,s)->exp(logλ(u,s,θ)),W)
l(θ₀)
gradl(θ::Vector) = "blah"



# Estimate params
est = EstimateParamsPL2(straussmodel,sstrauss,W)
straussmodel.θ
























\(s1::PointSet,s2::PointSet) = PointSet(setdiff(s1.items,s2.items))
function BermanTurnerPL(s::PointSet,logλ::Function,b::Box;nbox=10)
    boxes = partitions(b,nbox)
    as = [measure(box) for box in boxes]
    us = EmptyPointSet()
    for box in boxes
        lo, up = coordinates.(extrema(box))
        us = us ∪ Point((up+lo)./2)
    end
    # filter to only include points in b
    sw = PointSet(filter(p->p∈b,s.items))
    vs = sw ∪ us
    vsbox = [findfirst(box-> in(v,box),boxes) for v in vs.items]
    w = [as[vsbox[i]]/N(vs,boxes[vsbox[i]]) for i in eachindex(vs.items)]
    y = [ (vs.items[i] ∈ s.items) ? 1/w[i] : 0 for i in eachindex(vs.items)]
    return sum(w[i]*(y[i]*logλ(vs.items[i],s\PointSet(vs.items[i])) - exp(logλ(vs.items[i],s))) for i in eachindex(vs.items))
end

sstraussplus = BermanTurnerPL(sstrauss,(u,s)->straussmodel.logλ(u,s,straussmodel.θ),W)
θ₀ = Vector(rand(3))
sstraussplus = BermanTurnerPL(sstrauss,(u,s)->straussmodel.logλ(u,s,θ₀),W)
straussmodel.θ
θ₀ = Vector([4.5,1.6,-2.4])
sstraussplus = BermanTurnerPL(sstrauss,(u,s)->straussmodel.logλ(u,s,θ₀),W)


sum(sstraussplus)
sum(y!=0 for y in sstraussplus)

function EstimateParamsPL2(m::model_pp_scaled, s::PointSet, b::Box)
    lo, up = coordinates.(extrema(b))
    #println("Computing integral over cond. int.")
    loglik(θ::Vector) = BermanTurnerPL(s,(u,s)->m.logλ(u,s,θ),b)
    println("Trying some small deviation from true")
    θ₀ = m.θ + Vector(rand(3).-0.5)
    println(θ₀)
    res = optimize(θ->-loglik(θ), θ₀,SimulatedAnnealing())
    if Optim.converged(res)==true
        return Optim.minimizer(res)
    else
        return println("Optimization did not converge. Last value:", Optim.minimizer(res))
    end
end

using Cubature
using Optim
resPL = EstimateParamsPL2(straussmodel, sstrauss, W)
zxexp(resPL[1]) # λ₍
1/(1+exp(-resPL[2])) # γ
exp(resPL[3]) # R
resPL
straussmodel.θ


# Gridsearch...
function GridSearch(m::model_pp,s::PointSet,b; N=10)
    loglik(θ::Vector) = BermanTurnerPL(s,(u,s)->m.logλ(u,s,θ),b)
    extr = [(100,800), (0.75,1)]
    #r = range(0, 1, length = n)

    iter = Iterators.product((range(e[1],e[2],length=N) for e in extr)...)
    loglikmax = 0
    θmax = Vector([e[1] for e in extr])
    for i in iter
        θ = Vector(collect(i))
        logliknow = loglik(θ)
        if logliknow>loglikmax
            loglikmax = logliknow
            θmax = θ
        end
    end
    return (θmax,loglikmax)
    #thetas = vec([collect(i) for i in iter])
    #logliks = [loglik(θ) for θ in thetas]
    #return (thetas[argmax(logliks)], maximum(logliks))
end

R = 0.05
logλ4(u::Point,s::PointSet,θ::Vector) = log(θ[1]) + log(θ[2])*(NNdist(s ∪ u,R) - NNdist(s,R))
straussmodel = model_pp([250,0.8],[0,1],[Inf,Inf],logλ4,false)
pstrauss = GibbsProcess((u,s) -> straussmodel.logλ(u,s,straussmodel.θ))
#sstrauss = sample_pp(pstrauss,W)
import Random
Wextra = Box((-0.5,-0.5),(1.5,1.5))
sstrauss = sample_pp(Random.GLOBAL_RNG, pstrauss, Wextra; niter=100_000)
plot(sstrauss,Wextra)
#GridSearch(straussmodel,sstrauss,W)

sat = 8; R = 0.2
# Distance between points
#dist(x::Point,y::Point) = sqrt((coordinates(x)[1]-coordinates(y)[1])^2+(coordinates(x)[2]-coordinates(y)[2])^2)
t(u::Point,s::PointSet) = sum(dist(u,x)<R for x in s.items)
logλ5(u::Point,s::PointSet,θ::Vector) = log(θ[1]) + log(θ[2])*min(sat,t(u,s))
straussmodel = model_pp([50,1.2],[0,1],[Inf,Inf],logλ5,false)
pstrauss = GibbsProcess((u,s) -> straussmodel.logλ(u,s,straussmodel.θ))
#sstrauss = sample_pp(pstrauss,W)
import Random
Wextra = Box((-0.5,-0.5),(1.5,1.5))
sstrauss = sample_pp(Random.GLOBAL_RNG, pstrauss, Wextra; niter=10_000)
plot(sstrauss,Wextra)


logλ6(u::Point,s::PointSet,θ::Vector) = θ[1]+θ[2]*coordinates(u)[1] 
straussmodel = model_pp([5,1.2],[0,1],[Inf,Inf],logλ6,false)
pstrauss = GibbsProcess((u,s) -> straussmodel.logλ(u,s,straussmodel.θ))
#sstrauss = sample_pp(pstrauss,W)
import Random
Wextra = Box((-0.5,-0.5),(1.5,1.5))
sstrauss = sample_pp(Random.GLOBAL_RNG, pstrauss, Wextra; niter=10_000)
plot(sstrauss,Wextra)


function GridPlot(m::model_pp,s::PointSet,b; N=10)
    loglik(θ::Vector) = BermanTurnerPL(s,(u,s)->m.logλ(u,s,θ),b)
    extr = [(50,400), (0.5,0.95)]
    θ1 = range(extr[1][1], extr[1][2], length = N)
    θ2 = range(extr[2][1], extr[2][2], length = N)
    #iter = Iterators.product((range(e[1],e[2],length=N) for e in extr)...)
    #loglikmax = 0
    #θmax = Vector([e[1] for e in extr])
    logPL = zeros(N,N)
    for i in 1:N
        for j in 1:N
            θ = Vector([θ1[i],θ2[j]])
            logPL[i,j] = loglik(θ)
        end
    end
    return (θ1,θ2,logPL)
    #thetas = vec([collect(i) for i in iter])
    #logliks = [loglik(θ) for θ in thetas]
    #return (thetas[argmax(logliks)], maximum(logliks))
end
(θ1,θ2,logPL) = GridPlot(straussmodel,sstrauss,W;N=20)
heatmap(θ1,θ2,logPL')
scatter!([straussmodel.θ[1]],[straussmodel.θ[2]], label="true")
GridSearch(straussmodel,sstrauss,W;N=20)

using Cubature
function GridPlot3(m::model_pp,s::PointSet,b; N=10)
    #loglik(θ::Vector) = BermanTurnerPL(s,(u,s)->m.logλ(u,s,θ),b)
    lo, up = coordinates.(extrema(b))
    function integral(θ::Vector)
        (val,err) = hcubature(u->exp(m.logλ(Point(u),s,θ)), lo, up;
                      reltol=1e-8, abstol=1e-12, maxevals=100_000)
        return val
    end
    loglik(θ::Vector) = sum(m.logλ(x,s,θ) for x in s.items) - integral(θ)    
    extr = [(10,1500), (0.05,0.95)]
    θ1 = range(extr[1][1], extr[1][2], length = N)
    θ2 = range(extr[2][1], extr[2][2], length = N)
    #iter = Iterators.product((range(e[1],e[2],length=N) for e in extr)...)
    #loglikmax = 0
    #θmax = Vector([e[1] for e in extr])
    logPL = zeros(N,N)
    for i in 1:N
        for j in 1:N
            θ = Vector([θ1[i],θ2[j]])
            logPL[i,j] = loglik(θ)
        end
    end
    return (θ1,θ2,logPL)
    #thetas = vec([collect(i) for i in iter])
    #logliks = [loglik(θ) for θ in thetas]
    #return (thetas[argmax(logliks)], maximum(logliks))
end
(θ1,θ2,logPL) = GridPlot3(straussmodel,sstrauss,W)
heatmap(θ1,θ2,logPL')





function GridPlot2(m::model_pp,s::PointSet,b; N=10)
    k = 20; p=0.75;
    # Create CV folds
    CVfolds = [TVsplit(s,p) for i in 1:k]
    # Loss function
    L(θ::Vector) = Loss(CVfolds, (u,s)->m.logλ(u,s,θ),b,p; independent=m.independent)
    extr = [(20,400), (0.1,0.9)]
    θ1 = range(extr[1][1], extr[1][2], length = N)
    θ2 = range(extr[2][1], extr[2][2], length = N)
    #iter = Iterators.product((range(e[1],e[2],length=N) for e in extr)...)
    #loglikmax = 0
    #θmax = Vector([e[1] for e in extr])
    logPL = zeros(N,N)
    for i in 1:N
        for j in 1:N
            θ = Vector([θ1[i],θ2[j]])
            logPL[i,j] = L(θ)
        end
    end
    return (θ1,θ2,logPL)
    #thetas = vec([collect(i) for i in iter])
    #logliks = [loglik(θ) for θ in thetas]
    #return (thetas[argmax(logliks)], maximum(logliks))
end
(θ1,θ2,logPL) = GridPlot2(straussmodel,sstrauss,W)
heatmap(θ1,θ2,logPL')




####
# Use BermanTurner Device smarter : For PL it should lead to Poissonregression
# Can we do something similar for PPL ?
# ALSO:::: What kind of function h(u,Z) to choose? can't use 1/pλₓ(u,X) since not function of Z.
# Could use 1/pE[λₓ(u,X)|Z] = 1/λ_Z(u,Z), but don't know λ_Z.
# Suppose we use 1/pλₓ(u,Z). Then we have
# sum( 1/(p*λₓ(y,Z)) for y in Y) - ∫ λₓ(u,X)/λₓ(u,Z) du








######## Poisson

function logλtest(u::Point,s::PointSet,θ::Vector)
    return θ[1] + θ[2]*coordinates(u)[1]
end

poissonmodel = model_pp([2.5,3.5],[-Inf,-Inf],[Inf,Inf],logλtest,true)
p2 = GibbsProcess((u,s) -> poissonmodel.logλ(u,s,poissonmodel.θ))
s2 = sample_pp(p2,W)
plot(s2,W)

# Generate differently
p3 = PoissonProcess(u->exp(logλtest(u,EmptyPointSet(),[2.5,3.5])))
s3 = sample_pp(p3,W)
plot(s3,W)

################ Important !!!! 
#### rerun several times for different samples, should average to 0
s2 = sample_pp(p2,W)
p = 0.5
CVfolds = [TVsplit(s2,p) for i in 1:20]
pe = zeros(20)
for i in 1:20
    (sT,sV) = CVfolds[i]
    pe[i] = PredictionError(sT,sV,(u,s)->poissonmodel.logλ(u,s,poissonmodel.θ),W;p=p,independent=poissonmodel.independent)
end
scatter(pe)
#### Compare stoyan StoyanGrabarnik
hline!([StoyanGrabarnik(s2,(u,s)->poissonmodel.logλ(u,s,poissonmodel.θ),W)])


#EstimateParams(poissonmodel, s2, W; p=0.5,k=20)
#EstimateParamsPL(poissonmodel, s2, W)


### Trying to find parameters
results = zeros(30,2)
for i in 1:30
    results[i,:] = EstimateParams(poissonmodel, s2, W; p=0.5,k=20)
end
scatter(results)
hline!([mean(results[:,1]),mean(results[:,2])])
PLresults = EstimateParamsPL(poissonmodel, s2, W)
hline!(PLresults)

