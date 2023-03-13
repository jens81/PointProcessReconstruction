## Remember to run Pkg> activate .
using Meshes
using Plots
#import Random
using StatsBase
include("../src/PointProcessLearning.jl")
using .PointProcessLearning

# Set up window
W = Box((0.,0.),(1.,1.))


# Gibbs Point Process (Inhomogeneous Geyer)
λ₆(u) = exp(3.5+2.5*coordinates(u)[1])
γ = 2.5
R = 0.10
sat = 15
t(u::Point,s::PointSet) = sum(dist(u,x)<R for x in s.items)
logλ6(u::Point,s::PointSet) = log(λ₆(u)) + log(γ)^min(sat,t(u,s))
p6 = GibbsProcess(logλ6)
s6 = sample_pp(p6,W)
plot(s6,W,axis=false,ticks=false)

# Set resolution of partition
resolution = 5

# Some diagnostics
StoyanGrabarnik(s6,logλ6,W)
StoyanGrabarnikPlot(s6,logλ6,W,resolution)





######## Comparing diagnostics

\(s1::PointSet,s2::PointSet) = PointSet(setdiff(s1.items,s2.items))
function PredictionErrorCV(w::Function,sT::PointSet,sV::PointSet,logλ::Function,b::Box;p=0.5)
    #println("Running dependent")
    λ(u,s) = exp(logλ(u,s))
    h(u,s) = (1/w(p))*(1/λ(u,s))
    #h(u,s) = (1/p)*(1/λ(u,s))
    #score = sum( ( x∈b ? h(x,sT) : 0) for x in sV.items) -measure(b)
    score = sum( ( x∈b ? h(x, sT) : 0) for x in sV.items) -measure(b)
    return score
end



# Strauss model
λ₀ = 400
γ = 0.5
R = 0.075
logλ(u::Point,s::PointSet) = log(λ₀) + log(γ)*(NNdist(s ∪ u,R) - NNdist(s,R))
pprocess = GibbsProcess(logλ)
stest = sample_pp(pprocess,W)
plot(stest,W,axis=false,ticks=false)

# Geyer model
λ₀ = 100
γ = 1.5
R = 0.05
sat = 5
t(u::Point,s::PointSet) = sum(dist(u,x)<R for x in s.items)
logλ(u::Point,s::PointSet) = log(λ₀) + log(γ)*min(sat,t(u,s))
pprocess = GibbsProcess(logλ)
stest = sample_pp(pprocess,W)
plot(stest,W,axis=false,ticks=false)


# Number of realizations
npat = 100 # up to 100
pvals = [0.1, 0.3, 0.5, 0.7, 0.9]
w = [p->(1-p), p->p, p->p/(1-p), p->(1-p)/p]
#scoremat = zeros(npat,length(pvals), 2 + length(w))
scoremat = zeros(npat,length(pvals), 2+length(w))
for pind in eachindex(pvals)
    p = pvals[pind]
    for pat in 1:npat
        stest = sample_pp(pprocess,W)
        (st,sv) = TVsplit(stest,p)
        scoremat[pat,pind,1] = StoyanGrabarnik(stest,logλ,W)
        scoremat[pat,pind,2] = PredictionError(st,sv,logλ,W; p=p)
        for wind in eachindex(w)
            scoremat[pat,pind,2+wind] = PredictionErrorCV(w[wind],st,sv,logλ,W; p=p)
        end    
    end
end

titlelist = ["full", "sub", "cv: w(p)=1-p", "cv: w(p)=p", "cv: w(p)=p/(1-p)", "cv: w(p)=(1-p)/p"]
yext1 = max(abs(maximum(scoremat[:,:,1:3])), abs(minimum(scoremat[:,:,1:3])))
yext2 = max(abs(maximum(scoremat[:,:,4:6])), abs(minimum(scoremat[:,:,4:6])))
yext = [(t>3) ? yext2 : yext1 for t in eachindex(titlelist)]
jtr = 0.075
plt = [scatter(ylims=(-yext[t],yext[t]), xlabel="p", title=titlelist[t], titlefontsize=11) for t in eachindex(titlelist)]
for t in eachindex(titlelist)
    for i in eachindex(pvals)
        scatter!(plt[t],pvals[i] .+ jtr.*(rand(npat).-0.5),scoremat[:,i,t], 
            labels=false,
            marker=(2,:gray,0.3,stroke(0)))
    end
    plot!(plt[t], pvals, reshape(mean(scoremat[:,:,t], dims=1),length(pvals)), 
        yerror=reshape(std(scoremat[:,:,t],dims=1),length(pvals)), 
        markershape=:circle, color=:black, linewidth=2, label=false)
    hline!(plt[t],[0], linestyle=:dash, color=:gray, label=false)
end
plot(plt...)
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/prederrors_singlesplit_strongGeyer.pdf")






### Using average over K folds

# Strauss model
λ₀ = 400
γ = 0.5
R = 0.075
logλ(u::Point,s::PointSet) = log(λ₀) + log(γ)*(NNdist(s ∪ u,R) - NNdist(s,R))
pprocess = GibbsProcess(logλ)
stest = sample_pp(pprocess,W)
plot(stest,W,axis=false,ticks=false)

# Geyer model
λ₀ = 100
γ = 1.5
R = 0.05
sat = 5
t(u::Point,s::PointSet) = sum(dist(u,x)<R for x in s.items)
logλ(u::Point,s::PointSet) = log(λ₀) + log(γ)*min(sat,t(u,s))
pprocess = GibbsProcess(logλ)
stest = sample_pp(pprocess,W)
plot(stest,W,axis=false,ticks=false)


# Number of realizations
npat = 100 # up to 100
K = 20
pvals = [0.1, 0.3, 0.5, 0.7, 0.9]
#w = [p->(1-p), p->p, p->p/(1-p), p->(1-p)/p]
w = [p->(1-p)]
scoremat = zeros(npat,length(pvals), 2+length(w))
for pind in eachindex(pvals)
    p = pvals[pind]
    for pat in 1:npat
        stest = sample_pp(pprocess,W)
        CVpairs = [TVsplit(stest,p) for k in 1:K]
        scoremat[pat,pind,1] = StoyanGrabarnik(stest,logλ,W)
        scoremat[pat,pind,2] = mean(PredictionError(CVpairs[k][1],CVpairs[k][2],logλ,W; p=p) for k in 1:K)
        for wind in eachindex(w)
            scoremat[pat,pind,2+wind] = mean(PredictionErrorCV(w[wind],CVpairs[k][1],CVpairs[k][2],logλ,W; p=p) for k in 1:K)
        end    
    end
end

titlelist = ["full", "sub", "cv: w(p)=1-p"]
yext1 = max(abs(maximum(scoremat[:,:,1:3])), abs(minimum(scoremat[:,:,1:3])))
#yext2 = max(abs(maximum(scoremat[:,:,4:6])), abs(minimum(scoremat[:,:,4:6])))
yext = [(t>3) ? yext2 : yext1 for t in eachindex(titlelist)]
jtr = 0.075
plt = [scatter(ylims=(-yext[t],yext[t]), xlabel="p", title=titlelist[t], titlefontsize=11) for t in eachindex(titlelist)]
for t in eachindex(titlelist)
    for i in eachindex(pvals)
        scatter!(plt[t],pvals[i] .+ jtr.*(rand(npat).-0.5),scoremat[:,i,t], 
            labels=false,
            marker=(2,:gray,0.3,stroke(0)))
    end
    plot!(plt[t], pvals, reshape(mean(scoremat[:,:,t], dims=1),length(pvals)), 
        yerror=reshape(std(scoremat[:,:,t],dims=1),length(pvals)), 
        markershape=:circle, color=:black, linewidth=2, label=false)
    hline!(plt[t],[0], linestyle=:dash, color=:gray, label=false)
end
plot(plt..., layout=(1,3))
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/prederrors_meansplit_strongGeyer.pdf")



### Using mean square root over K folds

# Strauss model
λ₀ = 400
γ = 0.5
R = 0.075
logλ(u::Point,s::PointSet) = log(λ₀) + log(γ)*(NNdist(s ∪ u,R) - NNdist(s,R))
pprocess = GibbsProcess(logλ)
stest = sample_pp(pprocess,W)
plot(stest,W,axis=false,ticks=false)

# Geyer model
λ₀ = 100
γ = 1.5
R = 0.05
sat = 5
t(u::Point,s::PointSet) = sum(dist(u,x)<R for x in s.items)
logλ(u::Point,s::PointSet) = log(λ₀) + log(γ)*min(sat,t(u,s))
pprocess = GibbsProcess(logλ)
stest = sample_pp(pprocess,W)
plot(stest,W,axis=false,ticks=false)


# Number of realizations
npat = 100 # up to 100
K = 20
pvals = [0.1, 0.3, 0.5, 0.7, 0.9]
#w = [p->(1-p), p->p, p->p/(1-p), p->(1-p)/p]
w = [p->(1-p)]
scoremat = zeros(npat,length(pvals), 2+length(w))
for pind in eachindex(pvals)
    p = pvals[pind]
    for pat in 1:npat
        stest = sample_pp(pprocess,W)
        CVpairs = [TVsplit(stest,p) for k in 1:K]
        scoremat[pat,pind,1] = abs(StoyanGrabarnik(stest,logλ,W))
        scoremat[pat,pind,2] = sqrt(mean( (PredictionError(CVpairs[k][1],CVpairs[k][2],logλ,W; p=p))^2 for k in 1:K))
        for wind in eachindex(w)
            scoremat[pat,pind,2+wind] = sqrt(mean( (PredictionErrorCV(w[wind],CVpairs[k][1],CVpairs[k][2],logλ,W; p=p))^2 for k in 1:K))
        end    
    end
end

titlelist = ["full", "sub", "cv: w(p)=1-p"]
yext1 = max(abs(maximum(scoremat[:,:,1:3])), abs(minimum(scoremat[:,:,1:3])))
#yext2 = max(abs(maximum(scoremat[:,:,4:6])), abs(minimum(scoremat[:,:,4:6])))
yext = [(t>3) ? yext2 : yext1 for t in eachindex(titlelist)]
jtr = 0.075
plt = [scatter(ylims=(0,yext[t]), xlabel="p", title=titlelist[t], titlefontsize=11) for t in eachindex(titlelist)]
for t in eachindex(titlelist)
    for i in eachindex(pvals)
        scatter!(plt[t],pvals[i] .+ jtr.*(rand(npat).-0.5),scoremat[:,i,t], 
            labels=false,
            marker=(2,:gray,0.3,stroke(0)))
    end
    plot!(plt[t], pvals, reshape(mean(scoremat[:,:,t], dims=1),length(pvals)), 
        yerror=reshape(std(scoremat[:,:,t],dims=1),length(pvals)), 
        markershape=:circle, color=:black, linewidth=2, label=false)
    hline!(plt[t],[0], linestyle=:dash, color=:gray, label=false)
end
plot(plt..., layout=(1,3))
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/prederrors_meansqrt_strongGeyer.pdf")

























##### Reconstruction tests


# Reconstruct the point pattern
s6n = reconstruct(s6,W,resolution)
plot(s6n,W, axis=false, ticks=false)
StoyanGrabarnik(s6n,logλ6,W)
StoyanGrabarnikPlot(s6n,logλ6,W,resolution)

# original : -0.0298
# n=2 : 0.0584
# n=3 : 0.04872
# n=4 : 0.02518
# n=5 : 0.01512
# n=6 : 0.00243
# n=7 : 0.00321
# n=8 :-0.01623
# n=9 :-0.02715
# n=10:-0.00159



stest = sample_pp(p6,W)
OrigScore = StoyanGrabarnik(stest,logλ6,W)
# Loop over all resolutions
nres = 15
nrec = 20
scoremat = zeros(nres,nrec)
for res in 1:nres
    for rec in 1:nrec
        recpatt = reconstruct(stest,W,res)
        scoremat[res,rec] = StoyanGrabarnik(recpatt,logλ6,W) -OrigScore
    end
end

using StatsBase
means = reshape(mean(scoremat, dims=2),nres)
vars = reshape(var(scoremat, dims=2),nres)

plot!(1:nres,means, label="Mean")
plot!(Shape(vcat(collect(1:nres),reverse(collect(1:nres))),vcat(means + vars, reverse(means -vars))), opacity=0.3, label="Variance")
#hline!([OrigScore], linestyle=:dash, label="Original")

u1 = range(0,stop=1,length=50)
u2 = u1
z = [logλ6(Point([x,y]),stest) for x in u1, y in u2]
heatmap(u1,u2,z')
plot!(stest,W)


##################################

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

#### Diagnostics

p = 0.3
sV,sT = TVsplit(s6,p)
PredictionError(sT,sV,(u,s)->logλtest(u,s,[2.5,3.5]),W;p=p)
PredictionErrorPlot(sT,sV,(u,s)->logλtest(u,s,[2.5,3.5]),W,4;p=p)
