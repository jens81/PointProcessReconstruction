## Remember to run Pkg> activate .
using Meshes, Plots, StatsBase, GLM, Combinatorics
import Random
include("../src/PointProcessLearning.jl")
using .PointProcessLearning
import ColorSchemes
import ColorSchemes.okabe_ito
cscheme = okabe_ito

#########################################
# Set up window
#########################################
W = Box((0.,0.),(1.,1.))
Wplus = PointProcessLearning.enlarge(W,0.1)


# Covariate density (random)
ngauss = 3
μ = [rand(2) for _ in 1:ngauss]
σx = [sqrt(0.05*rand()) for _ in 1:ngauss]; 
σy = [sqrt(0.05*rand()) for _ in 1:ngauss];
w = [0.5+rand() for _ in 1:ngauss];
fN(x,y,μ,σx,σy,w) = w*exp(-(x-μ[1])^2/(2*σx^2) - (y-μ[2])^2/(2*σy^2))
f(x,y) = sum(fN(x,y,μ[i],σx[i],σy[i],w[i]) for i in 1:ngauss)
#plot(range(0,1,length=20),range(0,1,length=20),f,st=:surf)
contour(range(0,1,length=20),range(0,1,length=20),f)
M = sum(w)
M = 1.5


f(x,y) = x
contour(range(0,1,length=20),range(0,1,length=20),f)


using Distributions
betadist = product_distribution([Beta(5, 5/i) for i in 1:2])
f2(u1,u2) = pdf(betadist,[u1,u2])
contour(range(0,1,length=20),range(0,1,length=20),f2)





#########################################
# Set up Geyer process
#########################################
# Parameters
# Distance between points
R = 0.05; sat=4;
t(u::Point,s::PointSet) = sum(dist(u,x)<R for x in s.items)
# Parameters
λ₀ = 100; g₀ = 1.25;# 1.25;
δ = 2.5;
S₁(u) = f(coordinates(u)[1],coordinates(u)[2]);
θ₀ = [log(λ₀),log(δ),log(g₀)]
S = [(u,s)->1,(u,s)->S₁(u),(u,s)->min(sat,t(u,s))]
#θ₀ = [log(λ₀),log(δ)]
#S = [(u,s)->1,(u,s)->S₁(u)]

# Generate patter
logλ(u,s,θ) = sum(θ[i]S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
p = GibbsProcess(logλ)
s = sample_pp(Random.GLOBAL_RNG,p,Wplus,niter=100_000)
contourf(range(0,1,length=20), range(0,1,length=20),f, opacity=0.1, aspect_ratio=1)
plot!(s,W, aspect_ratio=1)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/sample_geyer_100_2.5_1.25_withdens.pdf")
StoyanGrabarnik(s,logλ,W)

srec = reconstruct_with_density(s,W,1,S₁,M)
contourf(range(0,1,length=20), range(0,1,length=20),f, opacity=0.1, aspect_ratio=1)
plot!(srec,W)


# Resolutions
resolutions = [1, 2, 4, 8, 16, 32, 64]
boxes = [PointProcessLearning.partitions(W,res) for res in resolutions]
# Create original samples
originals = [PointProcessLearning.sample_pp(Random.GLOBAL_RNG, p,Wplus, niter=40_000) for _ in 1:200]
reconstructed = [PointProcessLearning.reconstruct(s,W,res) for s in originals, res in resolutions]
#reconstructed = [PointProcessLearning.reconstruct_with_density(s,W,res,S₁,1.5) for s in originals, res in resolutions]
# Compute various scores
EMDscores = [PointProcessLearning.EMDreconstructed(originals[i],reconstructed[i,j],W,resolutions[j]) for i in eachindex(originals), j in eachindex(resolutions)]
SGscoresOrig = map(s->PointProcessLearning.StoyanGrabarnik(s,logλ,W),originals)
SGscoresRec = map(s->PointProcessLearning.StoyanGrabarnik(s,logλ,W),reconstructed)
ParamsOrig = map(s->EstimateParamsPL_Logit(S,s,W; nd=80),originals)
ParamsRec = map(s->EstimateParamsPL_Logit(S,s,W; nd=80),reconstructed)


# EMD means and variances
EMDmeans = vec(mean(EMDscores, dims=1))
EMDvars = vec(var(EMDscores, dims=1))
# Plot
plot(resolutions.^(-1),EMDmeans, yerr=EMDvars,
    labels="Geyer process (θ₀=log$λ₀, θ₁=log$δ, θ₂=log$g₀)",
    xlabel="a (side length)",
    ylabel="EMD",
    linewidth=3,
    color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)))
Nmean = mean(map(s->N(s,W),originals))
plot!(resolutions.^(-1),fill(2*sqrt(Nmean)/2,length(resolutions)),
        labels="√N (large scale appr.)",
        legend=:bottomright,
        color=cscheme[2],
        linewidth=2,
        linestyle=:dash
)
#hline!([1/(2*sqrt(λmax))])
# average distance between two points in square with side a is ≈0.52*a
    # For N points the total distance is N*0.52*1/k
plot!(resolutions.^(-1),Nmean*0.52*resolutions.^(-1),
        labels="0.52*N*a (small scale appr.)",
        color=cscheme[2],
        ylims = (0,30),
        linewidth=2,
        linestyle=:dashdot
)
vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2)    
vline!([R], linestyle=:dot, color=:red, labels="R")
vline!([2*max(mean(σx),mean(σy))],labels="2σ", linestyle=:dot, color=:brown)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/EMD_geyer_100_2.5_1.25.pdf")




SGscoresRecmeans = vec(mean(SGscoresRec, dims=1))
SGscoresRecvars = vec(var(SGscoresRec, dims=1))
SGscoresOrigmax = maximum(SGscoresOrig)
SGscoresOrigmin = minimum(SGscoresOrig)
SGscoresOrigmean = mean(SGscoresOrig)
SGscoresOrigvar = var(SGscoresOrig)

# Plot SG
plot(resolutions.^(-1),SGscoresRecmeans, yerr=SGscoresRecvars,
    labels="Geyer process (θ₀=log$λ₀, θ₁=log$δ, θ₂=log$g₀)",
    xlabel="a (side length)",
    ylabel="Stoyan-Grabarnik residuals",
    color=cscheme[2],
    linewidth=3,
    ylims=(-0.1,0.4),
    legend=:bottomright,
    marker = (3,cscheme[2],5,stroke(1,:black)))
plot!(resolutions.^(-1), fill(SGscoresOrigmean-SGscoresOrigvar,length(resolutions)), 
    fillrange=fill(SGscoresOrigmean+SGscoresOrigvar,length(resolutions)), 
    alpha=0.2,
    color=cscheme[3],
    labels="Variance of original")
hline!([0], linestyle=:dash, color=:black, labels="Theory")
vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
vline!([R],labels="R", linestyle=:dot, color=:red)
vline!([2*max(mean(σx),mean(σy))],labels="2σ", linestyle=:dot, color=:brown)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/SG_geyer_100_2.5_1.25.pdf")


theta0orig = map(t->coef(t)[1], ParamsOrig)
theta1orig = map(t->coef(t)[2], ParamsOrig)
theta2orig = map(t->coef(t)[3], ParamsOrig)
theta0rec = map(t->coef(t)[1], ParamsRec)
theta1rec = map(t->coef(t)[2], ParamsRec)
theta2rec = map(t->coef(t)[3], ParamsRec)

theta0origmean = fill(mean(theta0orig),length(resolutions))
theta0origvar = fill(var(theta0orig),length(resolutions))
theta1origmean = fill(mean(theta1orig),length(resolutions))
theta1origvar = fill(var(theta1orig),length(resolutions))
theta2origmean = fill(mean(theta2orig),length(resolutions))
theta2origvar = fill(var(theta2orig),length(resolutions))

theta0recmean = vec(mean(theta0rec, dims=1))
theta0recvar = vec(var(theta0rec, dims=1))
theta1recmean = vec(mean(theta1rec, dims=1))
theta1recvar = vec(var(theta1rec, dims=1))
theta2recmean = vec(mean(theta2rec, dims=1))
theta2recvar = vec(var(theta2rec, dims=1))

plot(resolutions.^(-1),theta0recmean, yerr=theta0recvar,
    labels="Geyer process (θ₀=log$λ₀, θ₁=log$δ, θ₂=log$g₀)",
    xlabel="a (side length)",
    ylabel="θ₀ estimate",
    color=cscheme[2],
    #ylims=(4.5,5.5),
    linewidth=3,
    legend=:bottomright,
    marker = (3,cscheme[2],5,stroke(1,:black)))
#hline!([log(Nmean)], linestyle=:dash, color=cscheme[2], labels="θ₀=log(N) (Homogeneous approx)")
plot!(resolutions.^(-1), theta0origmean, linestyle=:dash, color=:gray, labels="θ₀ true")
plot!(resolutions.^(-1), theta0origmean.-theta0origvar, 
    fillrange=theta0origmean.+theta0origvar, 
    alpha=0.2,
    color=cscheme[3],
    labels="Variance of original")
vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
vline!([R],labels="R", linestyle=:dot, color=:red)
vline!([2*max(mean(σx),mean(σy))],labels="2σ", linestyle=:dot, color=:brown)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/theta0_geyer_100_2.5_1.25.pdf")


plot(resolutions.^(-1),theta1recmean, yerr=theta1recvar,
    labels="Geyer process (θ₀=log$λ₀, θ₁=log$δ, θ₂=log$g₀)",
    xlabel="a (side length)",
    ylabel="θ₁ estimate",
    color=cscheme[2],
    #ylims=(-0.05,1.7),
    linewidth=3,
    legend=:right,
    marker = (3,cscheme[2],5,stroke(1,:black)))
#hline!([0], linestyle=:dash, color=cscheme[2], labels="θ₁=0 (Homogeneous approx)")
plot!(resolutions.^(-1), theta1origmean, linestyle=:dash, color=:gray, labels="θ₁ true")
plot!(resolutions.^(-1), theta1origmean.-theta1origvar, 
    fillrange=theta1origmean.+theta1origvar, 
    alpha=0.2,
    color=cscheme[3],
    labels="Variance of original")
vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
vline!([R],labels="R", linestyle=:dot, color=:red)
vline!([2*max(mean(σx),mean(σy))],labels="2σ", linestyle=:dot, color=:brown)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/theta1_geyer_100_2.5_1.25.pdf")

plot(resolutions.^(-1),theta2recmean, yerr=theta2recvar,
    labels="Geyer process (θ₀=log$λ₀, θ₁=log$δ, θ₂=log$g₀)",
    xlabel="a (side length)",
    ylabel="θ₂ estimate",
    color=cscheme[2],
    linewidth=3,
    legend=:right,
    marker = (3,cscheme[2],5,stroke(1,:black)))
#hline!([0], linestyle=:dash, color=cscheme[2], labels="θ₂=0 (Poisson approx)")
plot!(resolutions.^(-1), theta2origmean, linestyle=:dash, color=:gray, labels="θ₂ true")
plot!(resolutions.^(-1), theta2origmean.-theta2origvar, 
    fillrange=theta2origmean.+theta2origvar, 
    alpha=0.2,
    color=cscheme[3],
    labels="Variance of original")
vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
vline!([R],labels="R", linestyle=:dot, color=:red)
vline!([2*max(mean(σx),mean(σy))],labels="2σ", linestyle=:dot, color=:brown)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/theta2_geyer_100_2.5_1.25.pdf")


function MISE2(s1::PointSet,s2::PointSet,logλ1::Function,logλ2::Function,W::Box; nd = 50)
    boxes = PointProcessLearning.partitions(W,nd)
    λ1(u,s) = exp(logλ1(u,s))
    λ2(u,s) = exp(logλ2(u,s))
    Sum = 0
    for box in boxes
        sb1 = PointSet(filter(x->in(x,box),s1.items))
        sb2 = PointSet(filter(x->in(x,box),s2.items))
        sbpoints = sb1.items ∪ sb2.items
        sb = length(sbpoints)>0 ? PointSet(sbpoints) : EmptyPointSet()
        db = PointSet(Meshes.centroid(box))
        vb = sb ∪ db
        nb = length(vb.items)
        w = Meshes.measure(box)/nb
        Sum = Sum + sum( w*(λ1(v,s1)-λ2(v,s2))^2 for v in vb.items)
    end
    return Sum
end

MISEscore = zeros(size(reconstructed))
for i in eachindex(originals), j in eachindex(resolutions)
    θ1 = coef(ParamsOrig[i])
    θ2 = coef(ParamsRec[i,j])
    s1 = originals[i]
    s2 = reconstructed[i,j]
    MISEscore[i,j] = MISE2(s1,s2,(u,s)->logλ(u,s,θ1),(u,s)->logλ(u,s,θ2),W,nd=50)
end

MISEmean = vec(mean(MISEscore, dims=1))
MISEvar = vec(var(MISEscore, dims=1))

plot(resolutions.^(-1),sqrt.(MISEmean),#, yerr=MISEvar,
    labels="Geyer process (θ₀=log$λ₀, θ₁=log$δ, θ₂=log$g₀)",
    xlabel="a (side length)",
    ylabel="RMISE",
    color=cscheme[2],
    linewidth=3,
    legend=:right,
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
vline!([R],labels="R", linestyle=:dot, color=:red)
vline!([2*max(mean(σx),mean(σy))], linestyle=:dot, color=:brown, labels="2σ")
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/RMISE_geyer_100_2.5_1.25.pdf")










# Fixed
ngauss = 1
μ = [[0.625,0.625] for _ in 1:ngauss]
σx = [0.075 for _ in 1:ngauss]; 
σy = [0.075 for _ in 1:ngauss];
w = [1 for _ in 1:ngauss];
fN(x,y,μ,σx,σy,w) = w*exp(-(x-μ[1])^2/(2*σx^2) - (y-μ[2])^2/(2*σy^2))
f(x,y) = sum(fN(x,y,μ[i],σx[i],σy[i],w[i]) for i in 1:ngauss)
#plot(range(0,1,length=20),range(0,1,length=20),f,st=:surf)
contour(range(0,1,length=20),range(0,1,length=20),f)
M = sum(w)

#########################################
# Set up Geyer process
#########################################
# Parameters
# Distance between points
R = 0.05; sat=4;
t(u::Point,s::PointSet) = sum(dist(u,x)<R for x in s.items)
# Parameters
λ₀ = 50; g₀ = 1;# 1.25;
δ = 50;
S₂(u) = f(coordinates(u)[1],coordinates(u)[2]);
#θ₀ = [log(λ₀),log(δ),log(g₀)]
#S = [(u,s)->1,(u,s)->S₂(u),(u,s)->min(sat,t(u,s))]
θ₀ = [log(λ₀),log(δ)]
S = [(u,s)->1,(u,s)->S₂(u)]

# Generate patter
logλ(u,s,θ) = sum(θ[i]S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
p = GibbsProcess(logλ)
s = sample_pp(Random.GLOBAL_RNG,p,Wplus,niter=100_000)
contourf(range(0,1,length=20), range(0,1,length=20),f, opacity=0.1, aspect_ratio=1)
plot!(s,W, aspect_ratio=1)


srecs = [reconstruct(s,W,res) for res in resolutions]
contourf(range(0,1,length=20), range(0,1,length=20),f, opacity=0.1, aspect_ratio=1)
plot!(srecs[2],W, aspect_ratio=1)


plts = [contourf(range(0,1,length=20), range(0,1,length=20),f, opacity=0.1, aspect_ratio=1, colorbar=false, ticks=false) for _ in 1:length(resolutions)+1]
plot!(plts[end],s,W,aspect_ratio=1, title="original")
for i in 1:length(plts)-1
    plot!(plts[i],srecs[i],W,aspect_ratio=1, title="a=$(round(1/resolutions[i],digits=3)) (k=$(resolutions[i]))")
    plot!(plts[i],PointProcessLearning.Box2Shape.(boxes[i]), linestyle=:dot, strokecolor=:black, linewidth=1, fillcolor=false, strokeopacity=0.5)
end

plot(reverse(plts)..., layout=(2,4))
plot!(size=(2000/2,2000/4))
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/inhom_poisson_multiple.pdf")
