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
Wplus = PointProcessLearning.enlarge(W,0.05)

#########################################
# Set up linear network
#########################################

LR = RandomLines(25,PointProcessLearning.reduce(W,0.05))
L = LinearNetwork(LR)
PlotLinearNetwork(L)
plot!(EmptyPointSet(),W)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/linearnetwork1.pdf")
p = BinomialProcess(50)
s,segids = sample_pp(p,L)
plot!(s,W)

tL = thin(L,0.75)
PlotLinearNetwork(tL)
plot!(EmptyPointSet(),W)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/linearnetwork2.pdf")
contourf!(range(0,1,length=20), range(0,1,length=20),f, opacity=0.1, aspect_ratio=1)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/linearnetwork3.pdf")
p = BinomialProcess(100)
s = sample_pp(p,tL)
plot!(s,W)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/linearnetwork4.pdf")

L = tL # to simplify notation


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



#########################################
# Set up Geyer process
#########################################
# Parameters
# Distance between points
R = 0.025; sat=4;
t(u::Point,s::PointSet) = sum(dist(u,x)<R for x in s.items)
# Parameters
λ₀ = 5; g₀ = 1.25;# 1.25;
δ = 5;
S₁(u) = f(coordinates(u)[1],coordinates(u)[2]);
θ₀ = [log(λ₀),log(δ),log(g₀)]
S = [(u,s)->1,(u,s)->S₁(u),(u,s)->min(sat,t(u,s))]
#θ₀ = [log(λ₀),log(δ)]
#S = [(u,s)->1,(u,s)->S₁(u)]

# Generate patter
logλ(u,s,θ) = sum(θ[i]S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
p = GibbsProcess(logλ)
s,segids = sample_pp(p,L)
PlotLinearNetwork(L)
contourf!(range(0,1,length=20), range(0,1,length=20),f, opacity=0.1, aspect_ratio=1)
plot!(s,W, aspect_ratio=1)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/sample_geyer_linear.pdf")




# Resolutions
resolutions = [1, 2, 4, 8, 16, 32, 64]
boxes = [PointProcessLearning.partitions(W,res) for res in resolutions]
# Create original samples
originals = Array{PointSet}(undef,200)
originals_segids = Array{Vector}(undef,200)
reconstructed_pats = Array{Vector}(undef,(200,length(resolutions)))
reconstructed_segids = Array{Vector}(undef,(200,length(resolutions)))
reconstructed_nets = Array{Vector}(undef,(200,length(resolutions)))
reconstructed = Array{PointSet}(undef,(200,length(resolutions)))
for i in 1:200
    s, segs = sample_pp(p,L, niter=40_000)
    originals[i], originals_segids[i] = (s,segs)
    for j in eachindex(resolutions)
        pats, segs, nets = reconstruct(s,L,W,resolutions[j])
        reconstructed[i,j] = PointSet(vcat([ss.items for ss in pats]...))
        reconstructed_pats[i,j],reconstructed_segids[i,j], reconstructed_nets[i,j] = (pats,segs,nets)
    end
    println("finished i=$i")
end
#reconstructed = [PointProcessLearning.reconstruct_with_density(s,W,res,S₁,1.5) for s in originals, res in resolutions]
# Compute various scores
#EMDscores = [PointProcessLearning.EMDreconstructed(originals[i],reconstructed[i,j],W,resolutions[j]) for i in eachindex(originals), j in eachindex(resolutions)]
SGscoresOrig = Array{Float64}(undef,200)
SGscoresRec = Array{Float64}(undef,(200, length(resolutions)))
for i in eachindex(originals)
    SGscoresOrig[i] = StoyanGrabarnik(originals[i],logλ,L,originals_segids[i])
    for j in eachindex(resolutions)
        pats,segs,nets = (reconstructed_pats[i,j],reconstructed_segids[i,j],reconstructed_nets[i,j])
        SGscoresRec[i,j] = sum(StoyanGrabarnik(pats[l],logλ,nets[l],segs[l]) for l in eachindex(pats))
    end
end
ParamsOrig = map(s->EstimateParamsPL_Logit3(S,s,L; nd=80),originals)
ParamsRec = map(s->EstimateParamsPL_Logit3(S,s,L; nd=80),reconstructed)

# EMD means and variances
# EMDmeans = vec(mean(EMDscores, dims=1))
# EMDvars = vec(var(EMDscores, dims=1))
# # Plot
# plot(resolutions.^(-1),EMDmeans, yerr=EMDvars,
#     labels="Geyer process (θ₀=log$λ₀, θ₁=log$δ, θ₂=log$g₀)",
#     xlabel="a (side length)",
#     ylabel="EMD",
#     linewidth=3,
#     color=cscheme[2],
#     marker = (3,cscheme[2],5,stroke(1,:black)))
# Nmean = mean(map(s->N(s,W),originals))
# plot!(resolutions.^(-1),fill(2*sqrt(Nmean)/2,length(resolutions)),
#         labels="√N (large scale appr.)",
#         legend=:bottomright,
#         color=cscheme[2],
#         linewidth=2,
#         linestyle=:dash
# )
# #hline!([1/(2*sqrt(λmax))])
# # average distance between two points in square with side a is ≈0.52*a
#     # For N points the total distance is N*0.52*1/k
# plot!(resolutions.^(-1),Nmean*0.52*resolutions.^(-1),
#         labels="0.52*N*a (small scale appr.)",
#         color=cscheme[2],
#         ylims = (0,30),
#         linewidth=2,
#         linestyle=:dashdot
# )
# vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2)    
# vline!([R], linestyle=:dot, color=:red, labels="R")
# vline!([2*max(mean(σx),mean(σy))],labels="2σ", linestyle=:dot, color=:brown)
# savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/EMD_geyer_linear.pdf")




# SGscoresRecmeans = vec(mean(SGscoresRec, dims=1))
# SGscoresRecvars = vec(var(SGscoresRec, dims=1))
# SGscoresOrigmax = maximum(SGscoresOrig)
# SGscoresOrigmin = minimum(SGscoresOrig)
# SGscoresOrigmean = mean(SGscoresOrig)
# SGscoresOrigvar = var(SGscoresOrig)

# # Plot SG
# plot(resolutions.^(-1),SGscoresRecmeans, yerr=SGscoresRecvars,
#     labels="Geyer process (θ₀=log$λ₀, θ₁=log$δ, θ₂=log$g₀)",
#     xlabel="a (side length)",
#     ylabel="Stoyan-Grabarnik residuals",
#     color=cscheme[2],
#     linewidth=3,
#     #ylims=(-0.1,0.4),
#     legend=:bottomright,
#     marker = (3,cscheme[2],5,stroke(1,:black)))
# plot!(resolutions.^(-1), fill(SGscoresOrigmean-SGscoresOrigvar,length(resolutions)), 
#     fillrange=fill(SGscoresOrigmean+SGscoresOrigvar,length(resolutions)), 
#     alpha=0.2,
#     color=cscheme[3],
#     labels="Variance of original")
# hline!([0], linestyle=:dash, color=:black, labels="Theory")
# vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
# vline!([R],labels="R", linestyle=:dot, color=:red)
# vline!([2*max(mean(σx),mean(σy))],labels="2σ", linestyle=:dot, color=:brown)
# savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/SG_geyer_linear.pdf")


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
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/theta0_geyer_linear.pdf")


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
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/theta1_geyer_linear.pdf")

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
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/theta2_geyer_linear.pdf")



#########################################
# Set up inhom poisson process
#########################################
# Parameters
λ₀ = 5;
δ = 5;
S₁(u) = f(coordinates(u)[1],coordinates(u)[2]);
#θ₀ = [log(λ₀),log(δ),log(g₀)]
#S = [(u,s)->1,(u,s)->S₁(u),(u,s)->min(sat,t(u,s))]
θ₀ = [log(λ₀),log(δ)]
S = [(u,s)->1,(u,s)->S₁(u)]

# Generate patter
logλ(u,s,θ) = sum(θ[i]S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
p = GibbsProcess(logλ)
s,segids = sample_pp(p,L)
PlotLinearNetwork(L)
contourf!(range(0,1,length=20), range(0,1,length=20),f, opacity=0.1, aspect_ratio=1)
plot!(s,W, aspect_ratio=1)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/sample_inhom_poisson_linear.pdf")




# Resolutions
resolutions = [1, 2, 4, 8, 16, 32, 64]
boxes = [PointProcessLearning.partitions(W,res) for res in resolutions]
# Create original samples
originals = Array{PointSet}(undef,200)
originals_segids = Array{Vector}(undef,200)
reconstructed_pats = Array{Vector}(undef,(200,length(resolutions)))
reconstructed_segids = Array{Vector}(undef,(200,length(resolutions)))
reconstructed_nets = Array{Vector}(undef,(200,length(resolutions)))
reconstructed = Array{PointSet}(undef,(200,length(resolutions)))
for i in 1:200
    s, segs = sample_pp(p,L, niter=40_000)
    originals[i], originals_segids[i] = (s,segs)
    for j in eachindex(resolutions)
        pats, segs, nets = reconstruct(s,L,W,resolutions[j])
        reconstructed[i,j] = PointSet(vcat([ss.items for ss in pats]...))
        reconstructed_pats[i,j],reconstructed_segids[i,j], reconstructed_nets[i,j] = (pats,segs,nets)
    end
    println("finished i=$i")
end
#reconstructed = [PointProcessLearning.reconstruct_with_density(s,W,res,S₁,1.5) for s in originals, res in resolutions]
# Compute various scores
#EMDscores = [PointProcessLearning.EMDreconstructed(originals[i],reconstructed[i,j],W,resolutions[j]) for i in eachindex(originals), j in eachindex(resolutions)]
SGscoresOrig = Array{Float64}(undef,200)
SGscoresRec = Array{Float64}(undef,(200, length(resolutions)))
for i in eachindex(originals)
    SGscoresOrig[i] = StoyanGrabarnik(originals[i],logλ,L,originals_segids[i])
    for j in eachindex(resolutions)
        pats,segs,nets = (reconstructed_pats[i,j],reconstructed_segids[i,j],reconstructed_nets[i,j])
        SGscoresRec[i,j] = sum(StoyanGrabarnik(pats[l],logλ,nets[l],segs[l]) for l in eachindex(pats))
    end
end
ParamsOrig = map(s->EstimateParamsPL_Logit3(S,s,L; nd=80),originals)
ParamsRec = map(s->EstimateParamsPL_Logit3(S,s,L; nd=80),reconstructed)

# # EMD means and variances
# EMDmeans = vec(mean(EMDscores, dims=1))
# EMDvars = vec(var(EMDscores, dims=1))
# # Plot
# plot(resolutions.^(-1),EMDmeans, yerr=EMDvars,
#     labels="Geyer process (θ₀=log$λ₀, θ₁=log$δ, θ₂=log$g₀)",
#     xlabel="a (side length)",
#     ylabel="EMD",
#     linewidth=3,
#     color=cscheme[2],
#     marker = (3,cscheme[2],5,stroke(1,:black)))
# Nmean = mean(map(s->N(s,W),originals))
# plot!(resolutions.^(-1),fill(2*sqrt(Nmean)/2,length(resolutions)),
#         labels="√N (large scale appr.)",
#         legend=:bottomright,
#         color=cscheme[2],
#         linewidth=2,
#         linestyle=:dash
# )
# #hline!([1/(2*sqrt(λmax))])
# # average distance between two points in square with side a is ≈0.52*a
#     # For N points the total distance is N*0.52*1/k
# plot!(resolutions.^(-1),Nmean*0.52*resolutions.^(-1),
#         labels="0.52*N*a (small scale appr.)",
#         color=cscheme[2],
#         ylims = (0,30),
#         linewidth=2,
#         linestyle=:dashdot
# )
# vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2)    
# vline!([R], linestyle=:dot, color=:red, labels="R")
# vline!([2*max(mean(σx),mean(σy))],labels="2σ", linestyle=:dot, color=:brown)
# savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/EMD_poisson_linear.pdf")




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
    #ylims=(-0.1,0.4),
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
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/SG_poisson_linear.pdf")


theta0orig = map(t->coef(t)[1], ParamsOrig)
theta1orig = map(t->coef(t)[2], ParamsOrig)
#theta2orig = map(t->coef(t)[3], ParamsOrig)
theta0rec = map(t->coef(t)[1], ParamsRec)
theta1rec = map(t->coef(t)[2], ParamsRec)
#theta2rec = map(t->coef(t)[3], ParamsRec)

theta0origmean = fill(mean(theta0orig),length(resolutions))
theta0origvar = fill(var(theta0orig),length(resolutions))
theta1origmean = fill(mean(theta1orig),length(resolutions))
theta1origvar = fill(var(theta1orig),length(resolutions))
#theta2origmean = fill(mean(theta2orig),length(resolutions))
#theta2origvar = fill(var(theta2orig),length(resolutions))

theta0recmean = vec(mean(theta0rec, dims=1))
theta0recvar = vec(var(theta0rec, dims=1))
theta1recmean = vec(mean(theta1rec, dims=1))
theta1recvar = vec(var(theta1rec, dims=1))
#theta2recmean = vec(mean(theta2rec, dims=1))
#theta2recvar = vec(var(theta2rec, dims=1))

plot(resolutions.^(-1),theta0recmean, yerr=theta0recvar,
    labels="Inhom. Pois. process (θ₀=log$λ₀, θ₁=log$δ)",
    xlabel="a (side length)",
    ylabel="θ₀ estimate",
    color=cscheme[2],
    #ylims=(4.5,5.5),
    linewidth=3,
    legend=:right,
    marker = (3,cscheme[2],5,stroke(1,:black)))
#hline!([log(Nmean)], linestyle=:dash, color=cscheme[2], labels="θ₀=log(N) (Homogeneous approx)")
plot!(resolutions.^(-1), theta0origmean, linestyle=:dash, color=:gray, labels="θ₀ true")
plot!(resolutions.^(-1), theta0origmean.-theta0origvar, 
    fillrange=theta0origmean.+theta0origvar, 
    alpha=0.2,
    color=cscheme[3],
    labels="Variance of original")
vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
#vline!([R],labels="R", linestyle=:dot, color=:red)
vline!([2*max(mean(σx),mean(σy))],labels="2σ", linestyle=:dot, color=:brown)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/theta0_inhom_poisson_linear.pdf")


plot(resolutions.^(-1),theta1recmean, yerr=theta1recvar,
    labels="Inhom. Pois. process (θ₀=log$λ₀, θ₁=log$δ)",
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
#vline!([R],labels="R", linestyle=:dot, color=:red)
vline!([2*max(mean(σx),mean(σy))],labels="2σ", linestyle=:dot, color=:brown)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/theta1_inhom_poisson_linear.pdf")











function EstimateParamsPL_Logit3(S::Vector{Function}, s::PointSet, L::LinearNetwork; nd = 400)
    segments = L.segments
    probs = Meshes.measure.(segments)
    measureL = sum(probs)
    probs = probs./sum(probs) # normalize
    nu = floor.(Int64,probs.*nd)
    u = PointSet(vcat([PointProcessLearning.SamplePointsOnSegment(segments[i],nu[i]) for i in 1:length(segments)]...))
    #segids_u = vcat([repeat(i,nu[i]) for i in 1:length(segments)]...)
    #sw = PointSet(filter(x->x∈W,s.items))
    #v = u ∪ sw
    v = PointSet(vcat(u.items,s.items))
    #segids = vcat(segids_u,segids)
    Nv = length(v.items)
    K = length(S)
    #segind = [findfirst(seg -> in(x,seg),segments) for x in v.items] 
    r = nd/measureL
    ofs = fill(log(1/r),Nv)
    # Now define
    #w = zeros(Nv)
    y = zeros(Nv)
    X = zeros(Nv,K)
    for i in 1:Nv
        #seg = segments[segids[i]]
        vi = v.items[i]
        #w[i] = a / (N(v,box))
        y[i] = (vi ∈ s.items) ? 1 : 0
        for j in 1:K
            X[i,j] = S[j](vi,PointSet(setdiff(s.items,[vi])))
            #X[i,j] = S[j](vi,s)
        end
    end
    res = glm(X,y,Bernoulli(),LogitLink(); offset=ofs)
    return res
end