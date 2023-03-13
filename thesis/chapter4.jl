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

# Test 
λ = 20
logλ = (u,s)->log(200)
p = PoissonProcess(λ)
s = sample_pp(p,W)
plot(s,W)


# Resolutions
resolutions = [1, 2, 4, 8, 16, 32,64]
boxes = [PointProcessLearning.partitions(W,res) for res in resolutions]

# Create original samples
originals = [sample_pp(p,W) for _ in 1:200]
reconstructed = [reconstruct(s,W,res) for s in originals, res in resolutions]

EMDscores = [EMDreconstructed(originals[i],reconstructed[i,j],W,resolutions[j]) for i in eachindex(originals), j in eachindex(resolutions)]
#SGscoresOrig = map(s->PointProcessLearning.StoyanGrabarnik(s,logλ,W),originals)
#SGscoresRec = map(s->PointProcessLearning.StoyanGrabarnik(s,logλ,W),reconstructed)
#MISEscores = 
#ParamEstimates

EMDmeans = vec(mean(EMDscores, dims=1))
EMDvars = vec(var(EMDscores, dims=1))

plot(resolutions,EMDmeans, yerr=EMDvars,
    labels="Poisson point process",
    xlabel="k",
    ylabel="EMD",
    color=cscheme[2],
    linewidth=3,
    marker = (3,cscheme[2],5,stroke(1,:black)))
plot(resolutions.^(-1),EMDmeans, yerr=EMDvars,
    labels="Poisson point process, λ=150",
    xlabel="a (side length)",
    ylabel="EMD",
    legend=:bottomright,
    color=cscheme[2],
    linewidth=3,
    marker = (3,cscheme[2],5,stroke(1,:black))
)
Nmean = mean(map(s->N(s,W),originals))
plot!(resolutions.^(-1),fill(2*sqrt(Nmean)/2,length(resolutions)),
    labels="√N (large scale appr.)",
    color=cscheme[2],
    linewidth=2,
    linestyle=:dash
)
# average distance between two points in square with side a is ≈0.52*a
# For N points the total distance is N*0.52*1/k
plot!(resolutions.^(-1),Nmean*0.52*resolutions.^(-1), ylims=(0,1.25*maximum(EMDmeans)),
    labels="0.52*N*a (small scale appr.)",
    color=cscheme[2],
    linewidth=2,
    linestyle=:dashdot
)
vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2)    
#vline!([sqrt(2)/sqrt(Nmean)])
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/EMD_poisson_150.pdf")



#### Same for linear LinearNetwork
measurelin(L::LinearNetwork) = (length(L.segments)>0) ? sum(Meshes.measure.(L.segments)) : 0
function sample_pplin(p::PoissonProcess{<:Real}, L::LinearNetwork)
    # simulate number of points
    λ = p.λ; V = measurelin(L)
    n = rand(Poisson(λ*V))
    # product of uniform distributions
    pbin = BinomialProcess(n)
    s, segids = sample_pp(pbin,L)
    # return point pattern
    return s
end
lines = RandomLines(20,W)
L = LinearNetwork(lines)
s = sample_pplin(p,L)
PlotLinearNetwork(L)
plot!(s,W)





Wplus = enlarge(W,R)



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
logλ(u,s,θ) = sum(θ[i]S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
p = GibbsProcess(logλ)
s = sample_pp(Random.GLOBAL_RNG,p,Wplus,niter=50_000)
plot(s,W)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/sample_geyer_60_1.35_R0075sat8.pdf")
StoyanGrabarnik(s,logλ,W)

# Resolutions
resolutions = [1, 2, 4, 8, 16, 32, 64]
boxes = [PointProcessLearning.partitions(W,res) for res in resolutions]
# Create original samples
originals = [PointProcessLearning.sample_pp(Random.GLOBAL_RNG, p,Wplus, niter=100_000) for _ in 1:200]
reconstructed = [PointProcessLearning.reconstruct(s,W,res) for s in originals, res in resolutions]
# Compute various scores
EMDscores = [PointProcessLearning.EMDreconstructed(originals[i],reconstructed[i,j],W,resolutions[j]) for i in eachindex(originals), j in eachindex(resolutions)]
SGscoresOrig = map(s->PointProcessLearning.StoyanGrabarnik(s,logλ,W),originals)
SGscoresRec = map(s->PointProcessLearning.StoyanGrabarnik(s,logλ,W),reconstructed)
ParamsOrig = map(s->EstimateParamsPL_Logit(S,s,W; nd=100),originals)
ParamsRec = map(s->EstimateParamsPL_Logit(S,s,W; nd=100),reconstructed)


# EMD means and variances
EMDmeans = vec(mean(EMDscores, dims=1))
EMDvars = vec(var(EMDscores, dims=1))
# Plot
plot(resolutions.^(-1),EMDmeans, yerr=EMDvars,
    labels="Geyer process (θ₀=log 100, θ₂=log 1.25)",
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
    # average distance between two points in square with side a is ≈0.52*a
    # For N points the total distance is N*0.52*1/k
plot!(resolutions.^(-1),Nmean*0.52*resolutions.^(-1), ylims=(0,1.25*maximum(EMDmeans)),
        labels="0.52*N*a (small scale appr.)",
        color=cscheme[2],
        linewidth=2,
        linestyle=:dashdot
)
vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2)    
vline!([R], linestyle=:dot, color=:red, labels="R")
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/EMD_geyer_60_1.35_R0075sat8.pdf")




SGscoresRecmeans = vec(mean(SGscoresRec, dims=1))
SGscoresRecvars = vec(var(SGscoresRec, dims=1))
SGscoresOrigmax = maximum(SGscoresOrig)
SGscoresOrigmin = minimum(SGscoresOrig)
SGscoresOrigmean = mean(SGscoresOrig)
SGscoresOrigvar = var(SGscoresOrig)

# Plot SG
plot(resolutions.^(-1),SGscoresRecmeans, yerr=SGscoresRecvars,
    labels="Geyer process (θ₀=log 100, θ₂=log 1.25)",
    xlabel="a (side length)",
    ylabel="Stoyan-Grabarnik residuals",
    color=cscheme[2],
    linewidth=3,
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
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/SG_geyer_60_1.35_R0075sat8.pdf")

# Attempt to find bounds for SG
(Nmean-exp(θ₀[1]))/exp(θ₀[1])
(Nmean-exp(θ₀[1]+sat*θ₀[2]))/exp(θ₀[1]+sat*θ₀[2])

theta0orig = map(t->coef(t)[1], ParamsOrig)
theta2orig = map(t->coef(t)[2], ParamsOrig)
theta0rec = map(t->coef(t)[1], ParamsRec)
theta2rec = map(t->coef(t)[2], ParamsRec)

theta0origmean = fill(mean(theta0orig),length(resolutions))
theta0origvar = fill(var(theta0orig),length(resolutions))
theta2origmean = fill(mean(theta2orig),length(resolutions))
theta2origvar = fill(var(theta2orig),length(resolutions))

theta0recmean = vec(mean(theta0rec, dims=1))
theta0recvar = vec(var(theta0rec, dims=1))
theta2recmean = vec(mean(theta2rec, dims=1))
theta2recvar = vec(var(theta2rec, dims=1))

plot(resolutions.^(-1),theta0recmean, yerr=theta0recvar,
    labels="Geyer process (θ₀=log 100, θ₂=log 1.25)",
    xlabel="a (side length)",
    ylabel="θ₀ estimate",
    color=cscheme[2],
    linewidth=3,
    legend=:bottomright,
    marker = (3,cscheme[2],5,stroke(1,:black)))
hline!([log(Nmean)], linestyle=:dash, color=cscheme[2], labels="θ₀=log(N) (Poisson approx)")
plot!(resolutions.^(-1), theta0origmean, linestyle=:dash, color=:gray, labels="θ₀ true")
plot!(resolutions.^(-1), theta0origmean.-theta0origvar, 
    fillrange=theta0origmean.+theta0origvar, 
    alpha=0.2,
    color=cscheme[3],
    labels="Variance of original")
vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
vline!([R],labels="R", linestyle=:dot, color=:red)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/theta0_geyer_60_1.35_R0075sat8.pdf")

plot(resolutions.^(-1),theta2recmean, yerr=theta2recvar,
    labels="Geyer process (θ₀=log 100, θ₂=log 1.25)",
    xlabel="a (side length)",
    ylabel="θ₂ estimate",
    color=cscheme[2],
    linewidth=3,
    legend=:topright,
    marker = (3,cscheme[2],5,stroke(1,:black)))
hline!([0], linestyle=:dash, color=cscheme[2], labels="θ₂=0 (Poisson approx)")
plot!(resolutions.^(-1), theta2origmean, linestyle=:dash, color=:gray, labels="θ₀ true")
plot!(resolutions.^(-1), theta2origmean.-theta2origvar, 
    fillrange=theta2origmean.+theta2origvar, 
    alpha=0.2,
    color=cscheme[3],
    labels="Variance of original")
vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
vline!([R],labels="R", linestyle=:dot, color=:red)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/theta2_geyer_60_1.35_R0075sat8.pdf")


function MISE2(s1::PointSet,s2::PointSet,logλ1::Function,logλ2::Function,W::Box; nd = 100)
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
    labels="Geyer process (θ₀=log 100, θ₂=log 1.25)",
    xlabel="a (side length)",
    ylabel="RMISE",
    color=cscheme[2],
    linewidth=3,
    legend=:right,
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
vline!([R],labels="R", linestyle=:dot, color=:red)
vline!([R*2*sqrt(2)/(sqrt(2)-1)], linestyle=:dot, color=:orange, labels="Volume")
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/RMISE_geyer_100_1.25_R005sat4.pdf")


a = collect(range(0,1, step=0.001))
areafrac(a,R) = a>R ? (a - R)^2/(a^2) : 0
R1 = 0.025
areafrac(a) = areafrac(a,R1)
plot(a,areafrac.(a), linewidth=4, 
    xlabel="a (side length)", 
    ylabel="fraction of quadrat contained area",
    labels="R=$R1",
    color=cscheme[3],
    legend=:bottomright
)
R2 = 0.05
areafrac(a) = areafrac(a,R2)
plot!(a,areafrac.(a), linewidth=4, linestyle=:dash, color=cscheme[4], labels="R=$R2")
R3 = 0.1
areafrac(a) = areafrac(a,R3)
plot!(a,areafrac.(a), linewidth=4, linestyle=:dashdot, color=cscheme[5], labels="R=$R3")
R4 = 0.2
areafrac(a) = areafrac(a,R4)
plot!(a,areafrac.(a), linewidth=4, linestyle=:dashdotdot, color=cscheme[6], labels="R=$R4")
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/fractionplot.pdf")





# testing plotting against fraction of area
areafrac(a) = areafrac(a,R) + (4*R-2*R^2)*a^2
areafrac(a) = areafrac(a,R)
plot(areafrac.(resolutions.^(-1)),SGscoresRecmeans, yerr=SGscoresRecvars,
    labels="Geyer process (θ₀=log 100, θ₂=log 1.25)",
    xlabel="fraction of contained area",
    ylabel="Stoyan-Grabarnik residuals",
    color=cscheme[2],
    linewidth=3,
    legend=:bottomright,
    marker = (3,cscheme[2],5,stroke(1,:black)))
plot!(areafrac.(resolutions.^(-1)), fill(SGscoresOrigmean-SGscoresOrigvar,length(resolutions)), 
    fillrange=fill(SGscoresOrigmean+SGscoresOrigvar,length(resolutions)), 
    alpha=0.2,
    color=cscheme[3],
    labels="Variance of original")
hline!([0], linestyle=:dash, color=:black, labels="Theory")
vline!(areafrac.([1/sqrt(Nmean)]), linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
vline!(areafrac.([R]),labels="R", linestyle=:dot, color=:red)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/SG_geyer_60_1.35_R0075sat8_fraction.pdf")


plot(areafrac.(resolutions.^(-1)),theta0recmean, yerr=theta0recvar,
    labels="Geyer process (θ₀=log 100, θ₂=log 1.25)",
    xlabel="fraction of contained area",
    ylabel="θ₀ estimate",
    color=cscheme[2],
    linewidth=3,
    legend=:bottomright,
    marker = (3,cscheme[2],5,stroke(1,:black)))
hline!([log(Nmean)], linestyle=:dash, color=cscheme[2], labels="θ₀=log(N) (Poisson approx)")
plot!(areafrac.(resolutions.^(-1)), theta0origmean, linestyle=:dash, color=:gray, labels="θ₀ true")
plot!(areafrac.(resolutions.^(-1)), theta0origmean.-theta0origvar, 
    fillrange=theta0origmean.+theta0origvar, 
    alpha=0.2,
    color=cscheme[3],
    labels="Variance of original")
vline!(areafrac.([1/sqrt(Nmean)]), linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
#vline!(areafrac.([R]),labels="R", linestyle=:dot, color=:red)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/theta0_geyer_60_1.35_R0075sat8_fraction.pdf")

plot(areafrac.(resolutions.^(-1)),theta2recmean, yerr=theta2recvar,
    labels="Geyer process (θ₀=log 100, θ₂=log 1.25)",
    xlabel="fraction of contained area",
    ylabel="θ₂ estimate",
    color=cscheme[2],
    linewidth=3,
    legend=:topright,
    marker = (3,cscheme[2],5,stroke(1,:black)))
hline!([0], linestyle=:dash, color=cscheme[2], labels="θ₂=0 (Poisson approx)")
plot!(areafrac.(resolutions.^(-1)), theta2origmean, linestyle=:dash, color=:gray, labels="θ₀ true")
plot!(areafrac.(resolutions.^(-1)), theta2origmean.-theta2origvar, 
    fillrange=theta2origmean.+theta2origvar,
    color=cscheme[3], 
    alpha=0.2,
    labels="Variance of original")
vline!(areafrac.([1/sqrt(Nmean)]), linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
#vline!(areafrac.([R]),labels="R", linestyle=:dot, color=:red)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/theta2_geyer_60_1.35_R0075sat8_fraction.pdf")




plot(areafrac.(resolutions.^(-1)),sqrt.(MISEmean),#, yerr=MISEvar,
    labels="Geyer process (θ₀=log 100, θ₂=log 1.25)",
    xlabel="fraction of contained area",
    ylabel="RMISE",
    color=cscheme[2],
    linewidth=3,
    legend=:right,
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
vline!([R],labels="R", linestyle=:dot, color=:red)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/RMISE_geyer_100_1.25_R005sat4_fraction.pdf")





###############################
# Count data
###############################
noriginals = length(originals)
nresolutions = length(resolutions)
Nvectors = Array{Vector{Int64}}(undef, (noriginals,nresolutions))
for i in 1:noriginals, j in 1:nresolutions
    s = originals[i]
    boxes = PointProcessLearning.partitions(W,resolutions[j])
    Nvectors[i,j] = [N(s,box) for box in boxes]
end

using Distributions
function plot_Nvectors(j::Int64,Nvectors::Array{Vector{Int64}})
    res = resolutions[j]
    plt = plot(;title="k=$res",
            xlabel="nᵢ", ylabel="occurrences")
    #for i in 1:noriginals
    #    hist = countmap(Nvectors[i,j])
    #    scatter!(plt, collect(keys(hist)).-0.15.+0.3*rand(), collect(values(hist)).-0.15.+0.3*rand(), labels=false, mc=cscheme[2], opacity=0.5)
    #end

    hist_Nvector = countmap(vcat([Nvectors[i,j] for i in 1:noriginals]...))
    mean_hist = Dict(k=>v/noriginals for (k,v) in hist_Nvector)
    plot!(plt, mean_hist, labels=false, linewidth = 4, color=cscheme[2])
    bar!(plt, mean_hist, labels="mean count", linewidth = 0, color=cscheme[2], opacity=0.5)
    Nmean = mean([Nvectors[i,1][1] for i in 1:noriginals])
    L = Nmean/((resolutions[j])^2)
    nvals = collect(0:maximum(collect(keys(mean_hist)))) 
    #plot!(plt,nvals, (resolutions[j])^2*pdf.(Poisson(L),nvals), linewidth=3, linestyle=:dash, color=:red, label="Poisson")
    plot!(plt,nvals, (resolutions[j])^2*pdf.(Poisson(L),nvals), linewidth=3, linestyle=:dash, color=:red, label="Poisson")
    return plt
end

plotlist = [plot_Nvectors(j,Nvectors) for j in 1:nresolutions]
plot(reverse(plotlist)..., layout=(2,4))
plot!(size=(1000,1000/2))
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/Ncount_geyer_100_1.25_R005sat4.pdf")


####### chisquared Test
X² = Array{Float64}(undef, (noriginals,nresolutions))
for i in 1:noriginals, j in 1:nresolutions
    Nvector = Nvectors[i,j]
    Ntot = sum(Nvector)
    Ei = Ntot/(resolutions[j].^2)
    Oi = Nvector
    X²[i,j] = sum((Oi .- Ei).^2 ./ Ei)
end
meanX² = vec(mean(X²,dims=1))
plot(resolutions[2:nresolutions-1].^(-1),
    [meanX²[j] for j in 2:nresolutions-1], labels="X²",
    xlabel="a (side length)",
    ylabel="X²", color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)),
    xlims = (0,1))
vline!([R], linestyle=:dash, labels="R")
vline!([1/sqrt(Nmean)], linestyle=:dash, labels="1/√n")
plot!(resolutions[2:nresolutions-1].^(-1), [mean(Chisq(resolutions[j]^2-1)) for j in 2:nresolutions-1], linestyle=:dash, color=:gray)
plot!(resolutions[2:nresolutions-1].^(-1), [quantile(Chisq(resolutions[j]^2-1),0.1) for j in 2:nresolutions-1],
    fillrange = [quantile(Chisq(resolutions[j]^2-1),0.9) for j in 2:nresolutions-1],
    alpha = 0.2, color=cscheme[4])

#areafrac
plot(areafrac.(resolutions[2:nresolutions-1].^(-1)),
    [meanX²[j] for j in 2:nresolutions-1], labels="X²",
    xlabel="a (side length)",
    ylabel="X²", color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)),
    xlims = (0,1))
vline!(areafrac.([R]), linestyle=:dash, labels="R")
vline!(areafrac.([1/sqrt(Nmean)]), linestyle=:dash, labels="1/√n")
plot!(areafrac.(resolutions[2:nresolutions-1].^(-1)), [mean(Chisq(resolutions[j]^2-1)) for j in 2:nresolutions-1], linestyle=:dash, color=:gray)
plot!(areafrac.(resolutions[2:nresolutions-1].^(-1)), [quantile(Chisq(resolutions[j]^2-1),0.1) for j in 2:nresolutions-1],
    fillrange = [quantile(Chisq(resolutions[j]^2-1),0.9) for j in 2:nresolutions-1],
    alpha = 0.2, color=cscheme[4])
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/X2_geyer_100_1.25_R005sat4.pdf")


plot(areafrac.(resolutions[2:nresolutions].^(-1)),
    [meanX²[j] for j in 2:nresolutions], labels="X²",
    xlabel="a (side length)",
    ylabel="X²", color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)),
    xlims = (0,1), ylims=(0,1500))
vline!(areafrac.([R]), linestyle=:dash, labels="R")
vline!(areafrac.([1/sqrt(Nmean)]), linestyle=:dash, labels="1/√n")
# divide by mean = k^2-1
plot(resolutions[2:nresolutions].^(-1),
    [quantile(Chisq(resolutions[j]^2-1),0.1)/mean(Chisq(resolutions[j]^2-1)) for j in 2:nresolutions],
    fillrange = [quantile(Chisq(resolutions[j]^2-1),0.9)/mean(Chisq(resolutions[j]^2-1)) for j in 2:nresolutions],
    label="80% of χ²-distr. /(k²-1)",
    xlabel="a (side length)",
    ylabel="X²/(k²-1)", color=cscheme[4], alpha=0.4,
    xlims = (0,1))
vline!(areafrac.([R]), linestyle=:dash, labels="R")
vline!(areafrac.([1/sqrt(Nmean)]), linestyle=:dash, labels="1/√n")
hline!([1], linestyle=:dash, label="mean of χ²/(k²-1)" )
plot!(resolutions[2:nresolutions].^(-1),
    [meanX²[j]/mean(Chisq(resolutions[j]^2-1)) for j in 2:nresolutions], 
    color=cscheme[2],
    linewidth=4,
    marker = (3,cscheme[2],5,stroke(1,:black)),
    labels="X²/(k²-1)")
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/X2_geyer_100_1.25_R005sat4.pdf")
    # areafrac
plot(areafrac.(resolutions[2:nresolutions].^(-1)),
    [quantile(Chisq(resolutions[j]^2-1),0.1)/mean(Chisq(resolutions[j]^2-1)) for j in 2:nresolutions],
    fillrange = [quantile(Chisq(resolutions[j]^2-1),0.9)/mean(Chisq(resolutions[j]^2-1)) for j in 2:nresolutions],
    label="80% of χ²-distr. /(k²-1)",
    xlabel="Fc (fraction of contained area)",
    ylabel="X²/(k²-1)", color=cscheme[4], alpha=0.4,
    xlims = (0,1))
vline!(areafrac.([R]), linestyle=:dash, labels="R")
vline!(areafrac.([1/sqrt(Nmean)]), linestyle=:dash, labels="1/√n")
hline!([1], linestyle=:dash, label="mean of χ²/(k²-1)" )
plot!(areafrac.(resolutions[2:nresolutions].^(-1)),
    [meanX²[j]/mean(Chisq(resolutions[j]^2-1)) for j in 2:nresolutions], 
    color=cscheme[2],
    linewidth=4,
    marker = (3,cscheme[2],5,stroke(1,:black)),
    labels="X²/(k²-1)",
    xlabel="Fc (fraction of contained area)")
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/X2_fraction_geyer_100_1.25_R005sat4.pdf")







    # Note p-value probably not best choice
# since it depends also on the number of quadrats
# more quadrats means higher p-value
plot(resolutions[2:nresolutions].^(-1),
    [1-cdf(Chisq(resolutions[j]^2-1),meanX²[j]) for j in 2:nresolutions], labels="p: 1-CDF of Chisq(n²-1) at X²",
    xlabel="a (side length)",
    ylabel="p value", color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)),
    xlims = (0,1),
    ylims = (0,0.5))
vline!([R], linestyle=:dash, labels="R")
vline!([1/sqrt(Nmean)], linestyle=:dash, labels="1/√n")
#
plot(areafrac.(resolutions[2:nresolutions].^(-1)),
    [1-cdf(Chisq(resolutions[j]^2-1),meanX²[j]) for j in 2:nresolutions], labels="p: 1-CDF of Chisq(n²-1) at X²",
    xlabel="fraction of contained area",
    ylabel="p value", color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)),
    xlims = (0,1),
    ylims = (0,0.4))
vline!(areafrac.([R]), linestyle=:dash, labels="R")

#savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/GeyerProcess_chisqtest_pvals.pdf")

plot([1/n for n in resolutions[2:nresolutions]],[meanX²[j]/quantile(Chisq(resolutions[j]^2-1),0.999) for j in 2:nresolutions],
    labels="X²/χ²(k²-1,0.9)",
    xlabel="side length r",
    ylabel="", color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)),
    xlims = (0,1))
vline!([R], linestyle=:dash, labels="R")
vline!([1/sqrt(Nmean)], linestyle=:dash, labels="1/√n")
plot(areafrac.([1/n for n in resolutions[2:nresolutions]]),[meanX²[j]/quantile(Chisq(resolutions[j]^2-1),0.999) for j in 2:nresolutions],
    labels="X²/χ²(n²-1,0.5)",
    xlabel="Fc (fraction of contained area)",
    ylabel="", color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)),
    xlims = (0,1))
vline!(areafrac.([R]), linestyle=:dash, labels="R")
vline!(areafrac.([1/sqrt(Nmean)]), linestyle=:dash, labels="1/√n")





#### Dispersion index
Dmat = Array{Float64}(undef,(noriginals,nresolutions))
for i in 1:noriginals, j in 1:nresolutions 
    Nvector = Nvectors[i,j]
    Dmat[i,j] = var(Nvector)/mean(Nvector)
end
D = vec(mean(Dmat, dims=1))

#D = [mean(var2.(Nvectors[:,j])./mean.(Nvectors[:,j])) for j in 1:nresolutions]
plot([1/n for n in resolutions],D,
    labels="D(r)",
    xlabel="a (side length)",
    ylabel="Dispersion Index",
    color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)),
    xlims = (0,1))
vline!([R], linestyle=:dash, labels="R")
vline!([1/sqrt(Nmean)], linestyle=:dash, labels="1/√n")

plot(areafrac.([1/n for n in resolutions]),D,
    labels="D(r)",
    xlabel="fraction of contained area",
    ylabel="Dispersion Index",
    color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)),
    xlims = (0,1))
vline!(areafrac.([R]), linestyle=:dash, labels="R")
vline!(areafrac.([1/sqrt(Nmean)]), linestyle=:dash, labels="1/√n")

#### Morisita index
function MorisitaIndexMat(Nvectors)
    Mmat = Matrix{Float64}(undef,(noriginals,nresolutions)) 
    for i in 1:noriginals, j in 1:nresolutions
        Nvector = Nvectors[i,j]
        Ntot = sum(Nvector)
        k = resolutions[j]^2
        Mmat[i,j] = k*sum(Nvector[i]*(Nvector[i]-1) for i in eachindex(Nvector))/(Ntot*(Ntot-1))
    end
    return Mmat 
end

Mmat = MorisitaIndexMat(Nvectors)
mean_Mindex = vec(mean(Mmat,dims=1)) 
plot([1/n for n in resolutions], mean_Mindex, 
    labels="M(a)",
    xlabel="a (side length)",
    ylabel="Morisita Index",
    color=cscheme[2],
    linewidth = 4,
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!(areafrac.([R]), linestyle=:dash, labels="R")
vline!(areafrac.([1/sqrt(Nmean)]), linestyle=:dash, labels="1/√n")
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/morisita_geyer_100_1.25_R005sat4.pdf")


plot(areafrac.([1/n for n in resolutions]), mean_Mindex, 
    labels="M(r)",
    xlabel="fraction of contained area",
    ylabel="Morisita Index",
    color=cscheme[2],
    linewidth=4,
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!(areafrac.([R]), linestyle=:dash, labels="R")
vline!(areafrac.([1/sqrt(Nmean)]), linestyle=:dash, labels="1/√n")
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/morisita_fraction_geyer_100_1.25_R005sat4.pdf")





#########################################
# Set up Area interaction process
#########################################
# Parameters
#R = 0.025;
R = 0.05
η = 2.7 # η = γ^(π*R^2)
λ₀ = 500; β = λ₀/η
# Interaction function
C(u::Point,s::PointSet) = FractionOfContestedArea(u,s,R; nd=25)
# Set up exponential model (parameters and covariates)
θ₀ = [log(β),log(η)]
S = [(u,s)->1, (u,s)->-C(u,s)]
# Papangelou conditional intensity
logλ(u,s,θ) = sum(θ[i]*S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
# Define point Gibbs process and sample one pattern
p = GibbsProcess(logλ)
s = sample_pp(Random.GLOBAL_RNG,p,Wplus,niter=60_000)
plot(s,W)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/sample_areaint_185_2.7_R005.pdf")

# Resolutions
resolutions = [1, 2, 4, 8, 16, 32, 64]
boxes = [PointProcessLearning.partitions(W,res) for res in resolutions]
# Create original samples
originals = [PointProcessLearning.sample_pp(Random.GLOBAL_RNG, p,Wplus, niter=100_000) for _ in 1:200]
reconstructed = [PointProcessLearning.reconstruct(s,W,res) for s in originals, res in resolutions]
# Compute various scores
EMDscores = [PointProcessLearning.EMDreconstructed(originals[i],reconstructed[i,j],W,resolutions[j]) for i in eachindex(originals), j in eachindex(resolutions)]
SGscoresOrig = map(s->PointProcessLearning.StoyanGrabarnik(s,logλ,W),originals)
SGscoresRec = map(s->PointProcessLearning.StoyanGrabarnik(s,logλ,W),reconstructed)
ParamsOrig = map(s->EstimateParamsPL_Logit(S,s,W; nd=100),originals)
ParamsRec = map(s->EstimateParamsPL_Logit(S,s,W; nd=100),reconstructed)


# EMD means and variances
EMDmeans = vec(mean(EMDscores, dims=1))
EMDvars = vec(var(EMDscores, dims=1))
# Plot
plot(resolutions.^(-1),EMDmeans, yerr=EMDvars,
    labels="Area interaction process (θ₀=log185, θ₂=log2.7)",
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
    # average distance between two points in square with side a is ≈0.52*a
    # For N points the total distance is N*0.52*1/k
plot!(resolutions.^(-1),Nmean*0.52*resolutions.^(-1), ylims=(0,1.25*maximum(EMDmeans)),
        labels="0.52*N*a (small scale appr.)",
        color=cscheme[2],
        linewidth=2,
        linestyle=:dashdot
)
vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2)    
vline!([R], linestyle=:dot, color=:red, labels="R")
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/EMD_areaint_185_2.7_R005.pdf")




SGscoresRecmeans = vec(mean(SGscoresRec, dims=1))
SGscoresRecvars = vec(var(SGscoresRec, dims=1))
SGscoresOrigmax = maximum(SGscoresOrig)
SGscoresOrigmin = minimum(SGscoresOrig)
SGscoresOrigmean = mean(SGscoresOrig)
SGscoresOrigvar = var(SGscoresOrig)

# Plot SG
plot(resolutions.^(-1),SGscoresRecmeans, yerr=SGscoresRecvars,
    labels="Area interaction process (θ₀=log185, θ₂=log 2.7)",
    xlabel="a (side length)",
    ylabel="Stoyan-Grabarnik residuals",
    color=cscheme[2],
    linewidth=3,
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
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/SG_areaint_185_2.7_R005.pdf")

# Attempt to find bounds for SG
(Nmean-exp(θ₀[1]))/exp(θ₀[1])
(Nmean-exp(θ₀[1]+sat*θ₀[2]))/exp(θ₀[1]+sat*θ₀[2])

theta0orig = map(t->coef(t)[1], ParamsOrig)
theta2orig = map(t->coef(t)[2], ParamsOrig)
theta0rec = map(t->coef(t)[1], ParamsRec)
theta2rec = map(t->coef(t)[2], ParamsRec)

theta0origmean = fill(mean(theta0orig),length(resolutions))
theta0origvar = fill(var(theta0orig),length(resolutions))
theta2origmean = fill(mean(theta2orig),length(resolutions))
theta2origvar = fill(var(theta2orig),length(resolutions))

theta0recmean = vec(mean(theta0rec, dims=1))
theta0recvar = vec(var(theta0rec, dims=1))
theta2recmean = vec(mean(theta2rec, dims=1))
theta2recvar = vec(var(theta2rec, dims=1))

plot(resolutions.^(-1),theta0recmean, yerr=theta0recvar,
    labels="Area interaction process (θ₀=log185, θ₂=log2.7)",
    xlabel="a (side length)",
    ylabel="θ₀ estimate",
    color=cscheme[2],
    linewidth=3,
    legend=:bottomright,
    marker = (3,cscheme[2],5,stroke(1,:black)))
hline!([log(Nmean)], linestyle=:dash, color=cscheme[2], labels="θ₀=log(N) (Poisson approx)")
plot!(resolutions.^(-1), theta0origmean, linestyle=:dash, color=:gray, labels="θ₀ true")
plot!(resolutions.^(-1), theta0origmean.-theta0origvar, 
    fillrange=theta0origmean.+theta0origvar, 
    alpha=0.2,
    color=cscheme[3],
    labels="Variance of original")
vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
vline!([R],labels="R", linestyle=:dot, color=:red)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/theta0_areaint_185_2.7_R005.pdf")

plot(resolutions.^(-1),theta2recmean, yerr=theta2recvar,
    labels="Area interaction process (θ₀=log185, θ₂=log2.7)",
    xlabel="a (side length)",
    ylabel="θ₂ estimate",
    color=cscheme[2],
    linewidth=3,
    legend=:topright,
    marker = (3,cscheme[2],5,stroke(1,:black)))
hline!([0], linestyle=:dash, color=cscheme[2], labels="θ₂=0 (Poisson approx)")
plot!(resolutions.^(-1), theta2origmean, linestyle=:dash, color=:gray, labels="θ₀ true")
plot!(resolutions.^(-1), theta2origmean.-theta2origvar, 
    fillrange=theta2origmean.+theta2origvar, 
    alpha=0.2,
    color=cscheme[3],
    labels="Variance of original")
vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
vline!([R],labels="R", linestyle=:dot, color=:red)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/theta2_areaint_185_2.7_R005.pdf")

# testing plotting against fraction of area
areafrac(a) = areafrac(a,R) + (4*R-2*R^2)*a^2
areafrac(a) = areafrac(a,R)
plot(areafrac.(resolutions.^(-1)),SGscoresRecmeans, yerr=SGscoresRecvars,
    labels="Geyer process (θ₀=log 100, θ₂=log 1.25)",
    xlabel="fraction of contained area",
    ylabel="Stoyan-Grabarnik residuals",
    color=cscheme[2],
    linewidth=3,
    legend=:bottomright,
    marker = (3,cscheme[2],5,stroke(1,:black)))
plot!(areafrac.(resolutions.^(-1)), fill(SGscoresOrigmean-SGscoresOrigvar,length(resolutions)), 
    fillrange=fill(SGscoresOrigmean+SGscoresOrigvar,length(resolutions)), 
    alpha=0.2,
    color=cscheme[3],
    labels="Variance of original")
hline!([0], linestyle=:dash, color=:black, labels="Theory")
vline!(areafrac.([1/sqrt(Nmean)]), linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
vline!(areafrac.([R]),labels="R", linestyle=:dot, color=:red)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/SG_areaint_185_2.7_R005_fraction.pdf")


plot(areafrac.(resolutions.^(-1)),theta0recmean, yerr=theta0recvar,
    labels="Geyer process (θ₀=log 100, θ₂=log 1.25)",
    xlabel="fraction of contained area",
    ylabel="θ₀ estimate",
    color=cscheme[2],
    linewidth=3,
    legend=:topright,
    marker = (3,cscheme[2],5,stroke(1,:black)))
hline!([log(Nmean)], linestyle=:dash, color=cscheme[2], labels="θ₀=log(N) (Poisson approx)")
plot!(areafrac.(resolutions.^(-1)), theta0origmean, linestyle=:dash, color=:gray, labels="θ₀ true")
plot!(areafrac.(resolutions.^(-1)), theta0origmean.-theta0origvar, 
    fillrange=theta0origmean.+theta0origvar, 
    alpha=0.2,
    color=cscheme[3],
    labels="Variance of original")
vline!(areafrac.([1/sqrt(Nmean)]), linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
#vline!(areafrac.([R]),labels="R", linestyle=:dot, color=:red)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/theta0_areaint_185_2.7_R005_fraction.pdf")

plot(areafrac.(resolutions.^(-1)),theta2recmean, yerr=theta2recvar,
    labels="Geyer process (θ₀=log 100, θ₂=log 1.25)",
    xlabel="fraction of contained area",
    ylabel="θ₂ estimate",
    color=cscheme[2],
    linewidth=3,
    legend=:topright,
    marker = (3,cscheme[2],5,stroke(1,:black)))
hline!([0], linestyle=:dash, color=cscheme[2], labels="θ₂=0 (Poisson approx)")
plot!(areafrac.(resolutions.^(-1)), theta2origmean, linestyle=:dash, color=:gray, labels="θ₀ true")
plot!(areafrac.(resolutions.^(-1)), theta2origmean.-theta2origvar, 
    fillrange=theta2origmean.+theta2origvar,
    color=cscheme[3], 
    alpha=0.2,
    labels="Variance of original")
vline!(areafrac.([1/sqrt(Nmean)]), linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
#vline!(areafrac.([R]),labels="R", linestyle=:dot, color=:red)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/theta2_areaint_185_2.7_R005_fraction.pdf")


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
    labels="Area interaction process (θ₀=log185, θ₂=log2.7)",
    xlabel="a (side length)",
    ylabel="RMISE",
    color=cscheme[2],
    linewidth=3,
    legend=:right,
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
vline!([R],labels="R", linestyle=:dot, color=:red)
vline!([R*2*sqrt(2)/(sqrt(2)-1)], linestyle=:dot, color=:orange, labels="Volume")
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/RMISE_areaint_185_2.7_R005.pdf")


plot(areafrac.(resolutions.^(-1)),sqrt.(MISEmean),#, yerr=MISEvar,
    labels="Geyer process (θ₀=log 100, θ₂=log 1.25)",
    xlabel="fraction of contained area",
    ylabel="RMISE",
    color=cscheme[2],
    linewidth=3,
    legend=:right,
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([1/sqrt(Nmean)], linestyle=:dot, color=:black, labels="1/√N", linewidth=2) 
vline!([R],labels="R", linestyle=:dot, color=:red)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Thesis/figure/chapter4/RMISE_areaint_185_2.7_R005_fraction.pdf")
