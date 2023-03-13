## Remember to run Pkg> activate .
using Meshes, Plots, StatsBase, GLM, Combinatorics
import Random
include("../src/PointProcessLearning.jl")
using .PointProcessLearning

import ColorSchemes.okabe_ito
cscheme = okabe_ito

#########################################
# Set up window
#########################################
W = Box((0.,0.),(1.,1.))
Wplus = enlarge(W,0.1)

#########################################
# Set up Area interaction process
#########################################
# Parameters
#R = 0.025;
R = 0.05
η = 2.7 # η = γ^(π*R^2)
λ₀ = 1000; β = λ₀/η
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
s = sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=40_000)
# Check plots
plot(s,W)
StoyanGrabarnikPlot(s,(u,s)->logλ(u,s,θ₀),W,4)
CondIntPlot((u,s) -> logλ(u,s),s,W)

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
s = sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=40_000)
#s = sample_pp(p,Wplus)
plot(s,W)
StoyanGrabarnikPlot(s,(u,s)->logλ(u,s,θ₀),W,4)
CondIntPlot((u,s) -> logλ(u,s),s,W)


#########################################
# Generate original, and reconstructed patterns
#########################################
# Resolutions to be used in reconstruction
resolutions = [1,2,4,8,16,32,64]

noriginals = 30
nresolutions = length(resolutions)
nreconstructions = 10

originalpatterns = Array{PointSet}(undef,noriginals)
reconstructedpatterns = Array{PointSet}(undef,(noriginals,nresolutions,nreconstructions))

for i in 1:noriginals
    s = sample_pp(Random.GLOBAL_RNG,p,W; niter=40_000)
    originalpatterns[i] = s
    for j in 1:nresolutions
        resolution = resolutions[j]
        for k in 1:nreconstructions
            #reconstructedpatterns[i,j,k] = reconstruct(s,W,resolution)
            #reconstructedpatterns[i,j,k] = reconstruct_with_Gibbs(s,W,resolution,logλ)
            boxes = PointProcessLearning.partitions(W,resolution)
            counts = [N(s,box) for box in boxes]
            prec = ReconstructedPointProcess(counts,boxes,p)
            reconstructedpatterns[i,j,k] = sample_pp2(Random.GLOBAL_RNG,prec,W)
        end
    end
end

originalpatterns2 = Array{PointSet}(undef,noriginals)
reconstructedpatterns2 = Array{PointSet}(undef,(noriginals,nresolutions,nreconstructions))

for i in 1:noriginals
    s = sample_pp(Random.GLOBAL_RNG,p,W; niter=40_000)
    originalpatterns2[i] = s
    for j in 1:nresolutions
        resolution = resolutions[j]
        for k in 1:nreconstructions
            reconstructedpatterns2[i,j,k] = reconstruct(s,W,resolution)
            #reconstructedpatterns[i,j,k] = reconstruct_with_Gibbs(s,W,resolution,logλ)
            #boxes = PointProcessLearning.partitions(W,resolution)
            #counts = [N(s,box) for box in boxes]
            #prec = ReconstructedPointProcess(counts,boxes,p)
            #reconstructedpatterns[i,j,k] = sample_pp2(Random.GLOBAL_RNG,prec,W)
        end
    end
end

# Store and load data
using BSON
bson("reconstruct_with_gibbs.bson",Dict(:orig=>originalpatterns, :rec=>reconstructedpatterns))
load_data = BSON.load("reconstruct_with_gibbs.bson")
load_data[:orig]
load_data[:rec]

#########################################
# Some random plots
#########################################
# Plot 3 randomly chosen original patterns, and 1 reconstruced at each resolution
plotmatrix = Array{Plots.Plot}(undef,(3,nresolutions+1))
randomoriginals_idx =  sample(1:noriginals,3, replace=false)
for i in 1:3
    ridx = randomoriginals_idx[i]
    s = originalpatterns[ridx]
    plotmatrix[i,1] = plot(s,W;title="original (sample $ridx)",axis=false,ticks=false)
    for j in 1:nresolutions
        sRec = reconstructedpatterns[ridx,j,rand(1:nreconstructions)]
        res = resolutions[j]
        plt = plot(sRec,W;title="n=$res, (sample $ridx)",axis=false,ticks=false)
        boxes = PointProcessLearning.partitions(W,resolutions[j])
        plot!(PointProcessLearning.Box2Shape.(boxes), linestyle=:dash,fillcolor=false, stroke=:gray, label=false)
        plotmatrix[i,1+nresolutions-j+1] = plt
    end
end
plot(plotmatrix..., layout=(nresolutions+1,3))
plot!(size=(800,2400))
#savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/GeyerProcess_samples_multiscale.pdf")

# Same but using StoyanGrabarnikPlot
for i in 1:3
    ridx = randomoriginals_idx[i]
    s = originalpatterns[ridx]
    #plotmatrix[i,1] = plot(s,W;title="original (sample $ridx)",axis=false,ticks=false)
    plotmatrix[i,1] = StoyanGrabarnikPlot(s,logλ,W,1)
    for j in 1:nresolutions
        sRec = reconstructedpatterns[ridx,j,rand(1:nreconstructions)]
        res = resolutions[j]
        #plotmatrix[i,1+nresolutions-j+1] = StoyanGrabarnikPlot(sRec,logλ,W,res)
        plotmatrix[i,1+nresolutions-j+1] = StoyanGrabarnikPlot(sRec,logλ,W,1)
    end
end
plot(plotmatrix..., layout=(nresolutions+1,3))
plot!(size=(800,2400))


#########################################
# Stoyan Grabarnik Diagnostic
#########################################

originalpatterns_SG = Array{Float64}(undef,noriginals)
reconstructedpatterns_SG = Array{Float64}(undef,(noriginals,nresolutions,nreconstructions))

for i in 1:noriginals
    s = originalpatterns[i]
    originalpatterns_SG[i] = StoyanGrabarnik(s,logλ,W)
    for j in 1:nresolutions
        for k in 1:nreconstructions
            sRec = reconstructedpatterns[i,j,k]
            reconstructedpatterns_SG[i,j,k] = StoyanGrabarnik(sRec,logλ,W)
        end
    end
end

mean(originalpatterns_SG)

originalpatterns2_SG = Array{Float64}(undef,noriginals)
reconstructedpatterns2_SG = Array{Float64}(undef,(noriginals,nresolutions,nreconstructions))

for i in 1:noriginals
    s = originalpatterns2[i]
    originalpatterns2_SG[i] = StoyanGrabarnik(s,logλ,W)
    for j in 1:nresolutions
        for k in 1:nreconstructions
            sRec = reconstructedpatterns2[i,j,k]
            reconstructedpatterns2_SG[i,j,k] = StoyanGrabarnik(sRec,logλ,W)
        end
    end
end


# Plot SG values vs resolution
plt = plot(;title="Stoyan-Grabarnik vs resolution", xticks=(1:nresolutions,resolutions), labels=false)
for i in 1:noriginals
    origseries = repeat([originalpatterns_SG[i]],nresolutions)
    xvals = collect(1:nresolutions).- 0.02 .- 0.2*rand(nresolutions)
    scatter!(plt,xvals,origseries,
            labels=false, marker = (3,cscheme[1],0.5,stroke(0,cscheme[1])))
    for k in 1:nreconstructions
        recseries = reconstructedpatterns_SG[i,:,k]
        xvals = collect(1:nresolutions) .+0.02 .+ 0.2*rand(nresolutions)
        scatter!(plt,xvals, recseries,
                labels=false, marker = (3,cscheme[2],0.5,stroke(0,cscheme[2])))
        # Second series
        recseries = reconstructedpatterns2_SG[i,:,k]
        xvals = collect(1:nresolutions) .+0.22 .+ 0.2*rand(nresolutions)
        scatter!(plt,xvals, recseries,
                labels=false, marker = (3,cscheme[3],0.5,stroke(0,cscheme[3])))
    end
end
plt
origmeans = repeat([mean(originalpatterns_SG)],nresolutions)
xvals = collect(1:nresolutions) .- 0.1
plot!(plt,xvals,origmeans,
    labels="mean SG originals",
    color=cscheme[1],linewidth=3,
    marker = (3,cscheme[1],5,stroke(1,:black)))
recmeans = vec(mean(reconstructedpatterns_SG,dims=(1,3)))
xvals = collect(1:nresolutions) .+ 0.1
plot!(plt,xvals,recmeans,
    labels="mean SG reconstructed",
    color=cscheme[2],linewidth=3,
    marker = (3,cscheme[2],5,stroke(1,:black)))
recmeans = vec(mean(reconstructedpatterns2_SG,dims=(1,3)))
xvals = collect(1:nresolutions) .+ 0.3
plot!(plt,xvals,recmeans,
        labels="mean SG reconstructed (binomial)",
        color=cscheme[3],linewidth=3,
        marker = (3,cscheme[3],5,stroke(1,:black)))
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/GibbsReconstruction_SG_compareBinomial.pdf")


## Plot instead SG difference between reconstructed and original
plt = plot(;title="Stoyan-Grabarnik vs resolution", xticks=(1:nresolutions,resolutions), labels=false)
hline!(plt,[0], linestyle=:dash, color=:red, labels=false)
for i in 1:noriginals
    origseries = repeat([originalpatterns_SG[i]],nresolutions)
    #xvals = collect(1:nresolutions).- 0.02 .- 0.2*rand(nresolutions)
    #scatter!(plt,xvals,origseries,
    #        labels=false, marker = (3,:gray,0.5,stroke(0,:gray)))
    for k in 1:nreconstructions
        recseries = reconstructedpatterns_SG[i,:,k] .-origseries
        xvals = collect(1:nresolutions) .-0.1 .+ 0.2*rand(nresolutions)
        scatter!(plt,xvals, recseries,
                labels=false, marker = (3,cscheme[2],0.5,stroke(0,cscheme[2])))
    end
end
reconstructedpatterns_SG
SG_diff = reconstructedpatterns_SG .- originalpatterns_SG
mean_SG_diff = vec(mean(SG_diff,dims=(1,3)))
sd_SG_diff = sqrt.(vec(var(SG_diff,dims=(1,3))))
plot!(plt, 1:nresolutions, mean_SG_diff, 
    labels="SG diff (reconstructed - original)", color=cscheme[2],
    marker = (3,:black,5,stroke(1,:black)), yerror= sd_SG_diff)      
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/GibbsReconstruction_SGdiff.pdf")


# Plot against length of side of quadrats
plot([1/n for n in resolutions], mean_SG_diff, xlabel="length of side of quadrats",
    labels="SG diff (reconstructed - original)", color=cscheme[2],
    marker = (3,:black,5,stroke(1,:black)), yerror= sd_SG_diff) 
vline!([R], color=:red, linestyle=:dash)


# Compute and plot EMD




#########################################
# Analyze Ni distributions
#########################################

Nvectors = Array{Vector{Int64}}(undef, (noriginals,nresolutions))
for i in 1:noriginals, j in 1:nresolutions
    s = originalpatterns[i]
    boxes = PointProcessLearning.partitions(W,resolutions[j])
    Nvectors[i,j] = [N(s,box) for box in boxes]
end

function plot_Nvectors(j::Int64,Nvectors::Array{Vector{Int64}})
    res = resolutions[j]
    plt = plot(;title="Distr. of N_i (n=$res)",
            xlabel="Number of points", ylabel="occurrences")
    for i in 1:noriginals
        hist = countmap(Nvectors[i,j])
        scatter!(plt, collect(keys(hist)).-0.15.+0.3*rand(), collect(values(hist)).-0.15.+0.3*rand(), labels=false, mc=cscheme[2], opacity=0.5)
    end

    hist_Nvector = countmap(vcat([Nvectors[i,j] for i in 1:noriginals]...))
    mean_hist = Dict(k=>v/noriginals for (k,v) in hist_Nvector)
    plot!(plt, mean_hist, labels="mean count", linewidth = 4, color=cscheme[2])
    Nmean = mean([Nvectors[i,1][1] for i in 1:noriginals])
    L = Nmean/((resolutions[j])^2)
    nvals = collect(0:maximum(collect(keys(mean_hist)))) 
    plot!(plt,nvals, (resolutions[j])^2*pdf.(Poisson(L),nvals), linewidth=3, linestyle=:dash, color=:red, label="Poisson")
    return plt
end

plotlist = [plot_Nvectors(j,Nvectors) for j in 1:nresolutions]
plot(reverse(plotlist)..., layout=(nresolutions,1))
plot!(size=(2700/7,2000))
#savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/GeyerProcess_Ndistribution.pdf")



####### chisquared Test
X² = Array{Float64}(undef, (noriginals,nresolutions))
for i in 1:noriginals, j in 1:nresolutions
    Nvector = Nvectors[i,j]
    Ntot = sum(Nvector)
    Ei = Ntot/measure(W)
    Oi = Nvector
    X²[i,j] = sum((Oi .- Ei).^2 ./ Ei)
end
meanX² = vec(mean(X²,dims=1))
plot([1/n for n in resolutions[2:nresolutions]],
    [1-cdf(Chisq(resolutions[j]^2-1),meanX²[j]) for j in 2:nresolutions], labels="p: 1-CDF of Chisq(n²-1) at X²",
    xlabel="side length r",
    ylabel="p value", color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([R,2*R], linestyle=:dash, labels="R,2*R")
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/GeyerProcess_chisqtest_pvals.pdf")

plot([1/n for n in resolutions[2:nresolutions]],[meanX²[j]/quantile(Chisq(resolutions[j]^2-1),0.99) for j in 2:nresolutions],
    labels="X²/χ²(n²-1,0.99)",
    xlabel="side length r",
    ylabel="", color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([R,2*R], linestyle=:dash, labels="R,2*R")
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/GeyerProcess_chisqtest_X2overChi2.pdf")



jval = 4
plot_Nvectors(jval,Nvectors)
nvals = collect(0:20)
Nmean = mean([Nvectors[i,1][1] for i in 1:noriginals])
L = Nmean/((resolutions[jval])^2) 
pvals = [L^n*exp(-L)/factorial(n) for n in nvals]
plot!(nvals,(resolutions[jval])^2*pvals, linewidth=3, linestyle=:dash, color=:red, label="Poisson")




L
# Expectation of mean of Ni.... for poisson: E(mean(Ni))=L
m = mean(mean.(Nvectors[:,jval]))
# Expectation of mean of Ni.... for poisson: E(var(Ni))=L
v = mean(var.(Nvectors[:,jval]))
# So generally, overdispersed
D = v/m

var2(x) = sum((x .- mean(x)).^2)/(length(x)-1)

Dmat = Array{Float64}(undef,(noriginals,nresolutions))
for i in 1:noriginals, j in 1:nresolutions 
    #Nvector = Nvectors[i,j]
    Nvector = poisNvectors[i,j]
    Dmat[i,j] = var(Nvector)/mean(Nvector)
end
D = vec(mean(Dmat, dims=1))

#D = [mean(var2.(Nvectors[:,j])./mean.(Nvectors[:,j])) for j in 1:nresolutions]
plot(resolutions,D)
plot([1/n for n in resolutions],D,
    labels="D(r)",
    xlabel="side length r",
    ylabel="Dispersion Index",
    color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([R,2*R], linestyle=:dash, labels="R,2*R")
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/GeyerProcess_disperson_index_raw.pdf")

plot!([1/n for n in resolutions],D,
    labels="D(r) pois",
    xlabel="side length r",
    ylabel="Dispersion Index",
    color=cscheme[1],
    marker = (3,cscheme[1],5,stroke(1,:black)))
vline!([R,2*R], linestyle=:dash, labels="R,2*R")





plot([1/n for n in resolutions],[(resolutions[j]^2-1)*D[j] for j in 1:nresolutions])
plot([1/n for n in resolutions[2:nresolutions]],[(resolutions[j]^2-1)*D[j]/quantile(Chisq(resolutions[j]^2-1),0.99) for j in 2:nresolutions])
vline!([R,2*R])

# Morisita index
#Nvector = Nvectors[1,jval,1]
#Ntot = sum(Nvector)
#k = resolutions[jval]
#Mindex = k*sum(Nvector[i]*(Nvector[i]-1) for i in eachindex(Nvector))/(Ntot*(Ntot-1))

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
    labels="M(r)",
    xlabel="side length r",
    ylabel="Morisita Index",
    color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([R,2*R], linestyle=:dash, labels="R,2R")
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/GeyerProcess_morisita_index_raw.pdf")


#######   SEE Illians book p88-89 for better method, using grouping of quadrats. exactly what we need!
# Method uses chisquare comparison but at many different scales at the same time
# THis method allows to identify the scale at which clustering is apparent (if resolution is high enough)
# The strength of the deviation could also indicate interaction parameters (provided we are constrained to a specific family of models)
# Maybe we can use simulation studies to guess interaction parameters (given model family)
# We could then use this information to more accurately reconstruct the point pattern

function partitions(W::Box,dim::Tuple{Int64,Int64})
    lo, up = coordinates.(extrema(W))
    dx = (up[1]-lo[1])/dim[1]
    dy = (up[2]-lo[2])/dim[2]
    boxmat = [Box((lo[1]+k*dx,lo[2]+j*dy),(lo[1]+(k+1)*dx,lo[2]+(j+1)*dy)) for k=0:(dim[1]-1), j=0:(dim[2]-1)]
    return reshape(boxmat,dim[1]*dim[2])
end


sboxes = partitions(W,(nmax,nmax))
lboxes = partitions(W,(nmax,nmax/2))



function GreigSmith(s::PointSet,W::Box,nmax::Int)
    q = Int(log2(nmax^2))
    svec = Array{Float64}(undef,q-1)
    Nvec = Array{Float64}(undef,q-1)
    n = nmax
    m = nmax
    #testvec = Array{Tuple{Int,Int}}(undef,q-1)
    for i in 1:q-1
        # inner boxes
        Ninner = [N(s,box) for box in partitions(W,(n,m))]
        Nvec[i] = mean(Ninner)
        # outer boxes
        n = (i%2==0) ? Int(n/2) : n
        m = (i%2==0) ? n : Int(n/2)
        Nouter = [N(s,box) for box in partitions(W,(n,m))]
        svec[i] = sum(Ninner.^2) - 0.5*sum(Nouter.^2)
    end
    Ivec = svec./Nvec
    Xvec = Ivec./[2^(q-i) for i in 1:q-1]
    return svec, Ivec, Xvec
end

svec,Ivec,Xvec = GreigSmith(s,W,64)
plot(svec)
plot(Ivec)
plot(Ivec./[quantile(Chisq(2^(q-i)),0.5) for i in 1:q-1])
plot(Xvec./[quantile(Chisq(2^(q-i)),0.5) for i in 1:q-1])

q = Int(log2(64^2))
plot([1/(2^(q-i)) for i in 1:q-1], Xvec)
plot([(i+1)/64 for i in 1:q-1], Xvec)
vline!([R,2*R])

[2^(4-i) for i in 1:4-1]


jval = 7
q = log2(res*res)
Nvector = Nvectors[1,jval]


Nmatrix = reshape(Nvector,(res,res))
N2matrix = [Nmatrix[i,j] + Nmatrix[i+1,j] for i in 1:2:res-1, j in 1:res]
S1 = sum(Nmatrix.^2) - (1/2)*sum(N2matrix.^2)
I1 = S1/mean(Nvector)
X1 = I1/(2^(q-1))
N4matrix = [N2matrix[i,j] + N2matrix[i,j+1] for i in 1:Int(res/2), j in 1:2:res-1]
S2 = sum(N2Matrix.^2) - (1/2)*sum(N4matrix.^2)
I2 = S2/mean(Nvector)
X2 = I2/(2^(q-2))
N8matrix = [N4matrix[i,j] + N4matrix[i+1,j] for i in 1:2:Int(res/2)-1, j in 1:Int(res/2)]
S3 = sum(N4Matrix.^2) - (1/2)*sum(N8matrix.^2)
I3 = S3/mean(Nvector)
X3 = I3/(2^(q-3))
N16matrix = [N8matrix[i,j] + N8matrix[i,j+1] for i in 1:Int(res/4), j in 1:2:Int(res/2)-1]
S4 = sum(N8matrix.^2) - (1/2)*sum(N16matrix.^2)
I4 = S4/mean(Nvector)
X4 = I4/(2^(q-4))
N32matrix = [N16matrix[i,j] + N16matrix[i+1,j] for i in 1:2:Int(res/4)-1, j in 1:Int(res/4)]
S5 = sum(N16matrix.^2) - (1/2)*sum(N32matrix.^2)
I5 = S5/mean(Nvector)
X5 = I5/(2^(q-5))
N64matrix = [N32matrix[i,j] + N32matrix[i,j+1] for i in 1:Int(res/8), j in 1:2:Int(res/4)-1]
S6 = sum(N32matrix.^2) - (1/2)*sum(N64matrix.^2)
I6 = S6/mean(Nvector)
X6 = I6/(2^(q-6))
plot([X1,X2,X3,X4,X5,X6])
I = [I1,I2,I3,I4,I5,I6]
X = [X1,X2,X3,X4,X5,X6]
[I[i]/quantile(Chisq(2^(q-i)),0.5) for i in eachindex(I)]
plot([i/res for i in eachindex(I)], [X[i]/quantile(Chisq(2^(q-i)),0.5) for i in eachindex(I)], labels="Iᵢ/χ²(2^(q-1),0.5)")
vline!([R,2*R], labels="R*resolution", color=:red, linestyle=:dash)

4/res
3/res
2/res
1/res
R
R*res

using Distributions
p1 = 1-cdf(Chisq(2^(q-1)),I1)
p2 = 1-cdf(Chisq(2^(q-2)),I2)
p3 = 1-cdf(Chisq(2^(q-3)),I3)
p4 = 1-cdf(Chisq(2^(q-4)),I4)
p5 = 1-cdf(Chisq(2^(q-5)),I5)
p6 = 1-cdf(Chisq(2^(q-6)),I6)
plot([p1,p2,p3,p4,p5,p6])
vline!([R*res])

1-cdf(Chisq(6),1.63)


plot(s,W)
boxes = PointProcessLearning.partitions(W,resolutions[4])
plot!(PointProcessLearning.Box2Shape.(boxes), linestyle=:dash,fillcolor=false, stroke=:black, label=false)


#########################################
# Compute covariance / correlation matrix
#########################################

jval = 3
Cmat = StatsBase.cor(hcat(Nvectors[:,jval]...),dims=2)
heatmap(Cmat)

# use algo to get nearest neighbours of i, then compute correlation with those

# Adjacency matrix of n by m grid 
function adjacencymat(n::Int,m::Int)
    # dummy matrix....ugly hack
    B = Matrix{Int64}(undef,(n,m))
    x = reshape(CartesianIndices(B),n*m)
    A = Matrix{Int64}(undef,(length(x),length(x)))
    for i in eachindex(x), j in eachindex(x)
        dif = x[i]-x[j]
        A[i,j] = abs(dif[1])+abs(dif[2]) == 1 ? 1 : 0
    end
    return A
end


A = adjacencymat(resolutions[jval],resolutions[jval])

function MoranIndex(x)
    n = length(x)
    A = adjacencymat(Int(sqrt(n)),Int(sqrt(n)))
    xbar = mean(x)
    val = sum( A[i,j]*(x[i]-xbar)*(x[j]-xbar)  for i in eachindex(x), j in eachindex(x))
    val = val/sum(A)
    val = val*n/sum((x.-xbar).^2)
    return val
end

jval = 2
Nvector = Nvectors[4,jval]
MoranIndex(Nvector)

mean(MoranIndex.(Nvectors[:,jval]))

MIvals = [mean(MoranIndex.(Nvectors[:,j])) for j in 1:nresolutions]
plot(resolutions,MIvals)
plot([1/res for res in resolutions],MIvals,
    labels="M(r)",
    xlabel="side length r",
    ylabel="Moran Index",
    color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([R,2*R], linestyle=:dash, labels="R,2R")




#########################################
# Entropy measure
#########################################

function EntropyOneStep(s::PointSet, b::Box, n::Int)
    largerboxes = PointProcessLearning.partitions(b,n)
    entropyvector = Array{Float64}(undef,n^2)
    for i in 1:n^2
        lbox = largerboxes[i]
        Ntot = N(s,lbox)
        if Ntot>0
            Nvals = [N(s,sbox) for sbox in PointProcessLearning.partitions(lbox,2)]
            pvals = Nvals/Ntot
            entropyvector[i] = -sum(p>0 ? p*log(p) : 0 for p in pvals)
        else 
            entropyvector[i] = 0
        end
    end
    return mean(entropyvector)
end

EntropyOneStep(originalpatterns[2],W,resolutions[4])
Emat = Array{Float64}(undef,(noriginals,nresolutions))
for i in 1:noriginals, j in 1:nresolutions
    Emat[i,j] = EntropyOneStep(originalpatterns[i],W,resolutions[j])
end
Evals = vec(mean(Emat, dims=1))
plot([1/(2*n) for n in resolutions[1:nresolutions]],Evals, 
    labels="Entropy",
    xlabel="side length r",
    ylabel="Entropy",
    color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([R,2*R], labels="R,2R", color=:red, linestyle=:dash)

function Entropy(s::PointSet,b::Box,n::Int64)
    Ntot = N(s,b)
    entropy = 0 
    if Ntot>0
        Nvals = [N(s,box) for box in PointProcessLearning.partitions(b,n)]
        pvals = Nvals/Ntot
        entropy = -sum(p>0 ? p*log(p) : 0 for p in pvals)
    end
    return entropy
end

Emat = Array{Float64}(undef,(noriginals,nresolutions))
for i in 1:noriginals, j in 1:nresolutions
    Emat[i,j] = Entropy(originalpatterns[i],W,resolutions[j])
end
Evals = vec(mean(Emat, dims=1))
plot([1/(2*n) for n in resolutions[1:nresolutions]],Evals, 
    labels="Entropy",
    xlabel="side length r",
    ylabel="Entropy Index",
    color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([R,2*R], labels="R,2R", color=:red, linestyle=:dash)

plot([1/(2*n) for n in resolutions[1:nresolutions]],[Evals[i]/log(resolutions[i]^2) for i in 1:nresolutions], 
    labels="Entropy Index",
    xlabel="side length r",
    ylabel="Entropy Index",
    color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([R,2*R], labels="R,2R", color=:red, linestyle=:dash)



#########################################
# Earth mover distance
#########################################

EMDscores = Array{Float64}(undef,(noriginals,nresolutions,nreconstructions))
for i in 1:noriginals
    println("on original pattern number $i of $noriginals")
    s1 = originalpatterns[i]
    for j in 1:nresolutions
        for k in 1:nreconstructions
            s2 = reconstructedpatterns[i,j,k]
            res = resolutions[j]
            EMDscores[i,j,k] = EMDreconstructed(s1,s2,W,res)
        end
    end
end

EMDscores2 = Array{Float64}(undef,(noriginals,nresolutions,nreconstructions))
for i in 1:noriginals
    println("on original pattern number $i of $noriginals")
    s1 = originalpatterns2[i]
    for j in 1:nresolutions
        for k in 1:nreconstructions
            s2 = reconstructedpatterns2[i,j,k]
            res = resolutions[j]
            EMDscores2[i,j,k] = EMDreconstructed(s1,s2,W,res)
        end
    end
end

plt = plot(;title="EMD(x,x̃) vs resolution", xticks=(1:nresolutions,resolutions), labels=false)
hline!(plt,[0], linestyle=:dash, color=:red, labels=false)
for i in 1:noriginals
    for k in 1:nreconstructions
        recseries = EMDscores[i,:,k]
        xvals = collect(1:nresolutions) .-0.2 .+ 0.4*rand(nresolutions)
        scatter!(plt,xvals, recseries,
                labels=false, marker = (3,cscheme[2],0.5,stroke(0,cscheme[2])))
        recseries = EMDscores2[i,:,k]
        xvals = collect(1:nresolutions) .-0.0 .+ 0.4*rand(nresolutions)
        scatter!(plt,xvals, recseries,
        labels=false, marker = (3,cscheme[3],0.5,stroke(0,cscheme[3])))
    end
end
plt
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/GibbsReconstruction_EMD_compareBinomial.pdf")


mean_EMD = vec(mean(EMDscores,dims=(1,3)))
sd_EMD = sqrt.(vec(var(EMDscores,dims=(1,3))))
plot!(plt, 1:nresolutions, mean_EMD, 
    labels="mean EMD", color=cscheme[2],
    marker = (3,:black,5,stroke(1,:black)), yerror= sd_EMD) 
mean_EMD2 = vec(mean(EMDscores2,dims=(1,3)))
sd_EMD2 = sqrt.(vec(var(EMDscores2,dims=(1,3))))
plot!(plt, 1:nresolutions, mean_EMD2, 
    labels="mean EMD", color=cscheme[2],
    marker = (3,:black,5,stroke(1,:black)), yerror= sd_EMD2) 
    
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/GibbsReconstruction_EMD.pdf")






paramsOrig = Array{Vector}(undef,noriginals)
paramsRec = Array{Vector}(undef,(noriginals,nresolutions,nreconstructions))
for i in 1:noriginals
    sOrig = originalpatterns[i]
    nd = suggest_nd(sOrig,W)
    paramsOrig[i] = coef(EstimateParamsPL_Logit(S,sOrig,W; nd=nd))
    for j in 1:nresolutions
        for k in 1:nreconstructions
            sRec = reconstructedpatterns[i,j,k]
            paramsRec[i,j,k] = coef(EstimateParamsPL_Logit(S,sRec,W; nd=nd))
        end
    end
end

paramsOrig2 = Array{Vector}(undef,noriginals)
paramsRec2 = Array{Vector}(undef,(noriginals,nresolutions,nreconstructions))
for i in 1:noriginals
    sOrig = originalpatterns2[i]
    nd = suggest_nd(sOrig,W)
    paramsOrig2[i] = coef(EstimateParamsPL_Logit(S,sOrig,W; nd=nd))
    for j in 1:nresolutions
        for k in 1:nreconstructions
            sRec = reconstructedpatterns2[i,j,k]
            paramsRec2[i,j,k] = coef(EstimateParamsPL_Logit(S,sRec,W; nd=nd))
        end
    end
end


plt = [plot(xticks=(1:nresolutions,resolutions),legend=:topright),plot(xticks=(1:nresolutions,resolutions),legend=:bottomright)]
for i in 1:noriginals
    xvals = collect(1:nresolutions) .+ 0.05 .+ 0.2*rand(nresolutions)
    scatter!(plt[1],xvals,repeat([paramsOrig[i][1]],nresolutions),
            labels=false, marker = (3,cscheme[1],0.5,stroke(0,cscheme[1])))
    scatter!(plt[2],xvals,repeat([paramsOrig[i][2]],nresolutions),
            labels=false, marker = (3,cscheme[1],0.5,stroke(0,cscheme[1])))
    for k in 1:nreconstructions
        xvals = collect(1:nresolutions) .- 0.05 .- 0.2*rand(nresolutions)
        pars = map(x->x[1],paramsRec[i,:,k])
        scatter!(plt[1],xvals,pars,
                labels=false, marker = (3,cscheme[2],0.5,stroke(0,cscheme[2])))
        pars = map(x->x[2],paramsRec[i,:,k])
        scatter!(plt[2],xvals,pars,
                labels=false, marker = (3,cscheme[2],0.5,stroke(0,cscheme[2])))
    end
end

xvals  = collect(1:nresolutions) .+ 0.05 .+ 0.2*rand(nresolutions)
origmeans1 = mean(map(x->x[1],paramsOrig))
plot!(plt[1], xvals, repeat([origmeans1],nresolutions), labels="Mean est. θ₁ (original)", 
    color=cscheme[1],linewidth=3, marker = (3,cscheme[1],5,stroke(1,:black)))
origmeans2 = mean(map(x->x[2],paramsOrig))
plot!(plt[2], xvals, repeat([origmeans2],nresolutions), labels="Mean est. θ₂ (original)",
    color=cscheme[1],linewidth=3, marker = (3,cscheme[1],5,stroke(1,:black)))
xvals  = collect(1:nresolutions) .- 0.05 .- 0.2*rand(nresolutions)
recmeans1 = [mean(map(x->x[1],paramsRec[:,j,:])) for j=1:nresolutions]
plot!(plt[1], xvals, recmeans1, labels="Mean est. θ₁ (reconstructed)",
        color=cscheme[2],linewidth=3, marker = (3,cscheme[2],5,stroke(1,:black)))
recmeans2 = [mean(map(x->x[2],paramsRec[:,j,:])) for j=1:nresolutions]
plot!(plt[2], xvals, recmeans2, labels="Mean est. θ₂ (reconstructed)",
        color=cscheme[2],linewidth=3, marker = (3,cscheme[2],5,stroke(1,:black)))

plot!(plt[1],[0,3],[log(mean(map(x->x[1],Nvectors[:,1]))),log(mean(map(x->x[1],Nvectors[:,1])))], 
    linestyle=:dash, color=cscheme[2], labels="θ̃₁ = log(N/|W|)")
plot!(plt[2],[0,3],[0,0], linestyle=:dash, color=cscheme[2], labels="θ̃₂=0")
hline!(plt[1],[θ₀[1]], linestyle=:dash, color=cscheme[1], labels="True θ₁")
hline!(plt[2],[θ₀[2]], linestyle=:dash, color=cscheme[1], labels="True θ₂")

plot(plt..., layout=(2,1))
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/GibbsReconstruction_parameterestimates_vs_resolution.pdf")


# Now plot against r = 1/res

#plt = [plot(xticks=(1:nresolutions,resolutions),legend=:topright),plot(xticks=(1:nresolutions,resolutions),legend=:bottomright)]
plt = [plot(legend=:right,xlabel="r",ylabel="θ₁"),plot(legend=:right, xlabel="r",ylabel="θ₂")]

xvals  = [1/n for n in resolutions]
origmeans1 = mean(map(x->x[1],paramsOrig))
plot!(plt[1], xvals, repeat([origmeans1],nresolutions), labels="Mean est. θ₁ (original)", 
    color=cscheme[1],linewidth=3, marker = (3,cscheme[1],5,stroke(1,:black)))
origmeans2 = mean(map(x->x[2],paramsOrig))
plot!(plt[2], xvals, repeat([origmeans2],nresolutions), labels="Mean est. θ₂ (original)",
    color=cscheme[1],linewidth=3, marker = (3,cscheme[1],5,stroke(1,:black)))
recmeans1 = [mean(map(x->x[1],paramsRec[:,j,:])) for j=1:nresolutions]
plot!(plt[1], xvals, recmeans1, labels="Mean est. θ₁ (reconstructed)",
        color=cscheme[2],linewidth=3, marker = (3,cscheme[2],5,stroke(1,:black)))
recmeans2 = [mean(map(x->x[2],paramsRec[:,j,:])) for j=1:nresolutions]
plot!(plt[2], xvals, recmeans2, labels="Mean est. θ₂ (reconstructed)",
        color=cscheme[2],linewidth=3, marker = (3,cscheme[2],5,stroke(1,:black)))


        xvals  = [1/n for n in resolutions]
        recmeans1 = [mean(map(x->x[1],paramsRec2[:,j,:])) for j=1:nresolutions]
        plot!(plt[1], xvals, recmeans1, labels="Mean est. θ₁ (reconstructed,Binom)",
                color=cscheme[3],linewidth=3, marker = (3,cscheme[3],5,stroke(1,:black)))
        recmeans2 = [mean(map(x->x[2],paramsRec2[:,j,:])) for j=1:nresolutions]
        plot!(plt[2], xvals, recmeans2, labels="Mean est. θ₂ (reconstructed,Binom)",
                color=cscheme[3],linewidth=3, marker = (3,cscheme[3],5,stroke(1,:black)))
        


hline!(plt[1],[log(mean(map(x->x[1],Nvectors[:,1])))], 
    linestyle=:dash, color=cscheme[2], labels="θ̃₁ = log(N/|W|)")
hline!(plt[2],[0], linestyle=:dash, color=cscheme[2], labels="θ̃₂=0")
hline!(plt[1],[θ₀[1]], linestyle=:dash, color=cscheme[1], labels="True θ₁")
hline!(plt[2],[θ₀[2]], linestyle=:dash, color=cscheme[1], labels="True θ₂")

vline!(plt[1],[R], linestyle=:dash, color=:black, labels="r=R")
vline!(plt[2],[R], linestyle=:dash, color=:black, labels="r=R")
plot(plt..., layout=(2,1))
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/GibbsReconstruction_parameterestimates_compareBinomial.pdf")





#########################################
# Finite Gibbs Point Process fixex N
#########################################


"""
   FiniteGibbsProcess(logλ,N)
A Finite (fixed N) Gibbs Process with Papangelou conditional log intensity 'logλ' and N points.
"""
struct FiniteGibbsProcess2{logL<:Function,N<:Int} <: PointProcess
    logλ::logL
    n::N
end

using Distributions
# Birth death move
function sample_pp2(rng::Random.AbstractRNG,
    p::FiniteGibbsProcess2, 
    b::Box; niter=10_000, progress=false)

    # region configuration
    lo, up = coordinates.(extrema(b))
    V = measure(b)

    # Number of points
    N(S) = length(S.items)

    # product of uniform distributions
    U = product_distribution([Uniform(lo[i], up[i]) for i in 1:embeddim(b)])
    I(n) = DiscreteUniform(1,n) 

    # probabilities
    logλ(u,S) = N(S)>0 ? p.logλ(u,S) : 0
    r(u,S) = exp(logλ(u,S)+log(V)-log(N(S)+1))
    pbirth = 0
    pmove = 1
    rmove(u,ξ,S₋) = pmove*exp(logλ(u,S₋)-logλ(ξ,S₋))/(N(S₋)+1)

    # Initial sample
    S = sample_pp(BinomialProcess(p.n),b)
    #println("Performing ",niter," steps")
    for m in 1:niter
        if progress
            if mod(m,100)==0
                println("step ",m)
            end
        end
        Sprop = deepcopy(S)
        if rand() < pmove
            # Randomly select and move point
            i = rand(I(N(S)))
            ξ = Sprop.items[i]
            deleteat!(Sprop.items,i)
            u = Point(rand(U))
            # accept/reject move
            S = rand()<rmove(u,ξ,Sprop) ? Sprop ∪ u : S  # EDIT THIS!
        elseif rand() < pbirth
            # Randomly generate new point
            u = Point(rand(U))
            Sprop = Sprop ∪ u
            # accept/reject birth
            S = rand()<r(u,S) ? Sprop : S 
        elseif N(S)>1
            # randomly select and remove point
            i = rand(I(N(S)))
            deleteat!(Sprop.items,i)
            u = S.items[i]
            # accept/reject death
            S = rand()<1/r(u,Sprop) ? Sprop : S 
        end
    end
    return S
end

p = FiniteGibbsProcess2(logλ,200)
s = sample_pp2(Random.GLOBAL_RNG,p,W)
p1 = plot(s,W)

srec = reconstruct_with_Gibbs(s,W,5,logλ)
p2 = plot(srec,W)
boxes = PointProcessLearning.partitions(W,5)
plot!(p2,PointProcessLearning.Box2Shape.(boxes), linestyle=:dash,fillcolor=false, stroke=:gray, label=false)

plot(p1,p2)

function reconstruct_with_Gibbs(s::PointSet,W::Box,n::Int,logλ::Function)
    boxes = PointProcessLearning.partitions(W,n)
    snew = EmptyPointSet()
    n = N(s,boxes[1])
    for b in boxes
        n = N(s,b)
        if n>0
            p = FiniteGibbsProcess2(logλ,n)
            snew = snew ∪ sample_pp2(Random.GLOBAL_RNG,p,b)
        end
    end
    return snew
end