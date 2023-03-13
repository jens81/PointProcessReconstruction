## Remember to run Pkg> activate .
using Meshes
using Plots
import Random
using StatsBase
using GLM
include("../src/PointProcessLearning.jl")
using .PointProcessLearning

# Set up window
W = Box((0.,0.),(1.,1.))
Wplus = Box((-0.1,-0.1),(1.1,1.1))

# Inhomogeneous Point Process
λ₃(x::Point) = exp(3.5+2*coordinates(x)[1])
p3 = PoissonProcess(λ₃)
s3 = sample_pp(p3,W)
plot(s3,W)

s̃3 = reconstruct(s3,W,2)
plot(s̃3,W)
StoyanGrabarnikPlot(s3, (u,s) -> log(λ₃(u)), W, 2)
StoyanGrabarnikPlot(s̃3, (u,s) -> log(λ₃(u)), W, 2)





stest = sample_pp(p3,W)
logλ3 = (u,s) -> exp(λ₃(u))
OrigScore = StoyanGrabarnik(stest,logλ3,W)
# Loop over all resolutions
nres = 15
nrec = 20
scoremat = zeros(nres,nrec)
for res in 1:nres
    for rec in 1:nrec
        recpatt = reconstruct(stest,W,res)
        scoremat[res,rec] = StoyanGrabarnik(recpatt,logλ3,W) -OrigScore
    end
end

using StatsBase
means = reshape(mean(scoremat, dims=2),nres)
vars = reshape(var(scoremat, dims=2),nres)

plot(1:nres,means, label="Mean")
plot!(Shape(vcat(collect(1:nres),reverse(collect(1:nres))),vcat(means + vars, reverse(means -vars))), opacity=0.3, label="Variance")



# Strauss model
λ₀ = 200
γ = 0.6
R = 0.05
logλ4(u::Point,s::PointSet) = log(λ₀) + log(γ)*(NNdist(s ∪ u,R) - NNdist(s,R))
p4 = GibbsProcess(logλ4)
s4 = sample_pp(p4,W)
plot(s4,W)

# Loop over all resolutions
nres = 50
npat = 20
nrec = 20
scoremat = zeros(nres,nrec,npat)

for res in 1:nres
    for pat in 1:npat
        stest = sample_pp(p4,W)
        OrigScore = StoyanGrabarnik(stest,logλ,W)
        for rec in 1:nrec
            recpatt = reconstruct(stest,W,res)
            scoremat[res,pat,rec] = StoyanGrabarnik(recpatt,logλ,W) -OrigScore
        end
    end
end

means = reshape(mean(scoremat, dims=(2,3)),nres)
vars = reshape(var(scoremat, dims=(2,3)),nres)
plot(1:nres,means, label="Mean")
plot!(Shape(vcat(collect(1:nres),reverse(collect(1:nres))),vcat(means + vars, reverse(means -vars))), opacity=0.3, label="Variance")

means2 = reshape(mean(scoremat,dims=3),(nres,npat))
scatter(1:nres, means2, legend=false, mc=:gray)
plot!(1:nres,means, label="Mean", lw=5)

scores = reshape(scoremat,(nres,npat*nrec))
scatter(1:nres, scores, legend=false, marker = (2,:gray,0.2,stroke(0,:gray)))
hline!([0.], color=:black, linestyle=:dash)
plot!(1:nres,means, label="Mean", lw=5, color=:brown)




###########################
# Gibbs Point Process (Geyer)
# Distance between points
R = 0.03; sat=4;
t(u::Point,s::PointSet) = sum(dist(u,x)<R for x in s.items)
# Parameters
λ₀ = 350; g₀ = 1.4;
θ₀ = [log(λ₀),log(g₀)]
S = [(u,s)->1, (u,s)->min(sat,t(u,s))]
# Generate patter
logλ(u,s,θ) = sum(θ[i]S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
p = GibbsProcess(logλ)
s = sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=20_000)
#s = sample_pp(p,Wplus)
plot(s,W)
CondIntPlot(logλ,s,W;N=100)
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/sample_geyer_l350_g1.4_R0.03_sat4_full.pdf")
StoyanGrabarnik(s,(u,s)->logλ(u,s,θ₀),W)
# Sample reconstruction

# Loop over all resolutions
resolutions = [1,2,4,8,16,32,64,128]
nres = length(resolutions)
npat = 30
nrec = 10
scorematOrig = zeros(Union{Float64,Missing},nres,npat,nrec)
θmatOrig = zeros(Union{Float64,Missing},2,nres,npat,nrec)
scorematRec = zeros(Union{Float64,Missing},nres,npat,nrec)
θmatRec = zeros(Union{Float64,Missing},2,nres,npat,nrec)
Npoints = zeros(nres,npat,nrec)
# Loop
for (res,resolution) in enumerate(resolutions)
    println("Loop now at res=",res)
    for pat in 1:npat
        sOrig = sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=80_000);
        nd = suggest_nd(sOrig,W)
        for rec in 1:nrec
            Npoints[res,pat,rec] = N(sOrig,W)
            # Store originals 
            # (doing for each reconstruction since some randomnesss in methods)
            scorematOrig[res,pat,rec] = StoyanGrabarnik(sOrig,(u,s)->logλ(u,s,θ₀),W)
            try 
                resultOrig = EstimateParamsPL_Logit(S,sOrig,W; nd=nd)
                θmatOrig[:,res,pat,rec] = coef(resultOrig)
            catch e
                println(e)
                θmatOrig[:,res,pat,rec] .= missing
            end
            # Now reconstruct and same 
            sRec = reconstruct(sOrig,W,resolution)
            scorematRec[res,pat,rec] = StoyanGrabarnik(sRec,(u,s)->logλ(u,s,θ₀),W)
            try 
                resultRec = EstimateParamsPL_Logit(S,sRec,W; nd=nd)
                θmatRec[:,res,pat,rec] = coef(resultRec)
            catch e
                println(e)
                θmatRec[:,res,pat,rec] .= missing 
            end
        end
    end
end

# Full plots 
# (each point is an original pattern and reproduced pattern)
mean(Npoints)
# Stoyan-Grabarnik diagnostics
scoresOrig = reshape(scorematOrig,(nres,npat*nrec))
scoresRec = reshape(scorematRec,(nres,npat*nrec))
plot(title="Stoyan-Grabarnik Diagnostic vs resolution",xlabel="n: resolution=n*n", ylabel="R")
scatter!(resolutions .-0.1,scoresOrig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(scoresOrig, dims=2), linewidth=2,color=:gray, label="SG: meanOriginal")
scatter!(resolutions .+0.1,scoresRec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .+0.1,mean(scoresRec, dims=2), linewidth=2,color=:blue, label="SG: meanReconstructed")
hline!([0], linewidth=2,color=:brown, linestyle=:dash, label="Th. Expectation")
vline!([1/R sqrt(mean(Npoints))], linewidth=1, color=[:red :green], linestyle=:dash, label=false)
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/SG_vs_resolution_geyer_l350_g1.4_R0.03_sat4_full.pdf")


# First parameter (activation)
θ₁Orig = reshape(θmatOrig[1,:,:,:],(nres,npat*nrec))
θ₁Rec = reshape(θmatRec[1,:,:,:],(nres,npat*nrec))
plot(title="Estimates of θ₁ = log(β)",xlabel="n: resolution=n*n", ylabel="θ₁")
scatter!(resolutions .-0.1,θ₁Orig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₁Orig,dims=2), linewidth=2,color=:gray, label="θ₁ estimate from original")
scatter!(resolutions .+0.1,θ₁Rec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₁Rec,dims=2), linewidth=2,color=:blue, label="θ₁ estimate from reconstructed")
hline!([θ₀[1]], linewidth=2,color=:brown, linestyle=:dash, label="θ₁ true")
vline!([1/R sqrt(mean(Npoints))], linewidth=1, color=[:red :green], linestyle=:dash, label=false)
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/theta1est_vs_resolution_geyer_l350_g1.4_R0.03_sat4_full.pdf")


# Second parameter (interaction)
θ₂Orig = reshape(θmatOrig[2,:,:,:],(nres,npat*nrec))
θ₂Rec = reshape(θmatRec[2,:,:,:],(nres,npat*nrec))
plot(title="Estimates of θ₂ = log(γ)",xlabel="n: resolution=n*n", ylabel="θ₂", legend=:bottomright)
scatter!(resolutions .-0.1,θ₂Orig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₂Orig,dims=2), linewidth=2,color=:gray, label="θ₂ estimate from original")
scatter!(resolutions .+0.1,θ₂Rec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₂Rec,dims=2), linewidth=2,color=:blue, label="θ₂ estimate from reconstructed")
hline!([θ₀[2]], linewidth=2,color=:brown, linestyle=:dash, label="θ₂ true")
vline!([1/R sqrt(mean(Npoints))], linewidth=1, color=[:red :green], linestyle=:dash, label=false)
#ylims!((-1.5,1))
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/theta2est_vs_resolution_geyer_l350_g1.4_R0.03_sat4_full.pdf")


# Reduced plots (averaging over original patterns)
# (each point is an original pattern and reproduced pattern)

# Stoyan-Grabarnik diagnostics
scoresOrig = reshape(mean(scorematOrig,dims=2),(nres,nrec))
scoresRec = reshape(mean(scorematRec,dims=2),(nres,nrec))
plot(title="Stoyan-Grabarnik Diagnostic vs resolution",xlabel="n: resolution=n*n", ylabel="R")
scatter!(resolutions .-0.1,scoresOrig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(scoresOrig, dims=2), linewidth=2,color=:gray, label="SG: meanOriginal")
scatter!(resolutions .+0.1,scoresRec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .+0.1,mean(scoresRec, dims=2), linewidth=2,color=:blue, label="SG: meanReconstructed")
hline!([0], linewidth=2,color=:brown, linestyle=:dash, label="Th. Expectation")
vline!([1/R sqrt(mean(Npoints))], linewidth=1, color=[:red :green], linestyle=:dash, label=false)
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/SG_vs_resolution_geyer_l70_g2.2_R0.025_sat6.pdf")

# First parameter (activation)
θ₁Orig = reshape(mean(θmatOrig[1,:,:,:],dims=2),(nres,nrec))
θ₁Rec = reshape(mean(θmatRec[1,:,:,:],dims=2),(nres,nrec))
plot(title="Estimates of θ₁ = log(β)",xlabel="n: resolution=n*n", ylabel="θ₁")
scatter!(resolutions .-0.1,θ₁Orig,  xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₁Orig,dims=2), linewidth=2,color=:gray, label="θ₁ estimate from original")
scatter!(resolutions .+0.1,θ₁Rec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₁Rec,dims=2), linewidth=2,color=:blue, label="θ₁ estimate from reconstructed")
hline!([θ₀[1]], linewidth=2,color=:brown, linestyle=:dash, label="θ₁ true")
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/theta1est_vs_resolution_geyer_l70_g2.2_R0.025_sat6.pdf")

# Second parameter (interaction)
θ₂Orig = reshape(mean(θmatOrig[2,:,:,:],dims=2),(nres,nrec))
θ₂Rec = reshape(mean(θmatRec[2,:,:,:],dims=2),(nres,nrec))
plot(title="Estimates of θ₂ = log(γ)",xlabel="n: resolution=n*n", ylabel="θ₂", legend=:bottomright)
scatter!(resolutions .-0.1,θ₂Orig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₂Orig,dims=2), linewidth=2,color=:gray, label="θ₂ estimate from original")
scatter!(resolutions .+0.1,θ₂Rec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₂Rec,dims=2), linewidth=2,color=:blue, label="θ₂ estimate from reconstructed")
hline!([θ₀[2]], linewidth=2,color=:brown, linestyle=:dash, label="θ₂ true")
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/theta2est_vs_resolution_geyer_l70_g2.2_R0.025_sat6.pdf")










###########################
# Gibbs Point Process (Strauss)
# Distance between points
R = 0.025;
t(u::Point,s::PointSet) = sum(dist(u,x)<R for x in s.items)
# Parameters
λ₀ = 1500; g₀ = 0.3;
θ₀ = [log(λ₀),log(g₀)]
S = [(u,s)->1, (u,s)->(NNdist(s ∪ u,R) - NNdist(s,R))]
# Generate patter
logλ(u,s,θ) = sum(θ[i]S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
p = GibbsProcess(logλ)
s = sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=20_000)
#s = sample_pp(p,Wplus)
plot(s,W)
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/sample_strauss_l1500_g0.3_R0.025_full.pdf")
StoyanGrabarnik(s,(u,s)->logλ(u,s,θ₀),W)
# Sample reconstruction

# Loop over all resolutions
resolutions = [1,2,4,8,16,32,64,128]
nres = length(resolutions)
npat = 30
nrec = 10
scorematOrig = zeros(Union{Float64,Missing},nres,npat,nrec)
θmatOrig = zeros(Union{Float64,Missing},2,nres,npat,nrec)
scorematRec = zeros(Union{Float64,Missing},nres,npat,nrec)
θmatRec = zeros(Union{Float64,Missing},2,nres,npat,nrec)
Npoints = zeros(nres,npat,nrec)
# Loop
for (res,resolution) in enumerate(resolutions)
    println("Loop now at res=",res)
    for pat in 1:npat
        sOrig = sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=10_000);
        nd = suggest_nd(sOrig,W)
        for rec in 1:nrec
            Npoints[res,pat,rec] = N(sOrig,W)
            # Store originals 
            # (doing for each reconstruction since some randomnesss in methods)
            scorematOrig[res,pat,rec] = StoyanGrabarnik(sOrig,(u,s)->logλ(u,s,θ₀),W)
            try 
                resultOrig = EstimateParamsPL_Logit(S,sOrig,W; nd=nd)
                θmatOrig[:,res,pat,rec] = coef(resultOrig)
            catch e
                println(e)
                θmatOrig[:,res,pat,rec] .= missing
            end
            # Now reconstruct and same 
            sRec = reconstruct(sOrig,W,resolution)
            scorematRec[res,pat,rec] = StoyanGrabarnik(sRec,(u,s)->logλ(u,s,θ₀),W)
            try 
                resultRec = EstimateParamsPL_Logit(S,sRec,W; nd=nd)
                θmatRec[:,res,pat,rec] = coef(resultRec)
            catch e
                println(e)
                θmatRec[:,res,pat,rec] .= missing 
            end
        end
    end
end

# Full plots 
# (each point is an original pattern and reproduced pattern)
mean(Npoints)
# Stoyan-Grabarnik diagnostics
scoresOrig = reshape(scorematOrig,(nres,npat*nrec))
scoresRec = reshape(scorematRec,(nres,npat*nrec))
plot(title="Stoyan-Grabarnik Diagnostic vs resolution",xlabel="n: resolution=n*n", ylabel="R")
scatter!(resolutions .-0.1,scoresOrig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(scoresOrig, dims=2), linewidth=2,color=:gray, label="SG: meanOriginal")
scatter!(resolutions .+0.1,scoresRec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .+0.1,mean(scoresRec, dims=2), linewidth=2,color=:blue, label="SG: meanReconstructed")
hline!([0], linewidth=2,color=:brown, linestyle=:dash, label="Th. Expectation")
vline!([1/R sqrt(mean(Npoints))], linewidth=1, color=[:red :green], linestyle=:dash, label=false)
ylims!(-5,18)
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/SG_vs_resolution_strauss_l1500_g0.3_R0.025_full.pdf")


# First parameter (activation)
θ₁Orig = reshape(θmatOrig[1,:,:,:],(nres,npat*nrec))
θ₁Rec = reshape(θmatRec[1,:,:,:],(nres,npat*nrec))
plot(title="Estimates of θ₁ = log(β)",xlabel="n: resolution=n*n", ylabel="θ₁")
scatter!(resolutions .-0.1,θ₁Orig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₁Orig,dims=2), linewidth=2,color=:gray, label="θ₁ estimate from original")
scatter!(resolutions .+0.1,θ₁Rec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₁Rec,dims=2), linewidth=2,color=:blue, label="θ₁ estimate from reconstructed")
hline!([θ₀[1]], linewidth=2,color=:brown, linestyle=:dash, label="θ₁ true")
vline!([1/R sqrt(mean(Npoints))], linewidth=1, color=[:red :green], linestyle=:dash, label=false)
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/theta1est_vs_resolution_strauss_l1500_g0.3_R0.025_full.pdf")


# Second parameter (interaction)
θ₂Orig = reshape(θmatOrig[2,:,:,:],(nres,npat*nrec))
θ₂Rec = reshape(θmatRec[2,:,:,:],(nres,npat*nrec))
plot(title="Estimates of θ₂ = log(γ)",xlabel="n: resolution=n*n", ylabel="θ₂", legend=:bottomright)
scatter!(resolutions .-0.1,θ₂Orig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₂Orig,dims=2), linewidth=2,color=:gray, label="θ₂ estimate from original")
scatter!(resolutions .+0.1,θ₂Rec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₂Rec,dims=2), linewidth=2,color=:blue, label="θ₂ estimate from reconstructed")
hline!([θ₀[2]], linewidth=2,color=:brown, linestyle=:dash, label="θ₂ true")
vline!([1/R sqrt(mean(Npoints))], linewidth=1, color=[:red :green], linestyle=:dash, label=false)
#ylims!((-1.5,1))
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/theta2est_vs_resolution_strauss_l1500_g0.3_R0.025_full.pdf")


# Reduced plots (averaging over original patterns)
# (each point is an original pattern and reproduced pattern)

# Stoyan-Grabarnik diagnostics
scoresOrig = reshape(mean(scorematOrig,dims=2),(nres,nrec))
scoresRec = reshape(mean(scorematRec,dims=2),(nres,nrec))
plot(title="Stoyan-Grabarnik Diagnostic vs resolution",xlabel="n: resolution=n*n", ylabel="R")
scatter!(resolutions .-0.1,scoresOrig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(scoresOrig, dims=2), linewidth=2,color=:gray, label="SG: meanOriginal")
scatter!(resolutions .+0.1,scoresRec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .+0.1,mean(scoresRec, dims=2), linewidth=2,color=:blue, label="SG: meanReconstructed")
hline!([0], linewidth=2,color=:brown, linestyle=:dash, label="Th. Expectation")
vline!([1/R sqrt(mean(Npoints))], linewidth=1, color=[:red :green], linestyle=:dash, label=false)
ylims!(-2.5,7.5)
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/SG_vs_resolution_strauss_l1500_g0.3_R0.025.pdf")

# First parameter (activation)
θ₁Orig = reshape(mean(θmatOrig[1,:,:,:],dims=2),(nres,nrec))
θ₁Rec = reshape(mean(θmatRec[1,:,:,:],dims=2),(nres,nrec))
plot(title="Estimates of θ₁ = log(β)",xlabel="n: resolution=n*n", ylabel="θ₁")
scatter!(resolutions .-0.1,θ₁Orig,  xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₁Orig,dims=2), linewidth=2,color=:gray, label="θ₁ estimate from original")
scatter!(resolutions .+0.1,θ₁Rec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₁Rec,dims=2), linewidth=2,color=:blue, label="θ₁ estimate from reconstructed")
hline!([θ₀[1]], linewidth=2,color=:brown, linestyle=:dash, label="θ₁ true")
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/theta1est_vs_resolution_strauss_l1500_g0.3_R0.025.pdf")

# Second parameter (interaction)
θ₂Orig = reshape(mean(θmatOrig[2,:,:,:],dims=2),(nres,nrec))
θ₂Rec = reshape(mean(θmatRec[2,:,:,:],dims=2),(nres,nrec))
plot(title="Estimates of θ₂ = log(γ)",xlabel="n: resolution=n*n", ylabel="θ₂", legend=:bottomright)
scatter!(resolutions .-0.1,θ₂Orig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₂Orig,dims=2), linewidth=2,color=:gray, label="θ₂ estimate from original")
scatter!(resolutions .+0.1,θ₂Rec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₂Rec,dims=2), linewidth=2,color=:blue, label="θ₂ estimate from reconstructed")
hline!([θ₀[2]], linewidth=2,color=:brown, linestyle=:dash, label="θ₂ true")
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/theta2est_vs_resolution_strauss_l1500_g0.3_R0.025.pdf")







#
R = 0.025;
#γ = 10e18; η = γ^(π*R^2)
η = 2.7
λ₀ = 1000; β = λ₀/η
#
θ₀ = [log(β),log(η)]
C(u::Point,s::PointSet) = FractionOfContestedArea(u,s,R; nd=25)
S = [(u,s)->1, (u,s)->-C(u,s)]
# Generate patter
logλ(u,s,θ) = sum(θ[i]*S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
p = GibbsProcess(logλ)
s = sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=20_000)
#s = sample_pp(p,Wplus)
plot(s,W)
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/sample_areainteraction_l2500_eta2.7_R0.025_full.pdf")
StoyanGrabarnik(s,(u,s)->logλ(u,s,θ₀),W)
CondIntPlot((u,s) -> logλ(u,s),s,W)





resolutions = [1,2,4,8,16,32,64]
nres = length(resolutions)
npat = 10
nrec = 5
scorematOrig = zeros(Union{Float64,Missing},nres,npat,nrec)
θmatOrig = zeros(Union{Float64,Missing},2,nres,npat,nrec)
θmatOrig2 = zeros(Union{Float64,Missing},2,nres,npat,nrec)
scorematRec = zeros(Union{Float64,Missing},nres,npat,nrec)
θmatRec = zeros(Union{Float64,Missing},2,nres,npat,nrec)
θmatRec2 = zeros(Union{Float64,Missing},2,nres,npat,nrec)
Npoints = zeros(nres,npat,nrec)
MISEscore = zeros(nres,npat,nrec)
MISEscore1 = zeros(nres,npat,nrec)
MISEscore2 = zeros(nres,npat,nrec)
m = model_pp(θ₀,[5,0.5],[7,1.5],logλ,false)
# Loop
for (res,resolution) in enumerate(resolutions)
    println("Loop now at res=",res)
    for pat in 1:npat
        sOrig = sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=20_000);
        nd = suggest_nd(sOrig,W)
        for rec in 1:nrec
            Npoints[res,pat,rec] = N(sOrig,W)
            # Store originals 
            # (doing for each reconstruction since some randomnesss in methods)
            scorematOrig[res,pat,rec] = StoyanGrabarnik(sOrig,(u,s)->logλ(u,s,θ₀),W)
            try 
                resultOrig = EstimateParamsPL_Logit(S,sOrig,W; nd=nd)
                θmatOrig[:,res,pat,rec] = coef(resultOrig)
                θest,_,_ = EstimateParamsPPL(m,sOrig,W;p=0.9,k=30,nstep=15) 
                θmatOrig2[:,res,pat,rec] = θest
            catch e
                println(e)
                θmatOrig[:,res,pat,rec] .= missing
                θmatOrig2[:,res,pat,rec] .= missing
            end
            # Now reconstruct and same 
            sRec = reconstruct(sOrig,W,resolution)
            scorematRec[res,pat,rec] = StoyanGrabarnik(sRec,(u,s)->logλ(u,s,θ₀),W)
            try 
                resultRec = EstimateParamsPL_Logit(S,sRec,W; nd=nd)
                θmatRec[:,res,pat,rec] = coef(resultRec)
                θest,_,_ = EstimateParamsPPL(m,sRec,W;p=0.9,k=30,nstep=15) 
                θmatRec2[:,res,pat,rec] = θest
                #MISEscore[res,pat,rec] = MISE(sRec,(u,s)->logλ(u,s,θ₀),(u,s)->logλ(u,s,coef(resultRec)),W)
                #MISEscore1[res,pat,rec] = quadint(u->(exp(logλ(u,sRec,θ₀))-exp(logλ(u,sRec,coef(resultRec))))^2,W)
                MISEscore2[res,pat,rec] = quadint(u->(exp(logλ(u,sOrig))-exp(logλ(u,sRec)))^2,W)
            catch e
                println(e)
                θmatRec[:,res,pat,rec] .= missing 
                θmatRec2[:,res,pat,rec] .= missing 
            end
        end
    end
end

# Full plots 
# (each point is an original pattern and reproduced pattern)
mean(Npoints)
# Stoyan-Grabarnik diagnostics
scoresOrig = reshape(scorematOrig,(nres,npat*nrec))
scoresRec = reshape(scorematRec,(nres,npat*nrec))
plot(title="Stoyan-Grabarnik Diagnostic vs resolution",xlabel="n: resolution=n*n", ylabel="R")
scatter!(resolutions .-0.1,scoresOrig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(scoresOrig, dims=2), linewidth=2,color=:gray, label="SG: meanOriginal")
scatter!(resolutions .+0.1,scoresRec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .+0.1,mean(scoresRec, dims=2), linewidth=2,color=:blue, label="SG: meanReconstructed")
hline!([0], linewidth=2,color=:brown, linestyle=:dash, label="Th. Expectation")
vline!([1/R sqrt(mean(Npoints))], linewidth=1, color=[:red :green], linestyle=:dash, label=false)
#ylims!(-5,25)
#savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/SG_vs_resolution_areainteraction_l2500_eta2.7_R0.025_full.pdf")

scoresOrig = reshape(scorematOrig,(nres,npat*nrec))
scoresRec = reshape(scorematRec,(nres,npat*nrec))
scoresR = scoresRec .- scoresOrig
plot(title="Δ Stoyan-Grabarnik Diagnostic vs resolution",xlabel="n: resolution=n*n", ylabel="R")
scatter!(resolutions .-0.1,scoresR, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(scoresR, dims=2), linewidth=2,color=:gray, label="SG: meanOriginal")
#scatter!(resolutions .+0.1,scoresRec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
#plot!(resolutions .+0.1,mean(scoresRec, dims=2), linewidth=2,color=:blue, label="SG: meanReconstructed")
hline!([0], linewidth=2,color=:brown, linestyle=:dash, label="Th. Expectation")
vline!([1/R sqrt(mean(Npoints))], linewidth=1, color=[:red :green], linestyle=:dash, label=false)
#ylims!(-5,25)


MISEscores = reshape(MISEscore,(nres,npat*nrec))
plot(title="MISES vs resolution",xlabel="n: resolution=n*n", ylabel="MISE")
scatter!(resolutions .-0.1,MISEscores, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(MISEscores, dims=2), linewidth=2,color=:gray, label="MISE")
#scatter!(resolutions .+0.1,scoresRec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
#plot!(resolutions .+0.1,mean(scoresRec, dims=2), linewidth=2,color=:blue, label="SG: meanReconstructed")
hline!([0], linewidth=2,color=:brown, linestyle=:dash, label="Th. Expectation")
vline!([1/R sqrt(mean(Npoints))], linewidth=1, color=[:red :green], linestyle=:dash, label=false)
#ylims!(-5,25)

MISEscores = reshape(MISEscore1,(nres,npat*nrec))
plot(title="MISES vs resolution",xlabel="n: resolution=n*n", ylabel="MISE")
scatter!(resolutions .-0.1,MISEscores, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(MISEscores, dims=2), linewidth=2,color=:gray, label="MISE")
#scatter!(resolutions .+0.1,scoresRec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
#plot!(resolutions .+0.1,mean(scoresRec, dims=2), linewidth=2,color=:blue, label="SG: meanReconstructed")
hline!([0], linewidth=2,color=:brown, linestyle=:dash, label="Th. Expectation")
vline!([1/R sqrt(mean(Npoints))], linewidth=1, color=[:red :green], linestyle=:dash, label=false)
#ylims!(-5,25)

MISEscores = reshape(MISEscore2,(nres,npat*nrec))
plot(title="MISE(λ(x),λ(x̃)) vs resolution",xlabel="n: resolution=n*n", ylabel="MISE")
scatter!(resolutions .-0.1,MISEscores, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(MISEscores, dims=2), linewidth=2,color=:gray, label="MISE")
#scatter!(resolutions .+0.1,scoresRec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
#plot!(resolutions .+0.1,mean(scoresRec, dims=2), linewidth=2,color=:blue, label="SG: meanReconstructed")
hline!([0], linewidth=2,color=:brown, linestyle=:dash, label="Th. Expectation")
vline!([1/R sqrt(mean(Npoints))], linewidth=1, color=[:red :green], linestyle=:dash, label=false)
#ylims!(-5,25)


# First parameter (activation)
θ₁Orig = reshape(θmatOrig[1,:,:,:],(nres,npat*nrec))
θ₁Rec = reshape(θmatRec[1,:,:,:],(nres,npat*nrec))
θ₁Orig2 = reshape(θmatOrig2[1,:,:,:],(nres,npat*nrec))
θ₁Rec2 = reshape(θmatRec2[1,:,:,:],(nres,npat*nrec))
plot(title="Estimates of θ₁ = log(β)",xlabel="n: resolution=n*n", ylabel="θ₁")
scatter!(resolutions .-0.1,θ₁Orig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₁Orig,dims=2), linewidth=2,color=:gray, label="θ₁ estimate from original")
scatter!(resolutions .+0.1,θ₁Rec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₁Rec,dims=2), linewidth=2,color=:blue, label="θ₁ estimate from reconstructed")
scatter!(resolutions .-0.1,θ₁Orig2, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:brown,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₁Orig2,dims=2), linewidth=2,color=:brown, label="θ₁ estimate from original (PPL)")
scatter!(resolutions .+0.1,θ₁Rec2, labels=false, marker = (3,:green,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₁Rec2,dims=2), linewidth=2,color=:green, label="θ₁ estimate from reconstructed (PPL)")
hline!([θ₀[1]], linewidth=2,color=:brown, linestyle=:dash, label="θ₁ true")
vline!([1/R sqrt(mean(Npoints))], linewidth=1, color=[:red :green], linestyle=:dash, label=false)
#savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/theta1est_vs_resolution_areainteraction_l2500_eta2.7_R0.025_full.pdf")


# Second parameter (interaction)
θ₂Orig = reshape(θmatOrig[2,:,:,:],(nres,npat*nrec))
θ₂Rec = reshape(θmatRec[2,:,:,:],(nres,npat*nrec))
θ₂Orig2 = reshape(θmatOrig2[2,:,:,:],(nres,npat*nrec))
θ₂Rec2 = reshape(θmatRec2[2,:,:,:],(nres,npat*nrec))
plot(title="Estimates of θ₂ = log(γ)",xlabel="n: resolution=n*n", ylabel="θ₂", legend=:bottomright)
scatter!(resolutions .-0.1,θ₂Orig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₂Orig,dims=2), linewidth=2,color=:gray, label="θ₂ estimate from original")
scatter!(resolutions .+0.1,θ₂Rec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₂Rec,dims=2), linewidth=2,color=:blue, label="θ₂ estimate from reconstructed")
scatter!(resolutions .-0.1,θ₂Orig2, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:brown,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₂Orig2,dims=2), linewidth=2,color=:brown, label="θ₂ estimate from original (PPL)")
scatter!(resolutions .+0.1,θ₂Rec2, labels=false, marker = (3,:green,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₂Rec2,dims=2), linewidth=2,color=:green, label="θ₂ estimate from reconstructed (PPL)")
hline!([θ₀[2]], linewidth=2,color=:brown, linestyle=:dash, label="θ₂ true")
vline!([1/R sqrt(mean(Npoints))], linewidth=1, color=[:red :green], linestyle=:dash, label=false)
#ylims!((-1.5,1))
#savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/theta2est_vs_resolution_areainteraction_l2500_eta2.7_R0.025_full.pdf")







######## Some attempts at covariate (eg. population density)


μ = [rand(2) for _ in 1:4]
σx = [sqrt(0.05*rand()) for _ in 1:4]; 
σy = [sqrt(0.05*rand()) for _ in 1:4];
w = [0.5+rand() for _ in 1:4];

fN(x,y,μ,σx,σy,w) = w*exp(-(x-μ[1])^2/(2*σx^2) - (y-μ[2])^2/(2*σy))

f(x,y) = sum(fN(x,y,μ[i],σx[i],σy[i],w[i]) for i in 1:3)

#plot(range(0,1,length=20),range(0,1,length=20),f,st=:surf)
contour(range(0,1,length=20),range(0,1,length=20),f)

# Try to create point pattern
#
R = 0.025;
#γ = 10e18; η = γ^(π*R^2)
η = 1.6
λ₀ = 40; β = λ₀/η
δ = 20
S3(u::Point) = f(coordinates(u)[1],coordinates(u)[2])
#
θ₀ = [log(β),log(η),log(δ)]
C(u::Point,s::PointSet) = FractionOfContestedArea(u,s,R; nd=25)
S = [(u,s)->1, (u,s)->-C(u,s),(u,s)->S3(u)]
# Generate patter
logλ(u,s,θ) = sum(θ[i]*S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
p = GibbsProcess(logλ)
s = sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=20_000)
#s = sample_pp(p,Wplus)
contourf(range(0,1,length=20), range(0,1,length=20),f, opacity=0.3)
plot!(s,W)
#savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/sample_areainteraction_inhomogeneous_contour.pdf")
StoyanGrabarnik(s,(u,s)->logλ(u,s,θ₀),W)
CondIntPlot((u,s) -> logλ(u,s),s,W)
#savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/sample_areainteraction_inhomogeneous_condint.pdf")

M = sum(w)


using Distributions
function BinomialRejectionSampling(n::Integer, f::Function, b::Box,M::Real)
    # region configuration
    lo, up = coordinates.(extrema(b))
    V = measure(b)
    # product of uniform distributions
    U = product_distribution([Uniform(lo[i], up[i]) for i in 1:2])
    s = EmptyPointSet()
    while length(s.items) < n 
        p = rand()
        u = Point(rand(U))
        while p > f(u)/(M/V)
            p = rand()
            u = Point(rand(U))
        end
            s = s ∪ u
    end
    return s
end

srec = BinomialRejectionSampling(500,u->S3(u),W,M)
contourf(range(0,1,length=20), range(0,1,length=20),f, opacity=0.3)
plot!(srec,W)

function reconstruct_with_density(s::PointSet,W::Box,n::Int,f::Function,M::Real)
    boxes = PointProcessLearning.partitions(W,n)
    snew = EmptyPointSet()
    n = N(s,boxes[1])
    for b in boxes
        n = N(s,b)
        if n>0
            #snew = snew ∪ sample_pp(BinomialProcess(n),b)
            snew = snew ∪ BinomialRejectionSampling(n,f,b,M)
        end
    end
    return snew
end


srec = reconstruct_with_density(s,W,5,S3,M)
contourf(range(0,1,length=20), range(0,1,length=20),f, opacity=0.3)
plot!(srec,W)


#########

resolutions = [1,2,4,8,16,32,64,128]
nres = length(resolutions)
npat = 30
nrec = 10
nθ = length(θ₀)
scorematOrig = zeros(Union{Float64,Missing},nres,npat,nrec)
θmatOrig = zeros(Union{Float64,Missing},nθ,nres,npat,nrec)
scorematRec1 = zeros(Union{Float64,Missing},nres,npat,nrec)
θmatRec1 = zeros(Union{Float64,Missing},nθ,nres,npat,nrec)
scorematRec2 = zeros(Union{Float64,Missing},nres,npat,nrec)
θmatRec2 = zeros(Union{Float64,Missing},nθ,nres,npat,nrec)
Npoints = zeros(nres,npat,nrec)
# Loop
for (res,resolution) in enumerate(resolutions)
    println("Loop now at res=",res)
    for pat in 1:npat
        sOrig = sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=20_000);
        nd = suggest_nd(sOrig,W)
        for rec in 1:nrec
            Npoints[res,pat,rec] = N(sOrig,W)
            # Store originals 
            # (doing for each reconstruction since some randomnesss in methods)
            scorematOrig[res,pat,rec] = StoyanGrabarnik(sOrig,(u,s)->logλ(u,s,θ₀),W)
            try 
                resultOrig = EstimateParamsPL_Logit(S,sOrig,W; nd=nd)
                θmatOrig[:,res,pat,rec] = coef(resultOrig)
            catch e
                println(e)
                θmatOrig[:,res,pat,rec] .= missing
            end
            # Now reconstruct and same 
            sRec1 = reconstruct(sOrig,W,resolution)
            scorematRec1[res,pat,rec] = StoyanGrabarnik(sRec1,(u,s)->logλ(u,s,θ₀),W)
            try 
                resultRec1 = EstimateParamsPL_Logit(S,sRec1,W; nd=nd)
                θmatRec1[:,res,pat,rec] = coef(resultRec1)
            catch e
                println(e)
                θmatRec1[:,res,pat,rec] .= missing 
            end
            # With density
            sRec2 = reconstruct_with_density(sOrig,W,resolution,S3,M)
            scorematRec2[res,pat,rec] = StoyanGrabarnik(sRec2,(u,s)->logλ(u,s,θ₀),W)
            try 
                resultRec2 = EstimateParamsPL_Logit(S,sRec2,W; nd=nd)
                θmatRec2[:,res,pat,rec] = coef(resultRec2)
            catch e
                println(e)
                θmatRec2[:,res,pat,rec] .= missing 
            end
        end
    end
end

mudist = mean([sqrt((μ[i][1]-μ[i][1])^2+(μ[i][2]-μ[j][2])^2) for i in 1:3, j in 1:3])
mudist = maximum([sqrt((μ[i][1]-μ[i][1])^2+(μ[i][2]-μ[j][2])^2) for i in 1:3, j in 1:3])


# Full plots 
# (each point is an original pattern and reproduced pattern)
mean(Npoints)
# Stoyan-Grabarnik diagnostics
scoresOrig = reshape(scorematOrig,(nres,npat*nrec))
scoresRec1 = reshape(scorematRec1,(nres,npat*nrec))
scoresRec2 = reshape(scorematRec2,(nres,npat*nrec))
plot(title="Stoyan-Grabarnik Diagnostic vs resolution",xlabel="n: resolution=n*n", ylabel="R")
scatter!(resolutions .-0.1,scoresOrig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(scoresOrig, dims=2), linewidth=2,color=:gray, label="SG: meanOriginal")
scatter!(resolutions .+0.1,scoresRec1, labels=false, marker = (3,:blue,0.2,stroke(0,:blue)))
plot!(resolutions .+0.1,mean(scoresRec1, dims=2), linewidth=2,color=:blue, label="SG: meanReconstructed (without density)")
scatter!(resolutions .+0.1,scoresRec2, labels=false, marker = (3,:brown,0.2,stroke(0,:brown)))
plot!(resolutions .+0.1,mean(scoresRec2, dims=2), linewidth=2,color=:brown, label="SG: meanReconstructed (with density)")
hline!([0], linewidth=2,color=:brown, linestyle=:dash, label="Th. Expectation")
vline!([1/R sqrt(mean(Npoints)) 1/mean(vcat(σx,σy)) 1/mudist], linewidth=1, color=[:red :green :purple :brown], linestyle=:dash, label=false)
#ylims!(-5,25)
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/SG_vs_resolution_areainteraction_inhomogeneous_rechomo_full.pdf")


# First parameter (activation)
θ₁Orig = reshape(θmatOrig[1,:,:,:],(nres,npat*nrec))
θ₁Rec1 = reshape(θmatRec1[1,:,:,:],(nres,npat*nrec))
θ₁Rec2 = reshape(θmatRec2[1,:,:,:],(nres,npat*nrec))
plot(title="Estimates of θ₁ = log(β)",xlabel="n: resolution=n*n", ylabel="θ₁")
scatter!(resolutions .-0.1,θ₁Orig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₁Orig,dims=2), linewidth=2,color=:gray, label="θ₁ estimate from original")
scatter!(resolutions .+0.1,θ₁Rec1, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₁Rec1,dims=2), linewidth=2,color=:blue, label="θ₁ estimate from reconstructed (without density)")
scatter!(resolutions .+0.1,θ₁Rec2, labels=false, marker = (3,:brown,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₁Rec2,dims=2), linewidth=2,color=:brown, label="θ₁ estimate from reconstructed (with density)")
hline!([θ₀[1]], linewidth=2,color=:brown, linestyle=:dash, label="θ₁ true")
vline!([1/R sqrt(mean(Npoints)) 1/mean(vcat(σx,σy)) 1/mudist], linewidth=1, color=[:red :green :purple :brown], linestyle=:dash, label=false)
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/theta1est_vs_resolution_areainteraction_inhomogeneous_rechomo_full.pdf")


# Second parameter (interaction)
θ₂Orig = reshape(θmatOrig[2,:,:,:],(nres,npat*nrec))
θ₂Rec1 = reshape(θmatRec1[2,:,:,:],(nres,npat*nrec))
θ₂Rec2 = reshape(θmatRec2[2,:,:,:],(nres,npat*nrec))
plot(title="Estimates of θ₂ = log(η)",xlabel="n: resolution=n*n", ylabel="θ₂", legend=:bottomright)
scatter!(resolutions .-0.1,θ₂Orig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₂Orig,dims=2), linewidth=2,color=:gray, label="θ₂ estimate from original")
scatter!(resolutions .+0.1,θ₂Rec1, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₂Rec1,dims=2), linewidth=2,color=:blue, label="θ₂ estimate from reconstructed (without density)")
scatter!(resolutions .+0.1,θ₂Rec2, labels=false, marker = (3,:brown,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₂Rec2,dims=2), linewidth=2,color=:brown, label="θ₂ estimate from reconstructed (with density)")
hline!([θ₀[2]], linewidth=2,color=:brown, linestyle=:dash, label="θ₂ true")
vline!([1/R sqrt(mean(Npoints)) 1/mean(vcat(σx,σy)) 1/mudist], linewidth=1, color=[:red :green :purple :brown], linestyle=:dash, label=false)
#ylims!((-1.5,1))
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/theta2est_vs_resolution_areainteraction_inhomogeneous_rechomo_full.pdf")

# Third parameter (intensity covariate)
θ₃Orig = reshape(θmatOrig[3,:,:,:],(nres,npat*nrec))
θ₃Rec1 = reshape(θmatRec1[3,:,:,:],(nres,npat*nrec))
θ₃Rec2 = reshape(θmatRec2[3,:,:,:],(nres,npat*nrec))
plot(title="Estimates of θ₃ = log(δ)",xlabel="n: resolution=n*n", ylabel="θ₃", legend=:bottomright)
scatter!(resolutions .-0.1,θ₃Orig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₃Orig,dims=2), linewidth=2,color=:gray, label="θ₃ estimate from original")
scatter!(resolutions .+0.1,θ₃Rec1, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₃Rec1,dims=2), linewidth=2,color=:blue, label="θ₃ estimate from reconstructed (w.o. density)")
scatter!(resolutions .+0.1,θ₃Rec2, labels=false, marker = (3,:brown,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₃Rec2,dims=2), linewidth=2,color=:brown, label="θ₃ estimate from reconstructed (w. density)")
hline!([θ₀[3]], linewidth=2,color=:brown, linestyle=:dash, label="θ₃ true")
vline!([1/R sqrt(mean(Npoints)) 1/mudist 1/mean(vcat(σx,σy))], linewidth=1, color=[:red :green :purple :brown], linestyle=:dash, label=false)
#ylims!((-1.5,1))
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/theta3est_vs_resolution_areainteraction_inhomogeneous_rechomo_full.pdf")





#########################



####################
# Point Processes on linear networks

# Set up linear network
W = Box((0.,0.),(1.,1.))
LR = RandomLines(30,PointProcessLearning.reduce(W,0.05))
L = LinearNetwork(LR)
PlotLinearNetwork(L)
tL = thin(L,0.8)
PlotLinearNetwork(tL)

# Covariate density
μ = [rand(2) for _ in 1:4]
σx = [sqrt(0.05*rand()) for _ in 1:4]; 
σy = [sqrt(0.05*rand()) for _ in 1:4];
w = [0.5+rand() for _ in 1:4];
fN(x,y,μ,σx,σy,w) = w*exp(-(x-μ[1])^2/(2*σx^2) - (y-μ[2])^2/(2*σy))
f(x,y) = sum(fN(x,y,μ[i],σx[i],σy[i],w[i]) for i in 1:3)
#plot(range(0,1,length=20),range(0,1,length=20),f,st=:surf)
contour!(range(0,1,length=20),range(0,1,length=20),f)
M = sum(w)

# Area Interaction process
R = 0.05;
#γ = 10e18; η = γ^(π*R^2)
η = 1.5
λ₀ = 6; β = λ₀/η
δ = 3
S3(u::Point) = f(coordinates(u)[1],coordinates(u)[2])
#
θ₀ = [log(β),log(η),log(δ)]
C(u::Point,s::PointSet) = FractionOfContestedArea(u,s,R; nd=25)
S = [(u,s)->1, (u,s)->-C(u,s),(u,s)->S3(u)]
# Generate patter
logλ(u,s,θ) = sum(θ[i]*S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
p = GibbsProcess(logλ)
#s = sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=20_000)
s,segids = sample_pp(p,L; niter=50_000)
p1 = PlotLinearNetwork(L)
contourf!(p1,range(0,1,length=20), range(0,1,length=20),f, opacity=0.1)
plot!(p1,s,W, title="Original")
plot!(p1,PointProcessLearning.Box2Shape.(PointProcessLearning.partitions(W,6)), fillcolor=false, linestyle=:dash,labels=false)



function reconstruct(s::PointSet, L::LinearNetwork,W::Box, n::Int64)
    boxes = PointProcessLearning.partitions(W,n)
    networks = [subnetwork(L,b) for b in boxes]
    nboxes = n*n
    patterns = Vector{PointSet}(undef,nboxes)
    segidsall = Vector{Vector{Int64}}(undef,nboxes)
    for i in 1:nboxes
        ni = N(s,boxes[i])
        pi = BinomialProcess(ni)
        patterns[i], segidsall[i] = sample_pp(pi,networks[i])
    end
    return patterns, segidsall, networks
end
# Test
pat, segs, nets = reconstruct(s,L,W,6)
srec = PointSet(vcat([ss.items for ss in pat]...))
p2 = PlotLinearNetwork(L)
contourf!(p2,range(0,1,length=20), range(0,1,length=20),f, opacity=0.1)
plot!(p2,srec,W, title="Reconstructed")
plot!(p2,PointProcessLearning.Box2Shape.(PointProcessLearning.partitions(W,6)), fillcolor=false, linestyle=:dash,labels=false)
plot(p1,p2)
plot!(size=(1000,400))
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/sample_reconstruction_areainteraction_linnet_morecluster_resolution6.pdf")

# Set up loop
resolutions = [1,2,4,8,16,32,64]
nres = length(resolutions)
npat = 20
nrec = 10
scorematOrig = zeros(Union{Float64,Missing},nres,npat,nrec)
θmatOrig = zeros(Union{Float64,Missing},3,nres,npat,nrec)
scorematRec = zeros(Union{Float64,Missing},nres,npat,nrec)
θmatRec = zeros(Union{Float64,Missing},3,nres,npat,nrec)
Npoints = zeros(nres,npat,nrec)
# Loop
for (res,resolution) in enumerate(resolutions)
    println("Loop now at res=",res)
    for pat in 1:npat
        sOrig, segids = sample_pp(p,L; niter=50_000);
        nd = 5_000;
        for rec in 1:nrec
            Npoints[res,pat,rec] = N(sOrig,W)
            # Store originals 
            # (doing for each reconstruction since some randomnesss in methods)
            scorematOrig[res,pat,rec] = StoyanGrabarnik(sOrig,(u,s)->logλ(u,s,θ₀),L,segids)
            try 
                resultOrig = EstimateParamsPL_Logit(S,sOrig,L; nd=nd)
                θmatOrig[:,res,pat,rec] = coef(resultOrig)
            catch e
                println(e)
                θmatOrig[:,res,pat,rec] .= missing
            end
            # Now reconstruct and same 
            sRecs,segs,nets = reconstruct(sOrig,L,W,resolution)
            scorematRec[res,pat,rec] = sum(StoyanGrabarnik(sRecs[i],(u,s)->logλ(u,s,θ₀),nets[i],segs[i]) for i in 1:length(nets))
            sRec = PointSet(vcat([ss.items for ss in sRecs]...))
            try 
                resultRec = EstimateParamsPL_Logit(S,sRec,L; nd=nd)
                θmatRec[:,res,pat,rec] = coef(resultRec)
            catch e
                println(e)
                θmatRec[:,res,pat,rec] .= missing 
            end
        end
    end
end

# Full plots 
# (each point is an original pattern and reproduced pattern)
mean(Npoints)
# Stoyan-Grabarnik diagnostics
scoresOrig = reshape(scorematOrig,(nres,npat*nrec))
scoresRec = reshape(scorematRec,(nres,npat*nrec))
plot(title="Stoyan-Grabarnik Diagnostic vs resolution",xlabel="n: resolution=n*n", ylabel="R")
scatter!(resolutions .-0.1,scoresOrig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(scoresOrig, dims=2), linewidth=2,color=:gray, label="SG: meanOriginal")
scatter!(resolutions .+0.1,scoresRec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .+0.1,mean(scoresRec, dims=2), linewidth=2,color=:blue, label="SG: meanReconstructed")
hline!([0], linewidth=2,color=:brown, linestyle=:dash, label="Th. Expectation")
vline!([1/R sqrt(mean(Npoints))], linewidth=1, color=[:red :green], linestyle=:dash, label=false)
#ylims!(-5,25)
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/SG_vs_resolution_areainteraction_linnet_l6_eta1.5_delta3_R0.05_full.pdf")


# First parameter (activation)
θ₁Orig = reshape(θmatOrig[1,:,:,:],(nres,npat*nrec))
θ₁Rec = reshape(θmatRec[1,:,:,:],(nres,npat*nrec))
plot(title="Estimates of θ₁ = log(β)",xlabel="n: resolution=n*n", ylabel="θ₁")
scatter!(resolutions .-0.1,θ₁Orig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₁Orig,dims=2), linewidth=2,color=:gray, label="θ₁ estimate from original")
scatter!(resolutions .+0.1,θ₁Rec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₁Rec,dims=2), linewidth=2,color=:blue, label="θ₁ estimate from reconstructed")
hline!([θ₀[1]], linewidth=2,color=:brown, linestyle=:dash, label="θ₁ true")
vline!([1/R sqrt(mean(Npoints))], linewidth=1, color=[:red :green], linestyle=:dash, label=false)
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/theta1est_vs_resolution_areainteraction_linnet_l6_eta1.5_delta3_R0.05_full.pdf")


# Second parameter (interaction)
θ₂Orig = reshape(θmatOrig[2,:,:,:],(nres,npat*nrec))
θ₂Rec = reshape(θmatRec[2,:,:,:],(nres,npat*nrec))
plot(title="Estimates of θ₂ = log(γ)",xlabel="n: resolution=n*n", ylabel="θ₂", legend=:bottomright)
scatter!(resolutions .-0.1,θ₂Orig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₂Orig,dims=2), linewidth=2,color=:gray, label="θ₂ estimate from original")
scatter!(resolutions .+0.1,θ₂Rec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₂Rec,dims=2), linewidth=2,color=:blue, label="θ₂ estimate from reconstructed")
hline!([θ₀[2]], linewidth=2,color=:brown, linestyle=:dash, label="θ₂ true")
vline!([1/R sqrt(mean(Npoints))], linewidth=1, color=[:red :green], linestyle=:dash, label=false)
#ylims!((-1.5,1))
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/theta2est_vs_resolution_areainteraction_linnet_l6_eta1.5_delta3_R0.05_full.pdf")


# Second parameter (interaction)
θ₃Orig = reshape(θmatOrig[3,:,:,:],(nres,npat*nrec))
θ₃Rec = reshape(θmatRec[3,:,:,:],(nres,npat*nrec))
plot(title="Estimates of θ₃ = log(γ)",xlabel="n: resolution=n*n", ylabel="θ₃", legend=:bottomright)
scatter!(resolutions .-0.1,θ₃Orig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₃Orig,dims=2), linewidth=2,color=:gray, label="θ₃ estimate from original")
scatter!(resolutions .+0.1,θ₃Rec, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₃Rec,dims=2), linewidth=2,color=:blue, label="θ₃ estimate from reconstructed")
hline!([θ₀[3]], linewidth=2,color=:brown, linestyle=:dash, label="θ₃ true")
vline!([1/R sqrt(mean(Npoints))], linewidth=1, color=[:red :green], linestyle=:dash, label=false)
#ylims!((-1.5,1))
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/theta3est_vs_resolution_areainteraction__linnet_l5_eta1.25_delta3_full.pdf")



function BinomialRejectionSampling(n::Integer, f::Function, L::LinearNetwork,M::Real)
    # region configuration
    #lo, up = coordinates.(extrema(b))
    #V = measure(b)
    V = PointProcessLearning.measure(L)
    probs = Meshes.measure.(L.segments)./PointProcessLearning.measure(L)
    pts = Point2[]
    segids = Int64[]
    #for i in 1:p.n
    #    segids[i] = sample(1:length(L.segments), Weights(probs))
    #    pts[i] = SamplePointOnSegment(L.segments[segids[i]])
    #end
    #return PointSet(pts), segids
    # product of uniform distributions
    #U = product_distribution([Uniform(lo[i], up[i]) for i in 1:2])
    #s = EmptyPointSet()
    while length(pts) < n 
        p = rand()
        #u = Point(rand(U))
        segid = sample(1:length(L.segments), Weights(probs))
        u = SamplePointOnSegment(L.segments[segid])
        while p > f(u)/(M/V)
            p = rand()
            segid = sample(1:length(L.segments), Weights(probs))
            u = SamplePointOnSegment(L.segments[segid])
        end
        push!(pts, u)
        push!(segids,segid)
    end
    #return pts
    return PointSet(pts), segids
end


function reconstruct_with_density(s::PointSet, L::LinearNetwork,W::Box, n::Int64,f::Function,M::Real)
    boxes = PointProcessLearning.partitions(W,n)
    networks = [subnetwork(L,b) for b in boxes]
    nboxes = n*n
    patterns = Vector{PointSet}(undef,nboxes)
    segidsall = Vector{Vector{Int64}}(undef,nboxes)
    for i in 1:nboxes
        ni = N(s,boxes[i])
        patterns[i], segidsall[i] = BinomialRejectionSampling(ni,f,networks[i],M)
    end
    return patterns, segidsall, networks
end





# Set up linear network
W = Box((0.,0.),(1.,1.))
LR = RandomLines(30,PointProcessLearning.reduce(W,0.05))
L = LinearNetwork(LR)
PlotLinearNetwork(L; lowc=(0.75,2,:black))
tL = thin(L,0.8)
PlotLinearNetwork(tL)

# Covariate density
μ = [rand(2) for _ in 1:4]
σx = [sqrt(0.05*rand()) for _ in 1:4]; 
σy = [sqrt(0.05*rand()) for _ in 1:4];
w = [0.5+rand() for _ in 1:4];
fN(x,y,μ,σx,σy,w) = w*exp(-(x-μ[1])^2/(2*σx^2) - (y-μ[2])^2/(2*σy))
f(x,y) = sum(fN(x,y,μ[i],σx[i],σy[i],w[i]) for i in 1:3)
f1(u::Point) = f(coordinates(u)[1],coordinates(u)[2])
#plot(range(0,1,length=20),range(0,1,length=20),f,st=:surf)
contour(range(0,1,length=20),range(0,1,length=20),f)
M = sum(w)*PointProcessLearning.measure(L)

M/PointProcessLearning.measure(L)

f1(Point2(0.8,0.6))/(M/PointProcessLearning.measure(L))

BinomialRejectionSampling(10,f1,L,M)


# Area Interaction process
R = 0.025;
#γ = 10e18; η = γ^(π*R^2)
η = 1.3
λ₀ = 5; β = λ₀/η
δ = 2
S3(u::Point) = f(coordinates(u)[1],coordinates(u)[2])
#
θ₀ = [log(β),log(η),log(δ)]
C(u::Point,s::PointSet) = FractionOfContestedArea(u,s,R; nd=25)
S = [(u,s)->1, (u,s)->-C(u,s),(u,s)->S3(u)]
# Generate patter
logλ(u,s,θ) = sum(θ[i]*S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
p = GibbsProcess(logλ)
#s = sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=20_000)
s,segids = sample_pp(p,L; niter=50_000)
p1 = PlotLinearNetwork(L; lowc=(0.75,2,:black))
contourf!(p1,range(0,1,length=20), range(0,1,length=20),f, opacity=0.1)
plot!(p1,s,W, title="Original")
pat, segs, nets = reconstruct_with_density(s,L,W,6,f1,M)
srec = PointSet(vcat([ss.items for ss in pat]...))
p2 = PlotLinearNetwork(L)
contourf!(p2,range(0,1,length=20), range(0,1,length=20),f, opacity=0.1)
plot!(p2,srec,W, title="Reconstructed")
plot!(p2,PointProcessLearning.Box2Shape.(PointProcessLearning.partitions(W,6)), fillcolor=false, linestyle=:dash,labels=false)
plot(p1,p2)
plot!(size=(1000,400))

# Set up loop
resolutions = [1,2,4,8,16,32,64]
nres = length(resolutions)
npat = 10
nrec = 5
nθ = length(θ₀)
scorematOrig = zeros(Union{Float64,Missing},nres,npat,nrec)
θmatOrig = zeros(Union{Float64,Missing},nθ,nres,npat,nrec)
scorematRec1 = zeros(Union{Float64,Missing},nres,npat,nrec)
θmatRec1 = zeros(Union{Float64,Missing},nθ,nres,npat,nrec)
scorematRec2 = zeros(Union{Float64,Missing},nres,npat,nrec)
θmatRec2 = zeros(Union{Float64,Missing},nθ,nres,npat,nrec)
Npoints = zeros(nres,npat,nrec)
# Loop
for (res,resolution) in enumerate(resolutions)
    println("Loop now at res=",res)
    for pat in 1:npat
        sOrig, segids = sample_pp(p,L; niter=70_000);
        nd = 5_000;
        for rec in 1:nrec
            Npoints[res,pat,rec] = N(sOrig,W)
            # Store originals 
            # (doing for each reconstruction since some randomnesss in methods)
            scorematOrig[res,pat,rec] = StoyanGrabarnik(sOrig,(u,s)->logλ(u,s,θ₀),L,segids)
            try 
                resultOrig = EstimateParamsPL_Logit(S,sOrig,L; nd=nd)
                θmatOrig[:,res,pat,rec] = coef(resultOrig)
            catch e
                println(e)
                θmatOrig[:,res,pat,rec] .= missing
            end
            # Now reconstruct and same 
            sRecs,segs,nets = reconstruct(sOrig,L,W,resolution)
            scorematRec1[res,pat,rec] = sum(StoyanGrabarnik(sRecs[i],(u,s)->logλ(u,s,θ₀),nets[i],segs[i]) for i in 1:length(nets))
            sRec = PointSet(vcat([ss.items for ss in sRecs]...))
            try 
                resultRec = EstimateParamsPL_Logit(S,sRec,L; nd=nd)
                θmatRec1[:,res,pat,rec] = coef(resultRec)
            catch e
                println(e)
                θmatRec1[:,res,pat,rec] .= missing 
            end
            # Reconstruct with density
            sRecs,segs,nets = reconstruct_with_density(sOrig,L,W,resolution,f1,M)
            scorematRec2[res,pat,rec] = sum(StoyanGrabarnik(sRecs[i],(u,s)->logλ(u,s,θ₀),nets[i],segs[i]) for i in 1:length(nets))
            sRec = PointSet(vcat([ss.items for ss in sRecs]...))
            try 
                resultRec = EstimateParamsPL_Logit(S,sRec,L; nd=nd)
                θmatRec2[:,res,pat,rec] = coef(resultRec)
            catch e
                println(e)
                θmatRec2[:,res,pat,rec] .= missing 
            end
        end
    end
end

mudist = mean([sqrt((μ[i][1]-μ[i][1])^2+(μ[i][2]-μ[j][2])^2) for i in 1:3, j in 1:3])
mudist = maximum([sqrt((μ[i][1]-μ[i][1])^2+(μ[i][2]-μ[j][2])^2) for i in 1:3, j in 1:3])


# Full plots 
# (each point is an original pattern and reproduced pattern)
mean(Npoints)
# Stoyan-Grabarnik diagnostics
scoresOrig = reshape(scorematOrig,(nres,npat*nrec))
scoresRec1 = reshape(scorematRec1,(nres,npat*nrec))
scoresRec2 = reshape(scorematRec2,(nres,npat*nrec))
plot(title="Stoyan-Grabarnik Diagnostic vs resolution",xlabel="n: resolution=n*n", ylabel="R")
scatter!(resolutions .-0.1,scoresOrig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(scoresOrig, dims=2), linewidth=2,color=:gray, label="SG: meanOriginal")
scatter!(resolutions .+0.1,scoresRec1, labels=false, marker = (3,:blue,0.2,stroke(0,:blue)))
plot!(resolutions .+0.1,mean(scoresRec1, dims=2), linewidth=2,color=:blue, label="SG: meanReconstructed (without density)")
scatter!(resolutions .+0.1,scoresRec2, labels=false, marker = (3,:brown,0.2,stroke(0,:brown)))
plot!(resolutions .+0.1,mean(scoresRec2, dims=2), linewidth=2,color=:brown, label="SG: meanReconstructed (with density)")
hline!([0], linewidth=2,color=:brown, linestyle=:dash, label="Th. Expectation")
vline!([1/R sqrt(mean(Npoints)) 1/mean(vcat(σx,σy)) 1/mudist], linewidth=1, color=[:red :green :purple :brown], linestyle=:dash, label=false)
#ylims!(-5,25)
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/SG_vs_resolution_areainteraction_linnet_density_full.pdf")


# First parameter (activation)
θ₁Orig = reshape(θmatOrig[1,:,:,:],(nres,npat*nrec))
θ₁Rec1 = reshape(θmatRec1[1,:,:,:],(nres,npat*nrec))
θ₁Rec2 = reshape(θmatRec2[1,:,:,:],(nres,npat*nrec))
plot(title="Estimates of θ₁ = log(β)",xlabel="n: resolution=n*n", ylabel="θ₁")
scatter!(resolutions .-0.1,θ₁Orig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₁Orig,dims=2), linewidth=2,color=:gray, label="θ₁ estimate from original")
scatter!(resolutions .+0.1,θ₁Rec1, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₁Rec1,dims=2), linewidth=2,color=:blue, label="θ₁ estimate from reconstructed (without density)")
scatter!(resolutions .+0.1,θ₁Rec2, labels=false, marker = (3,:brown,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₁Rec2,dims=2), linewidth=2,color=:brown, label="θ₁ estimate from reconstructed (with density)")
hline!([θ₀[1]], linewidth=2,color=:brown, linestyle=:dash, label="θ₁ true")
vline!([1/R sqrt(mean(Npoints)) 1/mean(vcat(σx,σy)) 1/mudist], linewidth=1, color=[:red :green :purple :brown], linestyle=:dash, label=false)
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/theta1est_vs_resolution_areainteraction_linnet_density_full.pdf")


# Second parameter (interaction)
θ₂Orig = reshape(θmatOrig[2,:,:,:],(nres,npat*nrec))
θ₂Rec1 = reshape(θmatRec1[2,:,:,:],(nres,npat*nrec))
θ₂Rec2 = reshape(θmatRec2[2,:,:,:],(nres,npat*nrec))
plot(title="Estimates of θ₂ = log(η)",xlabel="n: resolution=n*n", ylabel="θ₂", legend=:bottomright)
scatter!(resolutions .-0.1,θ₂Orig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₂Orig,dims=2), linewidth=2,color=:gray, label="θ₂ estimate from original")
scatter!(resolutions .+0.1,θ₂Rec1, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₂Rec1,dims=2), linewidth=2,color=:blue, label="θ₂ estimate from reconstructed (without density)")
scatter!(resolutions .+0.1,θ₂Rec2, labels=false, marker = (3,:brown,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₂Rec2,dims=2), linewidth=2,color=:brown, label="θ₂ estimate from reconstructed (with density)")
hline!([θ₀[2]], linewidth=2,color=:brown, linestyle=:dash, label="θ₂ true")
vline!([1/R sqrt(mean(Npoints)) 1/mean(vcat(σx,σy)) 1/mudist], linewidth=1, color=[:red :green :purple :brown], linestyle=:dash, label=false)
#ylims!((-1.5,1))
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/theta2est_vs_resolution_areainteraction_linnet_density_full.pdf")

# Third parameter (intensity covariate)
θ₃Orig = reshape(θmatOrig[3,:,:,:],(nres,npat*nrec))
θ₃Rec1 = reshape(θmatRec1[3,:,:,:],(nres,npat*nrec))
θ₃Rec2 = reshape(θmatRec2[3,:,:,:],(nres,npat*nrec))
plot(title="Estimates of θ₃ = log(δ)",xlabel="n: resolution=n*n", ylabel="θ₃", legend=:bottomright)
scatter!(resolutions .-0.1,θ₃Orig, xaxis=:log, xticks=(resolutions,resolutions), labels=false, marker = (3,:gray,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₃Orig,dims=2), linewidth=2,color=:gray, label="θ₃ estimate from original")
scatter!(resolutions .+0.1,θ₃Rec1, labels=false, marker = (3,:blue,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₃Rec1,dims=2), linewidth=2,color=:blue, label="θ₃ estimate from reconstructed (w.o. density)")
scatter!(resolutions .+0.1,θ₃Rec2, labels=false, marker = (3,:brown,0.2,stroke(0,:gray)))
plot!(resolutions .-0.1,mean(θ₃Rec2,dims=2), linewidth=2,color=:brown, label="θ₃ estimate from reconstructed (w. density)")
hline!([θ₀[3]], linewidth=2,color=:brown, linestyle=:dash, label="θ₃ true")
vline!([1/R sqrt(mean(Npoints)) 1/mudist 1/mean(vcat(σx,σy))], linewidth=1, color=[:red :green :purple :brown], linestyle=:dash, label=false)
#ylims!((-1.5,1))
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/theta3est_vs_resolution_areainteraction_linnet_density_full.pdf")








BermanTurnerIntegral( u -> (exp(logλ(u,s,θ₀))-exp(logλ(u,s,θ₁)))^2, W)









function MIE(s::PointSet, logλ1::Function,logλ2::Function,W::Box; nd = 1000)
    λ1(u,s) = exp(logλ1(u,s))
    λ2(u,s) = exp(logλ2(u,s))
    pd = BinomialProcess(nd)
    d = sample_pp(pd,W)
    sd = s ∪ d
    w = measure(W)/N(sd,W)
    return sum( w*(λ1(u,s)-λ2(u,s)) for u in sd.items)
end

function MISE(s::PointSet, logλ1::Function,logλ2::Function,W::Box; nd = 1000)
    λ1(u,s) = exp(logλ1(u,s))
    λ2(u,s) = exp(logλ2(u,s))
    pd = BinomialProcess(nd)
    d = sample_pp(pd,W)
    sd = s ∪ d
    w = measure(W)/N(sd,W)
    return sum( w*(λ1(u,s)-λ2(u,s))^2 for u in sd.items)
end

function MIE(s::PointSet, logλ1::Function,logλ2::Function,L::LinearNetwork; nd = 1000)
    λ1(u,s) = exp(logλ1(u,s))
    λ2(u,s) = exp(logλ2(u,s))
    pd = BinomialProcess(nd)
    d, _ = sample_pp(pd,L)
    sd = s ∪ d
    w = PointProcessLearning.measure(L)/N(sd,W)
    return sum( w*(λ1(u,s)-λ2(u,s)) for u in sd.items)
end

function MISE(s::PointSet, logλ1::Function,logλ2::Function,L::LinearNetwork; nd = 1000)
    λ1(u,s) = exp(logλ1(u,s))
    λ2(u,s) = exp(logλ2(u,s))
    pd = BinomialProcess(nd)
    d,_ = sample_pp(pd,L)
    sd = s ∪ d
    w = PointProcessLearning.measure(L)/N(sd,W)
    return sum( w*(λ1(u,s)-λ2(u,s))^2 for u in sd.items)
end


θ₀
θ₁ = [6.34,0.99]


MISE(s, (u,s)->logλ(u,s,θ₀), (u,s)->logλ(u,s,θ₁),W, nd=50)
MISE(s, (u,s)->logλ(u,s,θ₀), (u,s)->logλ(u,s,θ₁),W, nd=100)

MIE(s, (u,s)->logλ(u,s,θ₀), (u,s)->logλ(u,s,θ₁),L)
MISE(s, (u,s)->logλ(u,s,θ₀), (u,s)->logλ(u,s,θ₁),L)
MISE2(s,segids, (u,s)->logλ(u,s,θ₀), (u,s)->logλ(u,s,θ₁),L,nd=50)
MISE2(s,segids, (u,s)->logλ(u,s,θ₀), (u,s)->logλ(u,s,θ₁),L,nd=100)


function MISE2(s::PointSet,logλ1::Function,logλ2::Function,W::Box; nd = 100)
    boxes = PointProcessLearning.partitions(W,nd)
    λ1(u,s) = exp(logλ1(u,s))
    λ2(u,s) = exp(logλ2(u,s))
    Sum = 0
    for box in boxes
        sb = PointSet(filter(x->in(x,box),s.items))
        db = PointSet(Meshes.centroid(box))
        vb = sb ∪ db
        nb = length(vb.items)
        w = Meshes.measure(box)/nb
        Sum = Sum + sum( w*(λ1(v,s)-λ2(v,s))^2 for v in vb.items)
    end
    return Sum
end



function MISE2(s::PointSet,segids::Vector{Int64},logλ1::Function,logλ2::Function,L::LinearNetwork; nd = 100)
    segments = L.segments
    λ1(u,s) = exp(logλ1(u,s))
    λ2(u,s) = exp(logλ2(u,s))
    probs = Meshes.measure.(segments)./PointProcessLearning.measure(L)
    Sum = 0
    for (i,seg) in enumerate(segments)
        sb = PointSet(s.items[segids .== i])
        db = PointSet([SamplePointOnSegment(seg) for _ in 1:ceil(Int64,nd*probs[i])])
        vb = sb ∪ db
        nb = length(vb.items)
        w = Meshes.measure(seg)/nb
        Sum = Sum + sum( w*(λ1(v,s)-λ2(v,s))^2 for v in vb.items)
    end
    return Sum
end
    




function MISE2(s::PointSet,segids::Vector{Int64},logλ1::Function,logλ2::Function,L::LinearNetwork; nd = 100)
    #segments = L.segments
    λ1(u,s) = exp(logλ1(u,s))
    λ2(u,s) = exp(logλ2(u,s))
    d = PointSet([centroid(seg) for seg in L.segments])
    dsegids = collect(eachindex(L.segments))
    v = u ∪ s
    vsegids = vcat(segids,dsegids)
    a = measure.(L.segments)
    #
    return sum( (a/N(v,boxes[boxind[i]]))*(λ1(u,s)-λ2(u,s))^2 for (i,u) in enumerate(v.items))
end

s
segids

centroid(L.segments[1])

function quadint(f::Function, W::Box; nd=50)
    boxes = PointProcessLearning.partitions(W,nd)
    d = PointSet([centroid(box) for box in boxes])
    a = Meshes.measure(W)/(nd*nd)
    return sum(f(u)*a for u in d.items)
end

quadint(u->(exp(logλ(u,s,θ₀))-exp(logλ(u,s,θ₁)))^2,W)


θ₀
m = model_pp(θ₀,[4,0.5],[7,1.2],logλ,false)
θest, θvals, Lossmatrix = EstimateParamsPPL(m,s,W;p=0.9,k=10, nstep=20)
θest
θ₀
res = EstimateParamsPL_Logit(S,s,W;nd=200)
coef(res)

function PredictionError2(sT::PointSet,sV::PointSet,logλ::Function,b::Box;p=0.5)
    #println("Running dependent")
    λ(u,s) = exp(logλ(u,s))
    h(u,s) = (1/(1-p))*(1/λ(u,s))
    #h(u,s) = (1-p)*(1/λ(u,s))
    #h(u,s) = (1/p)*(1/λ(u,s))
    #h(u,s) = (1/p)*(1/λ(u,s))
    #score = sum( ( x∈b ? h(x,sT) : 0) for x in sV.items) -measure(b)
    score = sum( ( x∈b ? h(x, sT) : 0) for x in sV.items) -measure(b)
    return score
end
function Loss2(TVset::Vector{Tuple{PointSet{2, Float64}, PointSet{2, Float64}}},
    logλ::Function,b::Box,p::Real;independent=false)
    k = length(TVset)
    I = zeros(k)
    for i in 1:k
        sT, sV = TVset[i]
        I[i] = PredictionError2(sT,sV,logλ,b;p=p)
    end
    return mean((I[i])^2 for i in 1:k)
end
function EstimateParamsPPL(m::model_pp, s::PointSet, b::Box; p=0.9, k=200, nstep=10)
    # Create CV folds
    CVfolds = [TVsplit(s,p) for i in 1:k]
    # Loss function
    L(θ) = Loss2(CVfolds, (u,s)->m.logλ(u,s,θ),b,p; independent=m.independent)
    # Iterator of all combinations of parameters within ranges
    Δθ = (m.up .- m.lo)./nstep
    iter = Iterators.product([m.lo[i]:Δθ[i]:m.up[i] for i in eachindex(m.up)]...);
    θvals = vec(collect.(iter))
    Lossmatrix = L.(θvals)
    return (θvals[argmin(Lossmatrix)], θvals, Lossmatrix)
end
