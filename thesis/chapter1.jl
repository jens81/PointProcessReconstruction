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
Wplus = enlarge(W,0.1)

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
s = sample_pp(p,W)
plot(s,W)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Chalmers University of Technology_ Master Thesis Template 2021/figure/chapter1/pointset1.pdf")

boxes = PointProcessLearning.partitions(W,6)
counts = [N(s,box) for box in boxes]
col = map(x->get(ColorSchemes.OrRd,x),counts./maximum(counts))
centers = coordinates.(centroid.(boxes))
plt = plot(s,W)
plot!(plt,PointProcessLearning.Box2Shape.(boxes), linestyle=:dash, fillcolor=false)
plt
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Chalmers University of Technology_ Master Thesis Template 2021/figure/chapter1/pointset2.pdf")


boxes = PointProcessLearning.partitions(W,6)
counts = [N(s,box) for box in boxes]
col = map(x->get(ColorSchemes.OrRd,x),counts./maximum(counts))
centers = coordinates.(centroid.(boxes))
plt = plot(s,W)
plot!(plt,PointProcessLearning.Box2Shape.(boxes), linestyle=:dash, fillcolor=col, alpha=1)
for i in eachindex(boxes)
    center = centers[i]
    annotate!(plt,center[1],center[2],text(counts[i], :black, :center, 12))
end
plt
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Chalmers University of Technology_ Master Thesis Template 2021/figure/chapter1/pointset3.pdf")


srec = PointProcessLearning.reconstruct(s,W,6)
plt = plot(srec,W)
plot!(plt,PointProcessLearning.Box2Shape.(boxes), linestyle=:dash, fillcolor=false)
plt
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Chalmers University of Technology_ Master Thesis Template 2021/figure/chapter1/pointset4.pdf")

plot(srec,W)
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/Thesis/Chalmers University of Technology_ Master Thesis Template 2021/figure/chapter1/pointset5.pdf")
