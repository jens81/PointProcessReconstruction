"""
    reconstruct(s,W,n)
A function to reconstruct a point pattern s, over window W using a partition of size nxn
"""
function reconstruct(s::PointSet,W::Box,n::Int)
    boxes = partitions(W,n)
    snew = EmptyPointSet()
    n = N(s,boxes[1])
    for b in boxes
        n = N(s,b)
        if n>0
            snew = snew ∪ sample_pp(BinomialProcess(n),b)
        end
    end
    return snew
end

"""
    reconstruct(s,L,W,n)
A function to reconstruct a point pattern s, on a Linear Network inside a window W using a partition of size nxn
"""
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

import Hungarian
function EMD(s1items::Array{Point2},s2items::Array{Point2})
    @assert (length(s1items)==length(s2items)) "Point sets must have the same length"
    len = length(s1items)
    if len>0
        distmat = Matrix{Float64}(undef,(len,len))
        for i in 1:len, j in 1:len
            distmat[i,j] = dist(s1items[i],s2items[j])
        end
        assignment, cost = Hungarian.hungarian(distmat)
    else
        cost = 0
    end
    return cost
end


# function EMD(s1items::Array{Point2},s2items::Array{Point2})
#     @assert (length(s1items)==length(s2items)) "Point sets must have the same length"
#     len = length(s1items)
#     distmat = Matrix{Float64}(undef,(len,len))
#     for i in 1:len, j in 1:len
#         distmat[i,j] = dist(s1items[i],s2items[j])
#     end
#     if len>0
#         val = Inf
#         for j in permutations(1:len)
#             #newval = sum(dist(s1items[i],s2items[j[i]]) for i in 1:len)
#             newval = sum(distmat[i,j[i]] for i in 1:len)
#             val = (newval<val) ? newval : val
#         end
#     else
#         val = 0
#     end
#     return val
# end

function EMDreconstructed(s1::PointSet,s2::PointSet,W::Box,n::Int)
    boxes = partitions(W,n)
    val = 0
    for box in boxes
        s1box = filter(x-> (x∈box), s1.items)
        s2box = filter(x-> (x∈box), s2.items)
        val = val + EMD(s1box,s2box)
    end
    return val
end


struct ReconstructedPointProcess{Counts<:Vector{Int64},Boxes<:Vector{Box{2, Float64}},P<:GibbsProcess} <: PointProcess
    counts :: Counts
    boxes :: Boxes
    p :: P
end

# Birth death move
function sample_pp2(rng::Random.AbstractRNG,
    prec::ReconstructedPointProcess,
    b::Box; niter=10_000, progress=false)

    # Number of points
    N(S) = length(S.items)

    # probabilities
    logλ(u,S) = N(S)>0 ? prec.p.logλ(u,S) : 0
    rmove(u,ξ,S₋) = exp(logλ(u,S₋)-logλ(ξ,S₋))/(N(S₋)+1)

    # Initial sample
    Svec = [prec.counts[i]>0 ? sample_pp(BinomialProcess(prec.counts[i]),prec.boxes[i]) : EmptyPointSet() for i in eachindex(prec.counts)]
    S = PointSet(union(Svec...))
    #println("Performing ",niter," steps")
    for m in 1:niter
        if progress
            if mod(m,100)==0
                println("step ",m)
            end
        end
        Svecprop = deepcopy(Svec)
        Sprop = deepcopy(S)
        # Randomly select box
        bind = rand(1:length(boxes))
        # region configuration
        lo, up = coordinates.(extrema(boxes[bind]))
        V = measure(boxes[bind])
        U = product_distribution([Uniform(lo[i], up[i]) for i in 1:embeddim(boxes[bind])])
        if  counts[bind]>0 #rand() < pmove
            # Randomly select and move point
            i = rand(1:counts[bind])
            ξ = Svecprop[bind].items[i]
            deleteat!(Svecprop[bind].items,i)
            Sprop = PointSet(union(Svecprop...))
            u = Point(rand(U))
            # accept/reject move
            if rand()<rmove(u,ξ,Sprop)
                Svec[bind] = Svecprop[bind] ∪ u
                S = PointSet(union(Svec...))
            end
        end
    end
    return S
end
