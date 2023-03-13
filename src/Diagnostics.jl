"""
    PredictionErrorDep(sT,sV,logλ,b; p=0.5)
Compute prediction error of pointpatter sV by way of sT,
under model defined by (DEPENDENT) logλ (log of conditional intensity),
over window b, using independent thinning with prob. p.
"""
function PredictionErrorDep(sT::PointSet,sV::PointSet,logλ::Function,b::Box;p=0.5)
    #println("Running dependent")
    λ(u,s) = exp(logλ(u,s))
    h(u,s) = (1/(1-p))*(1/λ(u,s))
    #h(u,s) = (1/p)*(1/λ(u,s))
    #score = sum( ( x∈b ? h(x,sT) : 0) for x in sV.items) -measure(b)
    score = sum( ( x∈b ? h(x, (sT ∪ sV) \ PointSet([x])) : 0) for x in sV.items) -measure(b)
    return score
end
# function PredictionErrorDep(sT::PointSet,sV::PointSet,logλ::Function,b::Box;p=0.5)
#     println("Running dependent")
#     λ(u,s) = exp(logλ(u,s))
#     h(u,s) = ((1-p)/p)*(1/λ(u,s))
#     score = sum( ( x∈b ? h(x,sT) : 0) for x in sV.items) -measure(b)
#     return score
# end
"""
    predictionerrorIndep(sT,sV,logλ,b; p=0.5)
Compute prediction error of pointpatter sV by way of sT,
under model defined by (INDEPENDENT) logλ (log of conditional intensity),
over window b, using independent thinning with prob. p.
"""
function PredictionErrorIndep(sT::PointSet,sV::PointSet,logλ::Function,b::Box;p=0.5)
    #println("Running independent")
    λ(u,s) = exp(logλ(u,s))
    h(u,s) = (1/(1-p))*(1/λ(u,s))
    score = sum( ( x∈b ? h(x,sT\PointSet(x)) : 0) for x in sT.items) -measure(b)
    return score
end
"""
    PredictionError(sT,sV,logλ,b; p=0.5)
Compute prediction error of pointpatter sV by way of sT,
under model defined by logλ (log of conditional intensity),
over window b, using independent thinning with prob. p.
"""
PredictionError(sT::PointSet,sV::PointSet,
    logλ::Function,b::Box;p=0.5,independent=false) = (independent==true) ? PredictionErrorIndep(sT,sV,logλ,b;p=p) : PredictionErrorDep(sT,sV,logλ,b;p=p) 


PredictionError(sT::PointSet,sV::PointSet,
    m::model_pp,b::Box; p=0.5) = PredictionError(sT,sV,m.logλ,b;p=p,independent=m.independent)


"""
    StoyanGrabarnik(s,λ,b)
Compute the StoyanGrabarnik diagnostic for point pattern 's'
    using Papangelou conditional intensity λ, over box b.
"""
StoyanGrabarnik(s::PointSet,logλ::Function,b::Box) = PredictionErrorIndep(s,s,logλ,b;p=0)

#function StoyanGrabarnik(s::PointSet, logλ::Function, b::Box)
#    λ(u,s) = exp(logλ(u,s))
#    return sum( ( x∈b ? 1/λ(x,s) : 0) for x in s.items) -measure(b)
#end

"""
    PredictionErrorPlot(sT,sV,logλ,W,n;p=0.5)
Diagnostic plot of prediction errors of sV given sT in nxn partition of W.
"""
function PredictionErrorPlot(sT::PointSet,sV::PointSet,logλ::Function,W::Box,n::Int;p=0.5,independent=false)
    λ(u,s) = exp(logλ(u,s))
    if independent == true
        h1(u,s) = (1/(1-p))*(1/λ(u,s))
        m = [h1(x,sT) for x in sV.items]
    else
        h2(u,s) = ((1-p)/p)*(1/λ(u,s))
        m = [h2(x,sT) for x in sV.items]
    end
    ms_val = 2 .+ 13 .* (m .- maximum(m))./(maximum(m)-minimum(m))
    boxes = partitions(W,n)
    shapes = Box2Shape.(boxes)
    score = PredictionError(sT,sV,logλ,W;p=p,independent=independent)
    plt = plot(sT,W,axis=false, ticks=false, title=string(round(score,digits=3)))
    plot!(plt,sV,W, ms=ms_val, mc=:white, msc=:green, opacity=0.6)
    for i in 1:(n*n)
        score = PredictionError(sT,sV,logλ,boxes[i];p=p,independent=independent)
        col = get(ColorSchemes.coolwarm, -n*n*score + 0.5)
        plot!(plt,shapes[i], fillcolor=col, opacity=0.2)
        plot!(plt,shapes[i], fillcolor=false, linewidth=2, linecolor="black")
        center = coordinates(centroid(boxes[i]))
        annotate!(plt,center[1],center[2],text(round(score,digits=3), (score>=0) ? :blue : :red, :center, 10))
    end
    return plt
end





function StoyanGrabarnikPlot(s::PointSet,logλ::Function,W::Box,n::Int)
    λ(u,s) = exp(logλ(u,s))
    m = [1/λ(x,s) for x in s.items]
    ms_val = 2 .+ 13 .* (m .- maximum(m))./(maximum(m)-minimum(m))
    boxes = partitions(W,n)
    shapes = Box2Shape.(boxes)
    score = StoyanGrabarnik(s,logλ,W)
    plt = plot(s,W, ms=ms_val, mc=:white, msc=:black, opacity=0.6, msw=3, axis=false, ticks=false, title=string(round(score,digits=3)))
    for i in 1:(n*n)
        score = StoyanGrabarnik(s,logλ,boxes[i])
        col = get(ColorSchemes.coolwarm, -n*n*score + 0.5)
        plot!(plt,shapes[i], fillcolor=col, opacity=0.2)
        plot!(plt,shapes[i], fillcolor=false, linewidth=2, linecolor="black")
        center = coordinates(centroid(boxes[i]))
        annotate!(plt,center[1],center[2],text(round(score,digits=3), (score>=0) ? :blue : :red, :center, 10))
    end
    return plt
end


function StoyanGrabarnik(s::PointSet,logλ::Function,L::LinearNetwork,segids::Vector{Int64})
    #println("Running independent")
    λ(u,s) = exp(logλ(u,s))
    h(u,s) = 1/λ(u,s)
    score = 0
    for i in 1:length(L.segments)
        si = PointSet(s.items[segids.==i])
        if length(si.items)>0
            score = score + sum( h(x,si\PointSet(x)) for x in si.items) - Meshes.measure(L.segments[i])
        else
            score = score - Meshes.measure(L.segments[i])
        end
    end
    return score
end


function BermanTurnerIntegral(f::Function,W::Box; nd = 100)
    boxes = partitions(W,nd)
    Sum = 0
    for box in boxes
        sb = PointSet(filter(x->in(x,box),s.items))
        db = PointSet(Meshes.centroid(box))
        vb = sb ∪ db
        nb = length(vb.items)
        w = Meshes.measure(box)/nb
        Sum = Sum + sum( w*f(u) for v in vb.items)
    end
    return Sum
end






function MIE(s::PointSet,logλ1::Function,logλ2::Function,W::Box; nd = 100)
    boxes = partitions(W,nd)
    λ1(u,s) = exp(logλ1(u,s))
    λ2(u,s) = exp(logλ2(u,s))
    Sum = 0
    for box in boxes
        sb = PointSet(filter(x->in(x,box),s.items))
        db = PointSet(Meshes.centroid(box))
        vb = sb ∪ db
        nb = length(vb.items)
        w = Meshes.measure(box)/nb
        Sum = Sum + sum( w*(λ1(v,s)-λ2(v,s)) for v in vb.items)
    end
    return Sum
end

function MIE(s::PointSet,segids::Vector{Int64},logλ1::Function,logλ2::Function,L::LinearNetwork; nd = 100)
    segments = L.segments
    λ1(u,s) = exp(logλ1(u,s))
    λ2(u,s) = exp(logλ2(u,s))
    probs = Meshes.measure.(segments)./measure(L)
    Sum = 0
    for (i,seg) in enumerate(segments)
        sb = PointSet(s.items[segids .== i])
        db = PointSet([SamplePointOnSegment(seg) for _ in 1:ceil(Int64,nd*probs[i])])
        vb = sb ∪ db
        nb = length(vb.items)
        w = Meshes.measure(seg)/nb
        Sum = Sum + sum( w*(λ1(v,s)-λ2(v,s)) for v in vb.items)
    end
    return Sum
end


function MISE(s::PointSet,logλ1::Function,logλ2::Function,W::Box; nd = 100)
    boxes = partitions(W,nd)
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

function MISE(s1::PointSet,s2::PointSet,logλ1::Function,logλ2::Function,W::Box; nd = 100)
    boxes = partitions(W,nd)
    λ1(u,s) = exp(logλ1(u,s))
    λ2(u,s) = exp(logλ2(u,s))
    Sum = 0
    for box in boxes
        sb1 = PointSet(filter(x->in(x,box),s1.items))
        sb2 = PointSet(filter(x->in(x,box),s2.items))
        sb = sb1 ∪ sb2
        db = PointSet(Meshes.centroid(box))
        vb = sb ∪ db
        nb = length(vb.items)
        w = Meshes.measure(box)/nb
        Sum = Sum + sum( w*(λ1(v,s1)-λ2(v,s2))^2 for v in vb.items)
    end
    return Sum
end


function MISE(s::PointSet,segids::Vector{Int64},logλ1::Function,logλ2::Function,L::LinearNetwork; nd = 100)
    segments = L.segments
    λ1(u,s) = exp(logλ1(u,s))
    λ2(u,s) = exp(logλ2(u,s))
    probs = Meshes.measure.(segments)./measure(L)
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

