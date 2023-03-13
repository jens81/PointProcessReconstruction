#

struct LinearNetwork{V<:Vector{Point2},E<:Vector{Tuple{Int64,Int64}},A<:Matrix{Int64},S<:Vector{Segment}}
    vertices :: V
    edges :: E
    adjacencymatrix :: A
    segments :: S
end


function LinearNetwork(V,E)
    segments = Array{Segment}(undef,length(E))
    adjacencymatrix = zeros(Int64,(length(V),length(V)))
    for (i,e) in enumerate(E)
        segments[i] = Segment(V[e[1]],V[e[2]])
        adjacencymatrix[e[1],e[2]] = 1
        adjacencymatrix[e[2],e[1]] = 1
    end
    return LinearNetwork(V,E,adjacencymatrix,segments)
end

# Nope... need to redo this!!!
# It misses intersections for the second line crossing
function LinearNetwork(segments::Vector{Segment})
    E = Tuple{Int64,Int64}[]
    V = Point2[]
    for i in 1:length(segments)
        si = segments[i]
        v1,v2 = vertices(si)
        intpoints = Point2[]
        for j in i+1:length(segments)
            sj = segments[j]
            intpoint = si ∩ sj
            if !isnothing(intpoint)
                push!(intpoints,intpoint)
            end
        end
        sort!(intpoints, by = p->coordinates(p)[1])
        v1,v2 = sort(vertices(si), by = p->coordinates(p)[1])
        ncurr = length(V)
        nlength = length(intpoints) + 2
        #V = push!(V,v1,intpoints...,v2)
        V = V ∪ vcat(v1,intpoints...,v2)
        for i in ncurr+1 : ncurr+nlength-1
            E = push!(E,(i,i+1))
        end
    end
    return LinearNetwork(V,E)
end

measureL(L::LinearNetwork) = (length(L.segments)>0) ? sum(Meshes.measure.(L.segments)) : 0

function thin(L::LinearNetwork, retain::Real)
    V = L.vertices
    E = L.edges
    nedges = floor(Int64,length(E)*retain)
    E = E[Random.randperm(length(E))[1:nedges]]
    return LinearNetwork(V,E)
end

function RandomLines(n::Int64,W::Box)
    #lo, up = coordinates.(extrema(W))
    boxvertices = vertices(W)
    boxsegments = [Segment(boxvertices[i],boxvertices[i+1]) for i in 1:3]
    push!(boxsegments,Segment(boxvertices[1],boxvertices[4]))
    segments = Array{Segment}(undef,n)
    for i in 1:n
        bs = boxsegments[Random.randperm(4)[1:2]]
        v1 = SamplePointOnSegment(bs[1])
        v2 = SamplePointOnSegment(bs[2])
        segments[i] = Segment(v1,v2)
    end
    return segments
end

function subnetwork(L::LinearNetwork,b::Box)
    r0 = 0.001
    b = enlarge(b,r0)
    boxvertices = vertices(b)
    boxsegments = [Segment(boxvertices[i],boxvertices[i+1]) for i in 1:3]
    push!(boxsegments,Segment(boxvertices[1],boxvertices[4]))
    V = Point2[]
    E = Tuple{Int64,Int64}[]
    for i in 1:length(L.segments)
        s = L.segments[i]
        v = vertices(s)
        vinb = filter(v->in(v,b) ,vertices(s))
        if length(vinb) == 2
            push!(V,vertices(s)...)
            push!(E,(length(V)-1,length(V)))
        elseif length(vinb) == 1
            vinb = vinb[1]
            intpoints = [s ∩ sb for sb in boxsegments]
            intpoints = filter(p->!isnothing(p),intpoints)
            intp = intpoints[1]
            push!(V,vinb,intp)
            push!(E,(length(V)-1,length(V)))
        else
            intpoints = [s ∩ sb for sb in boxsegments]
            intpoints = filter(p->!isnothing(p),intpoints)
            if length(intpoints) == 2
                push!(V,intpoints...)
                push!(E,(length(V)-1,length(V)))
            end
        end
        #intpoints = [s ∩ sb for sb in boxsegments]
        #intpoints = filter(p->!isnothing(p),intpoints)
        #println(intpoints)
        #if length(intpoints) == 2
        #    push!(V,intpoints...)
        #    push!(E,(length(V)-1,length(V)))
        #elseif length(intpoints) == 1
        #    intp = intpoints[1]
        #    v = filter(v->in(v,b) ,vertices(s))[1]
        #    push!(V,v,intp)
        #    push!(E,(length(V)-1,length(V)))
        #end
    end
    return LinearNetwork(V,E)
end


function SamplePointOnSegment(s::Segment)
    return collect(sample(s,HomogeneousSampling(1)))[1]
end

function SamplePointsOnSegment(s::Segment,n)
    return collect(sample(s,HomogeneousSampling(n)))
end

function SamplePointOnNetwork(L::LinearNetwork)
    probs = Meshes.measure.(L.segments)./measureL(L)
    s = sample(L.segments, Weights(probs))
    point = SamplePointOnSegment(s)
    return point
end



function PlotLinearNetwork(L::LinearNetwork; lowc=(0.5,4,:black), show_vertices=false)
    if show_vertices
        pts = PointSet(L.vertices)
        W = boundingbox(pts)
        plt = plot(pts,W)
    else
        plt = plot(aspect_ratio=1, opacity=0.5)
    end
    for s in L.segments
        v = vertices(s)
        v1 = coordinates(v[1])
        v2 = coordinates(v[2])
        x = [v1[1],v2[1]]
        y = [v1[2],v2[2]]
        plot!(plt,x,y, label=false, line=lowc)
    end
    return plt
end

function PlotLinearNetwork(L::LinearNetwork,W::Box; lowc=(0.5,4,:black),show_vertices=false)
    lo, up = coordinates.(extrema(W))
    if show_vertices
        pts = PointSet(L.vertices)
        W = boundingbox(pts)
        plt = plot(pts,W)
    else
        plt = plot(aspect_ratio=1, opacity=0.5)
    end
    for s in L.segments
        v = vertices(s)
        v1 = coordinates(v[1])
        v2 = coordinates(v[2])
        x = [v1[1],v2[1]]
        y = [v1[2],v2[2]]
        plot!(plt,x,y, label=false, line=lowc, xlim=(lo[1],up[1]), ylim=(lo[2],up[2]))
    end
    return plt
end

function PlotLinearNetwork!(plt,L::LinearNetwork,W::Box; lowc= (0.5,4,:black),show_vertices=false)
    lo, up = coordinates.(extrema(W))
    if show_vertices
        pts = PointSet(L.vertices)
        W = boundingbox(pts)
        plot!(plt,pts,W)
    end
    for s in L.segments
        v = vertices(s)
        v1 = coordinates(v[1])
        v2 = coordinates(v[2])
        x = [v1[1],v2[1]]
        y = [v1[2],v2[2]]
        plot!(plt,x,y, label=false, line=lowc, xlim=(lo[1],up[1]), ylim=(lo[2],up[2]))
    end
    return plt
end


# Recipe for plotting line segments
@recipe function f(s::Segment)
    v = vertices(s)
    v1 = coordinates(v[1])
    v2 = coordinates(v[2])
    x = [v1[1],v2[1]]
    y = [v1[2],v2[2]]
    #y = map(u -> (coordinates(u)[1],coordinates(u)[2]), s.items)
    # Settings
    seriestype --> :line  
    legend --> false         
    #linecolor --> :gray
    aspectratio --> 1.0
    linewidth --> 3
    linealpha --> 0.5
    return x,y
end

# Recipe for plotting line segments
@recipe function f(s::Segment,W::Box)
    v = vertices(s)
    v1 = coordinates(v[1])
    v2 = coordinates(v[2])
    x = [v1[1],v2[1]]
    y = [v1[2],v2[2]]
    #y = map(u -> (coordinates(u)[1],coordinates(u)[2]), s.items)
    # Settings
    lo, up = coordinates.(extrema(W))
    xlim --> (lo[1],up[1])
    ylim --> (lo[2],up[2])
    seriestype --> :line  
    legend --> false         
    linecolor --> :gray
    aspectratio --> 1.0
    linewidth --> 3
    linealpha --> 0.5
    return x,y
end